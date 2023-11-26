import matplotlib.pyplot as plt
import numpy as np
from nltk.tree import Tree
import stanza
import torch
from itertools import islice
from ldm.util import instantiate_from_config
import os
import sng_parser
from PIL import Image
from omegaconf import OmegaConf
import torch
from typing import Dict, List, Tuple
from einops import rearrange
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', verbose=False, use_gpu=True)

def preprocess_prompts(prompts):
    if isinstance(prompts, (list, tuple)):
        return [p.lower().strip().strip(".").strip() for p in prompts]
    elif isinstance(prompts, str):
        return prompts.lower().strip().strip(".").strip()
    else:
        raise NotImplementedError


def get_all_nps(tree, full_sent, tokens=None, highest_only=False, lowest_only=False):
    start = 0
    end = len(tree.leaves())

    idx_map = get_token_alignment_map(tree, tokens)

    def get_sub_nps(tree, left, right):
        if isinstance(tree, str) or len(tree.leaves()) == 1:
            return []
        sub_nps = []
        n_leaves = len(tree.leaves())
        n_subtree_leaves = [len(t.leaves()) for t in tree]
        offset = np.cumsum([0] + n_subtree_leaves)[:len(n_subtree_leaves)]
        assert right - left == n_leaves
        if tree.label() == 'NP' and n_leaves > 1:
            sub_nps.append([" ".join(tree.leaves()), (int(min(idx_map[left])), int(min(idx_map[right])))])
            if highest_only and sub_nps[-1][0] != full_sent: return sub_nps
        for i, subtree in enumerate(tree):
            sub_nps += get_sub_nps(subtree, left=left+offset[i], right=left+offset[i]+n_subtree_leaves[i])
        return sub_nps
    
    all_nps = get_sub_nps(tree, left=start, right=end)
    lowest_nps = []
    for i in range(len(all_nps)):
        span = all_nps[i][1]
        lowest = True
        for j in range(len(all_nps)):
            if i == j: continue
            span2 = all_nps[j][1]
            if span2[0] >= span[0] and span2[1] <= span[1]:
                lowest = False
                break
        if lowest:
            lowest_nps.append(all_nps[i])

    if lowest_only:
        all_nps = lowest_nps

    if len(all_nps) == 0:
        all_nps = []
        spans = []
    else:
        all_nps, spans = map(list, zip(*all_nps))
    if full_sent not in all_nps:
        all_nps = [full_sent] + all_nps
        spans = [(min(idx_map[start]), min(idx_map[end]))] + spans

    return all_nps, spans, lowest_nps


def get_token_alignment_map(tree, tokens):
    if tokens is None:
        return {i:[i] for i in range(len(tree.leaves())+1)}
        
    def get_token(token):
        return token[:-4] if token.endswith("</w>") else token

    idx_map = {}
    j = 0
    max_offset = np.abs(len(tokens) - len(tree.leaves()))
    mytree_prev_leaf = ""
    for i, w in enumerate(tree.leaves()):
        token = get_token(tokens[j])
        idx_map[i] = [j]
        if token == mytree_prev_leaf+w:
            mytree_prev_leaf = ""
            j += 1
        else:
            if len(token) < len(w):
                prev = ""
                while prev + token != w:
                    prev += token
                    j += 1
                    token = get_token(tokens[j])
                    idx_map[i].append(j)
                    # assert j - i <= max_offset
            else:
                mytree_prev_leaf += w
                j -= 1
            j += 1
    idx_map[i+1] = [j]
    return idx_map


def get_all_spans_from_scene_graph(caption):
    caption = caption.strip()
    graph = sng_parser.parse(caption)
    nps = []
    spans = []
    words = caption.split()
    for e in graph['entities']:
        start, end = e['span_bounds']
        if e['span'] == caption: continue
        if end-start == 1: continue
        nps.append(e['span'])
        spans.append(e['span_bounds'])

    for r in graph['relations']:
        start1, end1 = graph['entities'][r['subject']]['span_bounds']
        start2, end2 = graph['entities'][r['object']]['span_bounds']
        start = min(start1, start2)
        end = max(end1, end2)
        if " ".join(words[start:end]) == caption: continue
        nps.append(" ".join(words[start:end]))
        spans.append((start, end))
    
    return [caption] + nps, [(0, len(words))] + spans, None


def single_align(main_seq, seqs, spans, dim=1):
    main_seq = main_seq.transpose(0, dim)
    for seq, span in zip(seqs, spans):
        seq = seq.transpose(0, dim)
        start, end = span[0]+1, span[1]+1
        seg_length = end - start
        main_seq[start:end] = seq[1:1+seg_length]

    return main_seq.transpose(0, dim)


def multi_align(main_seq, seq, span, dim=1):
    seq = seq.transpose(0, dim)
    main_seq = main_seq.transpose(0, dim)
    start, end = span[0]+1, span[1]+1
    seg_length = end - start
    main_seq[start:end] = seq[1:1+seg_length]#+1 for <start> token

    return main_seq.transpose(0, dim)


def align_sequence(main_seq, seqs, spans, dim=1, single=False):
    aligned_seqs = []
    if single:
        return [single_align(main_seq, seqs, spans, dim=dim)]
    else:
        for seq, span in zip(seqs, spans):
            aligned_seqs.append(multi_align(main_seq.clone(), seq, span, dim=dim))
        return aligned_seqs


def chunk(it, size):
    it = iter(it)
    #state in maintained within it object
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_model_wrapper(ckpt:str, config:str, device="cuda"):
    
    config = OmegaConf.load(f"{config}")
    model = load_model_from_config(config, f"{ckpt}")
    model = model.to(device)

    return model


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def make_dir(outpath, folder_name, overwrite=True):
    sample_path = os.path.join(outpath, folder_name)
    os.makedirs(sample_path, exist_ok=True)
    if overwrite:
        base_count = 0
    else:
        base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    return sample_path, base_count, grid_count


def get_word_inds(text: str, word_place: int or str or list, tokenizer):
    """
    Get the indices of the words in the embedding
    """
    if "apple" in text:
        pause = True
    else:
        pause= False
    
    #NOTE: A bug in the original code!!!!
    #split punctuations in each word since tokenizer separates punc from nouns
    import re
    punc_split_text = re.split("([.,!?\"':;)(])", text.strip())
    #remove last punctuation split
    if punc_split_text[-1] == "":
        punc_split_text = punc_split_text[:-1]

    split_text = []
    for item in punc_split_text:
        split_text += item.strip().split(" ")

        
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0
 
        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1) # record all token positions of a word
            if cur_len >= len(split_text[ptr]): # move to next word
                ptr += 1
                cur_len = 0

    return np.array(out)


def get_tokenized_seq(text: str, tokenizer):
    encode_str = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)]
    if encode_str[0] == "<|startoftext|>":
        encode_str = encode_str[1:-1]
    return encode_str


def attn_map_analysis(
                    attn_maps: Dict[str, List[List[torch.Tensor]]],
                    noun_idx,
                    prompt,
                    tokenizer,
                    option="dot",
                    skip_anchor=False,
                    normalize=False,
                    ):
    """
    Compute the overlap between the attention map of the anchor token and the attention map of each token in the prompt.
    """
    
    #get the tokenized prompt
    prompt = get_tokenized_seq(prompt, tokenizer)
    overlaps = defaultdict(list)

    assert option in ["dot", "cross_entropy"]

    #compute overlap with each token
    for layer_name in attn_maps.keys():
        #not -1 because of <start> token
        attn_map_anchor = attn_maps[layer_name][0][0].squeeze()[:, :, :, noun_idx].squeeze()# (num_heads, h, w) or (num_heads, h, w, num_tokens)

        assert len(attn_map_anchor.shape) == 3 or len(attn_map_anchor.shape) == 4
        if len(attn_map_anchor.shape) == 4:
            print(f"noun in prompt \"{prompt}\" tokenized into multiple tokens!")
            attn_map_anchor = attn_map_anchor.mean(dim=-1)

        num_skip = 0
        #get attn map of each token
        for idx in range(len(prompt)):
            if (idx in noun_idx and skip_anchor):
                # num_skip -= 1
                continue
        
            #shape: (batch_size(1), num_heads, h, w, seq_len)
            attn_map = attn_maps[layer_name][0][0]
            attn_map_i = attn_map.squeeze()[:, :, :, idx].squeeze()# (num_heads, h, w)

            if len(attn_map_i.shape) == 4 :
                print(f"noun prompt \"{prompt}\" tokenized into multiple tokens!")
                # num_skip = attn_map_i.shape[-1] - 1
                attn_map_anchor = attn_map_anchor.mean(dim=-1)

            attn_map_i = attn_map_i.flatten(start_dim=1) # (num_heads, h*w)
            attn_map_anchor = attn_map_anchor.flatten(start_dim=1) # (num_heads, h*w)
            if option == "dot":
                
                spatial_dim = attn_map_anchor.shape[1] #h*w
                #column probabilities sum to 1, distributing each token to a location
                #(num_heads, h*w) * (num_heads, h*w, 1) -> (num_heads, 1, 1)
                try:
                    overlap = torch.bmm(attn_map_anchor.unsqueeze(1), attn_map_i.unsqueeze(-1)).squeeze()
                except:
                    print("Debug!!!!!!")
                    breakpoint()
                if normalize:
                    overlap = overlap / spatial_dim
                # overlap = torch.enisum("hs,hs->h", attn_map_anchor, attn_map_i) / spatial_dim

                #average across all heads
                overlap = torch.mean(overlap)
            elif option == "cross_entropy":
                overlap = F.cross_entropy(attn_map_i, attn_map_anchor, reduction="mean")

            overlaps[layer_name].append(overlap)

    return overlaps


