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

def get_seq_encode(text: str, tokenizer):
    encode_str = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)]
    if encode_str[0] == "<|startoftext|>":
        encode_str = encode_str[1:-1]
    return encode_str

############################## Visualization ##############################
############################## Visualization ##############################
############################## Visualization ##############################

#NOTE: Can't eliminate margins???
def make_grid(*args, img_size, dpi=100, margin=0):
    """
    Make subplots for images
    @args : number of rows and columns
    """
    nrows, ncols = args  # number of rows and columns of subplots
    # subplot_size = (img_size - margin // 2) // dpi
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(img_size * nrows // dpi, img_size * ncols // dpi), dpi=dpi)
    
    # set size of each subplot
    for ax in axes.flat:
        ax.set_aspect('equal') # don't adjust x-y unit ratio
        ax.set_xticks([]) # no ticks on x-axis
        ax.set_yticks([]) # no ticks on y-axis

    # set titles for columns
    fig.text(0.25, 0.95, "Vanila", fontsize=14)
    fig.text(0.75, 0.95, "Modified", fontsize=14)
    
    fig.subplots_adjust(wspace=margin, hspace=margin)
    return fig, axes


def opencv_compare_grid(compare_grid, indices, folder_names, outpath, grid_count):
    import cv2
    from torchvision.utils import make_grid
    
    compare_grid = compare_grid[indices.flatten()]
    compare_grid = make_grid(compare_grid, nrow=2)
    compare_grid = compare_grid.cpu().numpy().astype(np.uint8)

    #add blank for title
    margin_grid = np.vstack([np.full((100, compare_grid.shape[1], 3), 255, dtype=np.uint8), compare_grid])
    cv2.putText(margin_grid, folder_names[0], (int(margin_grid.shape[1] * 0.1), 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (2, 255, 3), 3, cv2.LINE_AA)
    cv2.putText(margin_grid, folder_names[1], (int(margin_grid.shape[1] * 0.6), 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (2, 255, 3), 3, cv2.LINE_AA)
    img = Image.fromarray(margin_grid)
    img.save(os.path.join(outpath, f'compare_grid-{grid_count:04}.png'))


def make_grid(nrows, ncols, W=10, H=10, dpi=100):

    plt.figure(figsize=(ncols * W // dpi, nrows * H // dpi))
    gs = gridspec.GridSpec(nrows, ncols,
                        wspace=0.0, hspace=0.0,
                        top=1.-0.5 / (nrows+1), bottom=0.5 / (nrows+1), 
                        left=0.5 / (ncols+1), right=1-0.5 / (ncols+1)
                        ) 
    return gs



def attn_map_analysis(
                    attn_maps: Dict[str, List[List[torch.Tensor]]],
                    noun_idx,
                    prompt,
                    tokenizer,
                    option="dot",
                    skip_anchor=False,
                    normalize=False,
                    ):

    prompt = get_seq_encode(prompt, tokenizer)
    overlaps = defaultdict(list)

    assert option in ["dot", "cross_entropy"]

    #compute overlap with each token
    for layer_name in attn_maps.keys():
        #attn_maps[layer_name][0][0]: (batch_size(1), num_heads, h, w, seq_len)
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

def show_overlap(gs,
                overlaps:Dict[str, Dict[str, List[float]]],
                noun_indices: List[int],
                option,
                tokenizer
                ):
    """
    @overlaps: {prompt: {layer: [overlap]}}
    """
    plt.suptitle(option)
    #average the layer dimension
    for prompt_idx, prompt in enumerate(overlaps.keys()):
        try:
            anchor_token = get_seq_encode(prompt, tokenizer)[noun_indices[prompt_idx][0] - 1].strip() #-1 for <start> token
        except:
            print("Debug!]")
            breakpoint()
        tokens = get_seq_encode(prompt, tokenizer)
        num_tokens = len(tokens)

        
        for token_idx in range(num_tokens):
            overlap = []
            token = tokens[token_idx]
            #unflatten the layer dimension
            for layer in overlaps[prompt].keys():
                try:
                    overlap += [overlaps[prompt][layer][token_idx]]
                except:
                    breakpoint()
            #average over heads
            mean = np.mean(overlap)
            #visualize token-wise overlap across all layers
            ax = plt.subplot(gs[prompt_idx, token_idx])
            ax.hist(overlap, bins=6)
            ax.axvline(mean, color='orange', linestyle='dashed', linewidth=1)
            ax.text(mean*1.1, ax.get_ylim()[1]*0.9, 'Mean: {:.2f}'.format(mean))
            ax.set_title(f"{anchor_token} vs {token}")
