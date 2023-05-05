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
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

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
                    breakpoint()
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


def make_dir(outpath, folder_name):
    sample_path = os.path.join(outpath, folder_name)
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    return sample_path, base_count, grid_count


def get_word_inds(text: str, word_place: int or str or list, tokenizer):
    """
    Get the indices of the words in the embedding
    """
    split_text = text.split(" ")
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

############################## Visualization ##############################
############################## Visualization ##############################
############################## Visualization ##############################

#NOTE: Can't eliminate margins???
def make_im_subplots(*args, img_size, dpi=100, margin=0):
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

