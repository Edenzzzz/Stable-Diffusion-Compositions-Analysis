import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from PIL import Image
from typing import Union, Tuple, List, Dict
from txt2img_utils import get_tokenized_seq

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


def show_overlap(gs,
                overlaps:Dict[str, Dict[str, List[float]]],
                noun_indices: List[int],
                option,
                tokenizer
                ):
    """
    Visulize overlap between the attention maps corresponding to different tokens.
    Args:
        overlaps: {prompt: {layer: [overlap]}}
    """

    plt.suptitle(option)
    #average the layer dimension
    for prompt_idx, prompt in enumerate(overlaps.keys()):
        try:
            anchor_token = get_tokenized_seq(prompt, tokenizer)[noun_indices[prompt_idx][0] - 1].strip() #-1 for <start> token
        except:
            print("Debug!]")
            breakpoint()
        tokens = get_tokenized_seq(prompt, tokenizer)
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
