import pprint
from typing import List

import pyrallis
import torch
from PIL import Image

from config import RunConfig
from pipeline_attend_and_excite import AttendAndExcitePipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import csv
from nltk.tree import Tree
import stanza

def load_model(config: RunConfig):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if config.sd_2_1:
        stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
    stable = AttendAndExcitePipeline.from_pretrained(stable_diffusion_version).to(device)
    return stable

'''
@Wenxuan 
'''
def parse_index_groups(csv_path='multi_obj_prompts_with_association.csv', group_split_char='|', shift_idxs=1):
    pass
'''
EDIT 
'''
def read_associated_indices(path='multi_obj_prompts_with_association.csv', group_split_char='|'):
    '''
    Inputs:
        path: path to file
        group_split_char: the character that separates tokens of an association group

    Returns groups, indices_to_alter
    groups:
        A list of lists of lists of int indices:
        1. Outermost list: Group association info for each prompt
        2. 2nd layer of list elements: for a given prompt, multiple groups of associations
        3. Innermost list: The indices in of words in the association group. The LAST element is the index of the 'anchor' word (usually the noun that the other words describe)  
        EX: Prompts: ["A red dog and blue cat", "A green ball"]
          2 prompts, anchor dog, cat, ball, with respective descriptor words of red, blue, green that form the groups
        Returns groups: [[ [1,2], [4,5] ], 
                         [ [1,2] ]
                        ]
    indices_to_alter:
        List of sorted, unique indices for each prompt. 
        Indices are of words that belong to a group and we need to retrieve/alter the attention map for it
        Returns indices:
            [[1,2,4,5],
             [1,2]]
    '''
    with open(path, mode ='r') as f:
        pairs=list(csv.reader(f))
    if pairs[0][0]=='prompts':
        pairs=pairs[1:]

    shift_idx = 1 # skip <start> token
    prompts = [pair[0] for pair in pairs]
    groups = [[ [int(i) + shift_idx for i in group_str.split(group_split_char)] for group_str in pair[1].split(',')] for pair in pairs]
    indices_to_alter = [ sorted([i for l2 in l1 for i in l2 ] ) for l1 in groups]

    return prompts, groups, indices_to_alter



def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def run_on_prompt(prompt: List[str],
                  model: AttendAndExcitePipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config: RunConfig,
                  groups: List[List[int]] = None, # EDIT
                  ae_ratio: float = 0.7
                ) -> Image.Image:
    if groups is not None:
        # Replace A&E's loss function with ours
        assert config.loss_type in ["l1", "cos", "wasserstein", "dc"], "Invalid loss type"
        
        #NOTE: Seems lower lr doesn't change anything...
        if config.loss_type == "cos":
            config.scale_factor = 20
            config.scale_range = (1.0, 0.5)
            config.max_iter_to_alter += 5
        elif config.loss_type == "dc":
            config.scale_factor = 30
            config.scale_range = (1.0, 0.3)
            config.max_iter_to_alter += 10
            
        print(f"Using {config.loss_type} loss with lr {config.scale_factor} and {ae_ratio} * A&E_loss + {round(1 - ae_ratio, 2)} * {config.loss_type}")
        
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    groups=groups, # EDIT
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1,
                    loss_type=config.loss_type,
                    ae_ratio=ae_ratio)
    image = outputs.images[0]
    return image


@pyrallis.wrap()
def main(config: RunConfig):

    stable = load_model(config)
    token_indices = []
    # Load a list of prompts and indices if specified; otherwise just use the one prompt
    if config.prompt_csv is not None:
        prompts, groups, token_indices = read_associated_indices(path=config.prompt_csv)
    else:
        token_indices = [get_indices_to_alter(stable, config.prompt) if config.token_indices is None else config.token_indices]
        prompts = [config.prompt]
        groups = [None]

    
    # images = []
    for i, prompt in enumerate(prompts):
        for seed in config.seeds:
            print(f"Seed: {seed}")
            g = torch.Generator('cuda').manual_seed(seed)
            controller = AttentionStore()
            image = run_on_prompt(prompt=prompt,
                                model=stable,
                                controller=controller,
                                token_indices=token_indices[i],
                                groups=groups[i],
                                seed=g,
                                config=config)
            
            subfolder = config.loss_type if groups[i] is not None else "A&E"
            prompt_output_path = config.output_path / subfolder
            prompt_output_path.mkdir(exist_ok=True, parents=True)
            image.save(prompt_output_path / f'seed={seed}_{prompt}.png')
            # images.append(image)

        # save a grid of results across all seeds
        # joined_image = vis_utils.get_image_grid(images)
        # joined_image.save(config.output_path / f'{prompt}.png')


if __name__ == '__main__':
    main()
