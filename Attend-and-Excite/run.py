import pprint
from typing import List

import pyrallis
import torch
from PIL import Image
import pandas as pd
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
import tqdm
import os
from transformers import AutoTokenizer
import re, nltk

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', verbose=False, use_gpu=True)

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

def traverse_tree(tree, arg_dict):
        
    if type(tree) != nltk.tree.tree.Tree:
        return 
    elif tree.label() == "NP":
        # A noun phrase is reached
        noun_idx = None
        for subtree in tree:
            traverse_tree(subtree, arg_dict)
            if subtree.label() in ["NN", "NNS"]:
                # Found a noun, save its index
                noun_idx = arg_dict['token_idx']
            elif subtree.label() == "JJ":
                # Found an adjective, add its index
                adjs.append(str(arg_dict['token_idx']))
            # Increment token_idx for each word
            arg_dict['token_idx'] += 1
        
        # If a noun is found in this NP, create a group
        if noun_idx is not None:
            group = '|'.join(adjs) + '|' + str(noun_idx)
            arg_dict['groups'].append(group)
            breakpoint()
    # else:
    #     # If it's a composed phrase, recursively traverse
    #     traverse_tree(tree, arg_dict)
        



def parse_index_groups(txt_path='ABC-6K.txt', group_split_char='|'):
    prompts = open(txt_path, 'r').read().split('\n')
    association_groups = []

    for prompt in tqdm.tqdm(prompts, "Parsing prompts"):
        doc = nlp(prompt)  # Assuming 'nlp' is defined and works on 'prompt'
        tree = Tree.fromstring(str(doc.sentences[0].constituency))

        # Initialize argument dictionary
        arg_dict = {'token_idx': 0, 'anchor_idx': -1, 'adjs': [], 'groups': []}
        traverse_tree(tree[-1], arg_dict)

        # Join groups with commas and add the leading '|' for the first group
        formatted_groups = ','.join(arg_dict['groups'])

        association_groups.append(formatted_groups)
        breakpoint()
    # Create DataFrame and save to CSV
    df = pd.DataFrame({'prompts': prompts, 'association_idxs': association_groups})
    df.to_csv(txt_path[:-4] + '.csv', index=False)


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

    offset = 1  # skip <start> token
    prompts = [pair[0] for pair in pairs]
    groups = [[ [int(i) + offset for i in group_str.split(group_split_char)] for group_str in pair[1].split(',')] for pair in pairs]
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
                  ae_ratio: float = 0.7,
                  **kwargs
                ) -> Image.Image:
    height = kwargs.pop('height', 512)
    width = kwargs.pop('width', 512)
    
    if groups is not None:
        # Replace A&E's loss function with ours
        assert config.loss_type in ["l1", "cos", "wasserstein", "dc"], "Invalid loss type"
        
        #NOTE: Seems lower lr doesn't change anything...
        if config.loss_type == "cos":
            config.scale_factor = 10
            config.scale_range = (1.0, 0.2)
            # config.max_iter_to_alter += 5
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
                    ae_ratio=ae_ratio,
                    height=height,
                    width=width)
    image = outputs.images[0]
    return image


@pyrallis.wrap()
def main(config: RunConfig):

    stable = load_model(config)
    token_indices = []
    
    # Parse and label coco prompts
    # if not os.path.exists('ABC-6K.csv'):
    #     parse_index_groups(txt_path='ABC-6K.txt', group_split_char='|')
    
    # Load a list of prompts and indices if specified; otherwise just use the one prompt
    if config.prompt is not None:
        token_indices = [get_indices_to_alter(stable, config.prompt) if config.token_indices is None else config.token_indices]
        prompts = [config.prompt]
        groups = [None]
    elif config.prompt_csv is not None:
        prompts, groups, token_indices = read_associated_indices(path=config.prompt_csv)
    else:
        raise ValueError("Must specify either prompt or prompt_csv")

    
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
