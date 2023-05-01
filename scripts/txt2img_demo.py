#NOTE: To run the script
##################################################
#To read a single prompt from the command line:
#python scripts/txt2img_demo.py --prompt "A red teddy bear in a christmas hat sitting next to a glass" --scheduler dpm --denoise_steps 25 --parser_type constituency 
##################################################
#To read multiple prompts from a csv file:
#python scripts/txt2img_demo.py --from_file prompts.csv --parser_type constituency --scheduler dpm --compare True --denoise_steps 25 --save_v_matrix True
##################################################
#(::)
import argparse, os, sys, glob
from collections import defaultdict
from ossaudiodev import SNDCTL_SEQ_CTRLRATE
from ast import parse
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from matplotlib import gridspec
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from structured_stable_diffusion.models.diffusion.ddim import DDIMSampler
from structured_stable_diffusion.models.diffusion.plms import PLMSSampler
from structured_stable_diffusion.models.diffusion.dpm_solver import DPMSolverSampler
from scripts.txt2img_utils import *
import cv2
import matplotlib.pyplot as plt
import logging 
logging.basicConfig(level=logging.ERROR)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="A room with blue walls and a white sink",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--scheduler",
        choices=["plms", "ddim", "dpm"],
        default="dpm",
        help="choose denoising scheduler. Should use dpm solver ++ for best speed ",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from_file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--parser_type",
        type=str,
        choices=['constituency', 'scene_graph'],
        default='constituency'
    )
    parser.add_argument(
        "--conjunction",
        action='store_true',
        help='If True, the input prompt is a conjunction of two concepts like "A and B"'
    )
    parser.add_argument(
        "--save_attn_maps",
        default='False',
        type=eval,
        help='If True, the attention maps will be saved as a .pth file with the name same as the image'
    )

    parser.add_argument(
        "--save_v_matrix",
        default='False',
        help="whether to save the value matrices"
    )

    parser.add_argument(
        "--compare",
        default="True",
        help="use both vanilla and modified value matrix and visualize the difference using a grid"
    )

    opt = parser.parse_args()

    if opt.save_v_matrix:
        print("Saving value matrices")
    if opt.compare:
        print("Comparing vanilla and modified value matrices using method in Structured diffusion paper")
    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"
    if opt.from_file:
        opt.outdir = os.path.join(opt.outdir, os.path.basename(opt.from_file).split(".")[0])

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir


    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
        opt.from_file = ""
    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            import pandas as pd
            df = pd.read_csv(opt.from_file)
            data = df["prompts"].tolist()
            noun_indices = df["noun_idx"].tolist()
            try:
                opt.end_idx = len(data) if opt.end_idx == -1 else opt.end_idx
                data = data[:opt.end_idx]
                data, filenames = zip(*[d.strip("\n").split("\t") for d in data])
                data = list(chunk(data, batch_size))
            except:
                data = [batch_size * [d] for d in data]
            noun_indices = [get_word_inds(data[idx][0], item, model.cond_stage_model.tokenizer) for idx, item in enumerate(noun_indices)]

    

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    #apply method in the paper to value matrix, key matrix and try adding gaussian noise
    #NOTE: DO NOT change the format of these names 
    options = ["vanilla"] if not opt.compare \
        else [ 
                "struct_value", "struct_key", 
                "struct_value_key",
                "vanilla",
                ## perturb with gaussian noise with std = original_std * strength 
                "gauss_perturb_value_0.1", "gauss_perturb_value_0.2", "gauss_perturb_value_0.4", "gauss_perturb_value_0.7",
                "gauss_perturb_key_0.1", "gauss_perturb_key_0.2", "gauss_perturb_key_0.4", "gauss_perturb_key_0.7",
            ]
    
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for idx, option in enumerate(options):
                    
                    #@Wenxuan
                    save_attn_maps = opt.save_attn_maps if "struct" in option else False
                    if opt.scheduler == "plms":
                        sampler = PLMSSampler(model, option=option, save_attn_maps=save_attn_maps, noun_idx=noun_indices)
                    elif opt.scheduler == "ddim":
                        sampler = DDIMSampler(model, option=option, save_attn_maps=save_attn_maps, noun_idx=noun_indices)
                    else:
                        sampler = DPMSolverSampler(model, option=option, save_attn_maps=save_attn_maps, noun_idx=noun_indices)


                    sample_path, base_count, grid_count = make_dir(outpath, option)
                    all_samples = list()
                    for n in trange(opt.n_iter, desc="Sampling"):
                        for prompt_idx, prompts in enumerate(tqdm(data, desc="data")):
                            prompts = preprocess_prompts(prompts)
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])

                            c = model.get_learned_conditioning(prompts)

                            if opt.parser_type == 'constituency':
                                doc = nlp(prompts[0])
                                mytree = Tree.fromstring(str(doc.sentences[0].constituency))
                                tokens = model.cond_stage_model.tokenizer.tokenize(prompts[0])
                                nps, spans, noun_chunk = get_all_nps(mytree, prompts[0], tokens)
                            elif opt.parser_type == 'scene_graph':
                                nps, spans, noun_chunk = get_all_spans_from_scene_graph(prompts[0].split("\t")[0])
                            else:
                                raise NotImplementedError
                            
                            nps = [[np]*len(prompts) for np in nps]

                            if "struct" not in option:
                                print(f"\nUsing vanilla value matrix with option {option}")
                                c = model.get_learned_conditioning(nps[0])

                            elif opt.conjunction:
                                print(f"\nUsing structure diffusion with conjunction with option {option}")
                                c = [model.get_learned_conditioning(np) for np in nps]
                                k_c = [c[0]] + align_sequence(c[0], c[1:], spans[1:])
                                v_c = align_sequence(c[0], c[1:], spans[1:], single=True)
                                c = {'k': k_c, 'v': v_c}
                                
                            else:
                                c = [model.get_learned_conditioning(np) for np in nps]
                                k_c = c[:1]
                                v_c = [c[0]] + align_sequence(c[0], c[1:], spans[1:])
                                c = {'k': k_c, 'v': v_c}
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            #only save when using structure diffusion
                            
                            samples_ddim, intermediates = sampler.sample(S=opt.denoise_steps,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code,
                                                            save_attn_maps=save_attn_maps,
                                                            option=option,
                                                            noun_idx=noun_indices[prompt_idx],
                                                            )

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            x_checked_image = x_samples_ddim

                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                            if not opt.skip_save:
                                for sid, x_sample in enumerate(x_checked_image_torch):
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))          
                                    try:
                                        count = prompt_idx * opt.n_samples + sid
                                        safe_filename = f"{n}-{count}-" + (filenames[count][:-4])[:150] + ".jpg"
                                    except:
                                        safe_filename = f"{base_count:05}-{n}-{prompts[0]}"[:100] + ".jpg"
                                    img.save(os.path.join(sample_path, f"{safe_filename}"))
                                    
                                    
                                    store_object = sampler if opt.scheduler != "dpm" else sampler.dpm_solver
                                    if save_attn_maps:
                                        path = os.path.join(sample_path, "attn_maps")
                                        os.makedirs(path, exist_ok=True)
                                        torch.save(store_object.attn_maps, os.path.join(path, f'{safe_filename}.pt'))
                                    if opt.save_v_matrix:
                                        path = os.path.join(sample_path, "value_matrices")
                                        os.makedirs(path, exist_ok=True)
                                        torch.save(store_object.v_matrix, os.path.join(path, f'{safe_filename}.pt'))
                                    base_count += 1  

                            if not opt.skip_grid:
                                all_samples.append(x_checked_image_torch)

                    if not opt.skip_grid:
                        if opt.compare:
                            grid = torch.stack(all_samples, 0)
                            grid = rearrange(grid, 'n b c h w -> (n b) h w c') * 255.
                            if "compare_grid" not in locals():
                                compare_grid = []
                            compare_grid.append(grid)

                            if len(compare_grid) == len(options) and opt.compare:
                                compare_grid = torch.cat(compare_grid)
                                #plot params
                                ncols = len(options)
                                nrows = compare_grid.shape[0] // ncols  
                                assert compare_grid.shape[0] % ncols == 0, "Error: Number of samples not the same for each option"
                                # generate indices showing images in parralel
                                indices = torch.arange(compare_grid.shape[0]).reshape(ncols, compare_grid.shape[0] // ncols).T
                                dpi = 100

                                #best way to eliminate margins
                                plt.figure(figsize=(ncols * opt.W // dpi, nrows * opt.H // dpi))
                                gs = gridspec.GridSpec(nrows, ncols,
                                                    wspace=0.0, hspace=0.0,
                                                    # top=1.-0.5 / (nrows+1), bottom=0.5 / (nrows+1), 
                                                    # left=0.5 / (ncols+1), right=1-0.5 / (ncols+1)
                                                    ) 
                                for i in range(nrows):
                                    for j in range(ncols):
                                        ax = plt.subplot(gs[i,j])
                                        if i == 0:
                                            ax.set_title(options[j], fontsize=19)
                                        ax.imshow(compare_grid[indices[i][j]].cpu().numpy().astype(np.uint8))
                                        ax.axis('off')
                                plt.savefig(os.path.join(outpath, f'compare_grid-{grid_count:04}.png'))
                                          
                                #NOTE: Can't eliminate margins between subplots this way 
                                # fig, axes = make_im_subplots(indices.shape[0], indices.shape[1], img_size=opt.W)
                                # for i in range(axes.shape[0]):
                                #     for j in range(axes.shape[1]):
                                #         axes[i][j].imshow(compare_grid[indices[i][j]].cpu().numpy().astype(np.uint8))
                                #         axes[i][j].axis('off')
                                # fig.savefig(os.path.join(outpath, f'compare_grid-{grid_count:04}.png'))
                                                                
                                grid_count += 1
                        else:
                            #save single row grid 
                            grid = torch.stack(all_samples, 0)
                            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                            grid = make_grid(grid, nrow=n_rows)

                            # to image
                            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                            img = Image.fromarray(grid.astype(np.uint8))
                            img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                            grid_count += 1

        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f" \nEnjoy.")


if __name__ == "__main__":
    main()
