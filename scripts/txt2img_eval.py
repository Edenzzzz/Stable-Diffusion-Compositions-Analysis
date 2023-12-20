#NOTE: To run the script
##################################################
#To read a single prompt from the command line:
#python scripts/txt2img_demo.py --prompt "A red teddy bear in a christmas hat sitting next to a glass" --scheduler dpm --denoise_steps 25 --parser_type constituency 
##################################################
#To read multiple prompts from a csv file:
#python scripts/txt2img_demo.py --from_file prompts/multi_obj_prompts.csv --parser_type constituency --scheduler dpm --compare True --denoise_steps 25
##################################################
#(::)
import argparse, os, sys, glob
from ossaudiodev import SNDCTL_SEQ_CTRLRATE
from ast import parse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

#add the parent dir to allow access to ldm package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if os.path.split(os.getcwd())[-1] == 'scripts':
    os.chdir('..')
    
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from scripts.txt2img_utils import *
from scripts.vis_utils import *
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
        default=True,
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
        default="models/ldm/stable-diffusion/sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
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
        help='If True, the attention maps will be saved as a .pt file with the name same as the image'
    )

    parser.add_argument(
        "--save_v_matrix",
        default='False',
        type=eval,
        help="whether to save the value matrices"
    )

    parser.add_argument(
        "--compare",
        default="False",
        type=eval,
        help="use both vanilla and modified value matrix and visualize the difference using a grid"
    )
    
    parser.add_argument(
        "--test_attn_overlaps",
        action="store_true",
        help="test the overlap between the attention maps of the two concepts"
    )
    
    parser.add_argument(
        "--no_grid",
        action="store_true",
        help="If True, the output images will not be stored as a grid"
                        )
    opt = parser.parse_args()

    if opt.save_v_matrix:
        print("Saving value matrices")
    if opt.compare:
        print("Comparing vanilla and modified value matrices using method in structure diffusion paper")
    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"
    if opt.from_file:
        opt.outdir = os.path.join(opt.outdir, os.path.split(opt.from_file)[-1].split(".")[0])
    
    if opt.seed:
        seed_everything(opt.seed)
    else:
        import warnings
        warnings.warn("Sampling without a seed. This will lead to non-reproducible results.")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model_wrapper(opt.ckpt, opt.config, device)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir


    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if opt.from_file:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            import pandas as pd
            df = pd.read_csv(opt.from_file)
            data = df["prompts"].tolist()
            #if multiple nouns annotated, split by ,
            if df["noun_idx"].dtype == object:
                multi_noun_indices = df["noun_idx"].apply(lambda x: int(x.split(",")))
                noun_indices = df["noun_idx"].apply(lambda x: int(x.split(",")[0]))
            else:
                noun_indices = df["noun_idx"].tolist()
        
            try:
                opt.end_idx = len(data) if opt.end_idx == -1 else opt.end_idx
                data = data[:opt.end_idx]
                data, filenames = zip(*[d.strip("\n").split("\t") for d in data])
                data = list(chunk(data, batch_size))
            except:
                data = [batch_size * [d] for d in data]
            # campus house
            #@Wenxuan extract noun indices after tokenizing
            tokenizer = model.cond_stage_model.tokenizer
            noun_indices = [get_word_inds(data[idx][0], item, tokenizer) for idx, item in enumerate(noun_indices)]
            max_seq_length = max([len(get_tokenized_seq(prompt[0], tokenizer)) for prompt in data])
            breakpoint()
    else:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
        opt.from_file = ""
        noun_indices = [None] * len(data)
        
    start_code = None
    if opt.test_attn_overlaps:
        dot_overlaps = defaultdict(dict)
        cross_entropy_overlaps = defaultdict(dict)
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    #compare different v2 models
    v2_models = [
        ("models/ldm/stable-diffusion/v2-1_512-ema-pruned.ckpt", "configs/stable-diffusion/v2-inference.yaml"),
        ## these two won't work for some reason
        # ("models/ldm/stable-diffusion/v2-1_768-v-ema.ckpt", "configs/stable-diffusion/v2-inference-v.yaml"),
        # ("models/ldm/stable-diffusion/v2-1_768-ema-pruned.ckpt", "configs/stable-diffusion/v2-inference.yaml"),
                 ]
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    #apply method in the paper to value matrix, key matrix and try adding gaussian noise
    #NOTE: DO NOT change the format of these names 
    
    options = [v2_models[0]] if not opt.compare \
        else [ 
                # "average_value", "average_key", 
                # "average_value_key",
                # "vanilla",
                v2_models[0],
                ## perturb with gaussian noise with std = original_std * strength 
                # "gauss_perturb_value_0.7", "gauss_perturb_value_1.0", "gauss_perturb_value_3", "gauss_perturb_value_5",
                # "gauss_perturb_key_0.7", "gauss_perturb_key_1.0", "gauss_perturb_key_3", "gauss_perturb_key_5",
            ]
    
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for idx, option in enumerate(options):
                    #choose model and config by option
                    if isinstance(option, tuple):
                        option, config = option
                        ckpt = option
                        # for plotting short title
                        options[idx] = os.path.split(option)[-1].split(".")[0] 
                        model = load_model_wrapper(ckpt, config, device)
                        tokenizer = model.cond_stage_model.tokenizer
                    elif "v2" in option:
                        #default best v2 
                        model = load_model_wrapper("models/ldm/stable-diffusion/v2-1_768-ema-pruned.ckpt", "configs/stable-diffusion/v2-inference.yaml")
                    
                    elif "model" not in locals():
                        #default v1.4 model
                        model = load_model_wrapper(opt.ckpt, opt.config, device)

                    #@Wenxuan
                    ###set up the sampler for each option
                    save_attn_maps = opt.save_attn_maps
                    if opt.scheduler == "plms":
                        sampler = PLMSSampler(model, option=option, save_attn_maps=save_attn_maps, noun_idx=noun_indices)
                    elif opt.scheduler == "ddim":
                        # sampler = DDIMSampler(model, option=option, save_attn_maps=save_attn_maps, noun_idx=noun_indices)
                        raise NotImplementedError("Haven't got time to modify DDIM for these experiments ")
                    else:
                        sampler = DPMSolverSampler(model, option=option, save_attn_mapsx=save_attn_maps, noun_idx=noun_indices)
                    
                    #set different save path for each option
                    sample_path, base_count, grid_count = make_dir(outpath, options[idx], overwrite=True)

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
                                tokens = tokenizer.tokenize(prompts[0])
                                nps, spans, noun_chunk = get_all_nps(mytree, prompts[0], tokens)
                            elif opt.parser_type == 'scene_graph':
                                nps, spans, noun_chunk = get_all_spans_from_scene_graph(prompts[0].split("\t")[0])
                            else:
                                raise NotImplementedError
                            
                            nps = [[np]*len(prompts) for np in nps]

                            if "average" not in option:
                                print(f"\nUsing vanilla value matrix with option {option}")
                                c = model.get_learned_conditioning(nps[0])

                            elif opt.conjunction:
                                print(f"\nUsing conjunction with option {option}")
                                c = [model.get_learned_conditioning(np) for np in nps]
                                k_c = [c[0]] + align_sequence(c[0], c[1:], spans[1:])
                                v_c = align_sequence(c[0], c[1:], spans[1:], single=True)
                                c = {'k': k_c, 'v': v_c}
                                
                            else:
                                print(f"\nUsing structure diffusion with option {option}")
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
                                                            seed=opt.seed,
                                                            )

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                            x_checked_image = x_samples_ddim
                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                            assert x_checked_image_torch.shape[0] == opt.n_samples

                            #save individual samples
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
                                
                                
                                if opt.test_attn_overlaps:
                                    overlap = attn_map_analysis(store_object.attn_maps, 
                                                noun_indices[prompt_idx],
                                                prompts[0],
                                                tokenizer,
                                                option = "dot"
                                    )
                                    dot_overlaps[prompts[0]] = overlap

                                    overlap = attn_map_analysis(store_object.attn_maps, 
                                                noun_indices[prompt_idx],
                                                prompts[0],
                                                tokenizer,
                                                option = "cross_entropy"
                                                )
                                    cross_entropy_overlaps[prompts[0]] = overlap
                                    

                                if opt.save_v_matrix:
                                    path = os.path.join(sample_path, "value_matrices")
                                    os.makedirs(path, exist_ok=True)
                                    torch.save(store_object.v_matrix, os.path.join(path, f'{safe_filename}.pt'))
                                base_count += 1  
                            all_samples.append(x_checked_image_torch)
                    

                    if opt.test_attn_overlaps:
                        gs = make_grid(nrows=len(data), ncols=max_seq_length, W=opt.W, H=opt.H)
                        show_overlap(gs, dot_overlaps, noun_indices, option="dot", tokenizer=tokenizer)
                        plt.savefig(os.path.join(outpath, f'{options[idx]}_overlaps_dot.jpg'))
                        plt.close()

                        gs = make_grid(nrows=len(data), ncols=max_seq_length, W=opt.W, H=opt.H)
                        show_overlap(gs, cross_entropy_overlaps, noun_indices, option="cross_entropy", tokenizer=tokenizer)
                        plt.savefig(os.path.join(outpath, f'{options[idx]}_overlaps_cross_entropy.jpg'))
                        plt.close()

                    
                    #save grid for model comparison
                    if not opt.no_grid:
                        if opt.compare:
                            grid = torch.stack(all_samples, 0)
                            grid = rearrange(grid, 'n b c h w -> (n b) h w c') * 255.
                            if "compare_grid" not in locals():
                                compare_grid = []
                            compare_grid.append(grid)

                            if len(compare_grid) == len(options):
                                compare_grid = torch.cat(compare_grid)
                                #plot params
                                ncols = len(options)
                                nrows = compare_grid.shape[0] // ncols  
                                assert compare_grid.shape[0] % ncols == 0, "Error: Number of samples not the same for each option"
                                # generate indices showing images in parralel
                                indices = torch.arange(compare_grid.shape[0]).reshape(ncols, compare_grid.shape[0] // ncols).T

                                #best way to eliminate margins
                                gs = make_grid(nrows, ncols, opt.W, opt.H)
                                
                                for i in range(nrows):
                                    for j in range(ncols):
                                        ax = plt.subplot(gs[i,j])
                                        if i == 0:
                                            title = ax.set_title(options[j], fontsize=20, c='r', pad=19)
                                        if j == 0:
                                            ax.text(0.5, 2, data[i][0], fontsize=16)

                                        ax.imshow(compare_grid[indices[i][j]].cpu().numpy().astype(np.uint8))
                                        ax.axis('off')
                                plt.savefig(os.path.join(outpath, f'compare_grid-{grid_count:04}.png'))
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
