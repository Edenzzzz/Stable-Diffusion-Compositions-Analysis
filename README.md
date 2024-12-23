# Structured Diffusion Guidance (ICLR 2023)
## We propose a method to fuse language structures into diffusion guidance for compositionality text-to-image generation.

### [Project Page](https://weixi-feng.github.io/structure-diffusion-guidance/) | [arxiv](https://arxiv.org/abs/2212.05032) | [OpenReview](https://openreview.net/forum?id=PUIqjT4rzq7)
<!-- [![][colab]][composable-demo] [![][huggingface]][huggingface-demo] -->

This is the official codebase for **Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis**.

[Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis](https://weixi-feng.github.io/structure-diffusion-guidance/)
  <br>
    [Weixi Feng](https://weixi-feng.github.io/) <sup>1</sup>,
    [Xuehai He](https://scholar.google.com/citations?user=kDzxOzUAAAAJ&) <sup>2</sup>,
    [Tsu-Jui Fu](https://tsujuifu.github.io/)<sup>1</sup>,
    [Varun Jampani](https://varunjampani.github.io/)<sup>3</sup>,
    [Arjun Akula](https://www.arjunakula.com/)<sup>3</sup>,
    [Pradyumna Narayana](https://scholar.google.com/citations?user=BV2dbjEAAAAJ&)<sup>3</sup>,
    [Sugato Basu](https://sites.google.com/site/sugatobasu/)<sup>3</sup>,
    [Xin Eric Wang](https://eric-xw.github.io/)<sup>2</sup>,
    [William Yang Wang](https://sites.cs.ucsb.edu/~william/) <sup>1</sup>
    <br>
    <sup>1</sup>UCSB, <sup>2</sup>UCSC, <sup>3</sup>Google
    <br>

## Update:
Apr. 4th: updated links, uploaded benchmarks and GLIP eval scripts, updated bibtex.

## Setup

Clone this repository and then create a conda environment with:
```
conda env create -f environment.yaml
conda activate structure_diffusion
```
If you already have a [stable diffusion](https://github.com/CompVis/stable-diffusion/) environment, you can run the following commands:
```
pip install stanza nltk scenegraphparser tqdm matplotlib
pip install -e .
```

## Inference
This repository supports stable diffusion 1.4 for now. Please refer to the official [stable-diffusion](https://github.com/CompVis/stable-diffusion/#weights) repository to download the pre-trained model and put it under ```models/ldm/stable-diffusion-v1/```. 
Our method is training-free and can be applied to the trained stable diffusion checkpoint directly.

To generate an image, run
```
python scripts/txt2img_eval.py --prompt "A red teddy bear in a christmas hat sitting next to a glass" --plms --parser_type constituency
```

By default, the guidance scale is set to 7.5 and output image size is 512x512. We only support PLMS sampling and batch size equals to 1 for now. 
Apart from the default arguments from [Stable Diffusion](https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/scripts/txt2img.py), we add ```--parser_type``` and ```--conjunction```.

```
usage: txt2img_eval.py [-h] [--prompt [PROMPT]] ...
                       [--parser_type {constituency,scene_graph}] [--conjunction] [--save_attn_maps]

optional arguments:
    ...
  --parser_type {constituency,scene_graph}
  --conjunction         If True, the input prompt is a conjunction of two concepts like "A and B"
  --save_attn_maps      If True, the attention maps will be saved as a .pth file with the name same as the image
```

Without specifying the ```conjunction``` argument, the model applies one ```key``` and multiple ```values``` for each cross-attention layer.
For concept conjunction prompts, you can run:
```
python scripts/txt2img_eval.py --prompt "A red car and a white sheep" --plms --parser_type constituency --conjunction
```

Overall, compositional prompts remains a challenge for Stable Diffusion v1.4. It may still take several attempts to get a correct image with our method. 
The improvement is system-level instead of sample-level, and we are still looking for good evaluation metrics for compositional T2I synthesis. 
We observe less missing objects in [Stable Diffusion v2](https://github.com/Stability-AI/stablediffusion), and we are implementing our method on top of it as well. 
Please feel free to reach out for a discussion.

## Benchmarks
CC-500.txt: Concept Conjunction of two objects with different colors (line1-446).
ABC-6K.txt: ~6K attribute binding prompts collected and created from COCO captions. 

## GLIP Eval
For our GLIP eval, please first clone and setup your environment according to the official [GLIP](https://github.com/microsoft/GLIP) repo and download the model checkpoint(s). Then refer to our ```GLIP_eval/eval.py``` and you may need to modify line 59&82. We assumed that each image file name contains the text prompt. 

## Comments
Our codebase builds heavily on [Stable Diffusion](https://github.com/CompVis/stable-diffusion). Thanks for open-sourcing!


## Citing our Paper

If you find our code or paper useful for your research, please consider citing
``` 
@inproceedings{feng2023trainingfree,
title={Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis},
author={Weixi Feng and Xuehai He and Tsu-Jui Fu and Varun Jampani and Arjun Reddy Akula and Pradyumna Narayana and Sugato Basu and Xin Eric Wang and William Yang Wang},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=PUIqjT4rzq7}
}
```
