# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
import os
from PIL import Image

def img_loader(path):
    return Image.open(path)

def to_cuda(samples, targets, device):
    samples = samples.to(device, non_blocking=True)
    targets = [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
    for t in targets
]

    return samples, targets

class data_prefetcher():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for k, v in t.items():
                        if isinstance(v, torch.Tensor):
                            v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets

class SDEvalDataset(torch.utils.data.Dataset):
    def __init__(self, path, transforms=None):
        """
        Args:
            path (str): path to the folder of images
        """
        #model version subfolder
        if len(os.listdir(path)) == 1 and os.path.isdir(os.path.join(path, os.listdir(path)[0])):
            path = os.path.join(path, os.listdir(path)[0])
        
        self.path = path
        self.img_names = [name for name in os.listdir(path) 
                     if name.endswith('.jpg') or name.endswith('.png')]
        self.img_paths = [os.path.join(path, name) for name in self.img_names]
        self.num_samples = len(self.img_names)
        assert os.path.isdir(self.path) and self.num_samples > 0, "No images found in {}".format(path)
        
        #get objects and attributes from prompts
        prompts = [name.split('-')[-1].split(".")[0] for name in self.img_names]
        #e.g. 00000-0-a black airplane.jpg
        extracted = [self._get_objects(prompt) for prompt in prompts]

        self.objects = [part[0] for part in extracted]
        self.attr = [part[1] for part in extracted]
        self.transforms = transforms
        self.dummy_target = torch.tensor(1)

    def __getitem__(self, idx):
        samples = img_loader(self.img_paths[idx])
        samples.save("../fresh_sample!.jpg")
        targets = {"objects": self.objects[idx], "attr": self.attr[idx], "img_path": self.img_paths[idx]}
        if self.transforms:
            samples, targets = self.transforms(samples, targets)
        return samples, targets
    
    
    def __len__(self):
        return self.num_samples
    

    def _get_objects(self, prompt):
        #e.g. a black cat and a brown bird and a purple car and a black truck
        words = prompt.split(' ')
        
        objects = []
        attr = []  #attributes (adjectives)
        index = -2
        while (index < len(words) - 4):
            index += 4
            objects.append(words[index])
            attr.append(words[index - 1])
        
        return objects, attr
            