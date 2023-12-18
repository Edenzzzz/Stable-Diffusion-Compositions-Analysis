# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

# ./configs/deta_swin_ft.sh --eval -f "/nobackup/wenxuan/Stable-Diffusion-Compositions-Analysis/outputs"  --resume weights/adet_swin_ft.pth
import torch
import argparse
import datetime
import json
import random
import time
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import DataLoader
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_sd_dataset
from models import build_model
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from datasets.data_prefetcher import data_prefetcher

#80 categories
CLASSES = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
def show_boxes(img, boxes, labels):
	box = draw_bounding_boxes(img, boxes,
						  labels=labels,
						  colors="red",
						  width=4, font_size=30)
	im = to_pil_image(box.detach())
	im.save("../demo_detection.jpg")

def get_args_parser():
	parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
	parser.add_argument('--lr', default=2e-4, type=float)
	parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
	parser.add_argument('--lr_backbone', default=2e-5, type=float)
	parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
	parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
	parser.add_argument('--batch_size', default=4, type=int)
	parser.add_argument('--weight_decay', default=1e-4, type=float)
	parser.add_argument('--epochs', default=50, type=int)
	parser.add_argument('--lr_drop', default=40, type=int)
	parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
	parser.add_argument('--clip_max_norm', default=0.1, type=float,
						help='gradient clipping max norm')


	# Variants of Deformable DETR
	parser.add_argument('--with_box_refine', default=False, action='store_true')
	parser.add_argument('--two_stage', default=False, action='store_true')

	# Model parameters
	parser.add_argument('--frozen_weights', type=str, default=None,
						help="Path to the pretrained model. If set, only the mask head will be trained")

	# * Backbone
	parser.add_argument('--backbone', default='resnet50', type=str,
						help="Name of the convolutional backbone to use")
	parser.add_argument('--dilation', action='store_true',
						help="If true, we replace stride with dilation in the last convolutional block (DC5)")
	parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
						help="Type of positional embedding to use on top of the image features")
	parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
						help="position / size * scale")
	parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

	# * Transformer
	parser.add_argument('--enc_layers', default=6, type=int,
						help="Number of encoding layers in the transformer")
	parser.add_argument('--dec_layers', default=6, type=int,
						help="Number of decoding layers in the transformer")
	parser.add_argument('--dim_feedforward', default=1024, type=int,
						help="Intermediate size of the feedforward layers in the transformer blocks")
	parser.add_argument('--hidden_dim', default=256, type=int,
						help="Size of the embeddings (dimension of the transformer)")
	parser.add_argument('--dropout', default=0.1, type=float,
						help="Dropout applied in the transformer")
	parser.add_argument('--nheads', default=8, type=int,
						help="Number of attention heads inside the transformer's attentions")
	parser.add_argument('--num_queries', default=300, type=int,
						help="Number of query slots")
	parser.add_argument('--dec_n_points', default=4, type=int)
	parser.add_argument('--enc_n_points', default=4, type=int)

	# * Segmentation
	parser.add_argument('--masks', action='store_true',
						help="Train segmentation head if the flag is provided")

	# Loss
	parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
						help="Disables auxiliary decoding losses (loss at each layer)")

	# * Matcher
	parser.add_argument('--assign_first_stage', action='store_true')
	parser.add_argument('--assign_second_stage', action='store_true')
	parser.add_argument('--set_cost_class', default=2, type=float,
						help="Class coefficient in the matching cost")
	parser.add_argument('--set_cost_bbox', default=5, type=float,
						help="L1 box coefficient in the matching cost")
	parser.add_argument('--set_cost_giou', default=2, type=float,
						help="giou box coefficient in the matching cost")

	# * Loss coefficients
	parser.add_argument('--mask_loss_coef', default=1, type=float)
	parser.add_argument('--dice_loss_coef', default=1, type=float)
	parser.add_argument('--cls_loss_coef', default=2, type=float)
	parser.add_argument('--bbox_loss_coef', default=5, type=float)
	parser.add_argument('--giou_loss_coef', default=2, type=float)
	parser.add_argument('--focal_alpha', default=0.25, type=float)

	# dataset parameters
	parser.add_argument('--dataset_file', default='coco')
	parser.add_argument('--coco_path', default='./data/coco', type=str)
	parser.add_argument('--coco_panoptic_path', type=str)
	parser.add_argument('--remove_difficult', action='store_true')
	parser.add_argument('--bigger', action='store_true')

	parser.add_argument('--output_dir', default='',
						help='path where to save, empty for no saving')
	parser.add_argument('--device', default='cuda',
						help='device to use for training / testing')
	parser.add_argument('--seed', default=42, type=int)
	parser.add_argument('--resume', default='', help='resume from checkpoint')
	parser.add_argument('--finetune', default='', help='finetune from checkpoint')
	parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
						help='start epoch')
	parser.add_argument('--eval', action='store_true')
	parser.add_argument('--num_workers', default=2, type=int)
	parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
	
 # @Wenxuan
	parser.add_argument("-f","--file", help="SD dataset with subfolders by num_object ")
	parser.add_argument("--img_size", default=512, type=int)
	return parser


def sd_eval(args):
	utils.init_distributed_mode(args)
	print("git:\n  {}\n".format(utils.get_sha()))

	if args.frozen_weights is not None:
		assert args.masks, "Frozen training is meant for segmentation only"
	print(args)

	device = torch.device(args.device)

	# fix the seed for reproducibility
	seed = args.seed + utils.get_rank()
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	model, criterion, postprocessors = build_model(args)
	if args.resume:
		if args.resume.startswith('https'):
			checkpoint = torch.hub.load_state_dict_from_url(
				args.resume, map_location='cpu', check_hash=True)
		else:
			checkpoint = torch.load(args.resume, map_location='cpu')
		missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
		unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
		if len(missing_keys) > 0:
			print('Missing Keys: {}'.format(missing_keys))
		if len(unexpected_keys) > 0:
			print('Unexpected Keys: {}'.format(unexpected_keys))
	model.to(device)
	model.eval()

		
	if args.distributed:
		model_without_ddp = model
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
	
	#eval-txt2img-three-objects-and
	subfolders = [name for name in os.listdir(args.file) if os.path.isdir(os.path.join(args.file, name)) and "eval" in name]
	order = {"one": 0, "two": 1, "three": 2, "four": 3}
	subfolders.sort(key=lambda x: order[x.split('-')[-3]])


	message = ""
	for subfolder in tqdm(subfolders, desc='Subfolder'):
		dataset_test = build_sd_dataset(os.path.join(args.file, subfolder))
		all_relevant_obj = []
		all_relevant_boxes = []
		if args.distributed:
			if args.cache_mode:
				sampler_val = samplers.NodeDistributedSampler(dataset_test, shuffle=False)
			else:
				sampler_val = samplers.DistributedSampler(dataset_test, shuffle=False)
		else:
			sampler_val = torch.utils.data.SequentialSampler(dataset_test)

		data_loader = DataLoader(dataset_test, args.batch_size, sampler=sampler_val,
									drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
									pin_memory=True)
		data_fetcher = data_prefetcher(data_loader, device=device)

		
		max_keep = 7 #select top 7 boxes only
		samples, targets = data_fetcher.next()
		for batch_idx in tqdm(range(len(data_loader)), desc="Running SD evaluation using DETA"):
			all_boxes_dt = []#for vis and debug
			all_classes_dt = []

			outputs = model(samples)
			target = targets[0]
			if len(targets) >= 1:
				Warning("Batch size larget than 1 in targets")
			
			#For visualization, use resized. Must one image at a time because sizes are different
			target_sizes = target['size'].unsqueeze(0).to(device)
			results = postprocessors['bbox'](outputs, target_sizes)

			
			results = {key: results[0][key][:max_keep] for key in results[0]} #List[Dict[Str, Tensor]]
			scores, labels, boxes = results.values()
			assert len(scores) == len(labels) == len(boxes) == max_keep
			obj_dt = set()
			box_dt = dict()

			for score, label, box in zip(scores, labels, boxes):
				label = label.item()
				#detect objects only for now; no attributes like color
				if CLASSES[label] in target["objects"] and CLASSES[label] not in obj_dt:
					obj_dt.add(CLASSES[label])
					box_dt[CLASSES[label]] = box.cpu()
				elif not CLASSES[label] in all_classes_dt :
					all_boxes_dt.append(box)
					all_classes_dt.append(CLASSES[label])
			
			all_relevant_obj.append(obj_dt)
			all_relevant_boxes.append(box_dt)
			
			#prefetch next batch
			try:
				data = data_fetcher.next()
				if None not in data:
					samples, targets = data
			except Exception as e:
				print(e)
				breakpoint()
		
		#show a detected box example if any
		if len(box_dt) != 0:
			box_dt = torch.stack(list(*box_dt.values())) #to (N, 4) tensor
			if len(box_dt.shape) == 1:
				box_dt = box_dt.unsqueeze(0)
			if len(box_dt) != 0:
				show_boxes(samples[-1].to(torch.uint8), box_dt, list(obj_dt) )

		#compute average number of objects detected	
		num_obj_freq = len(subfolders) * [0]
		for detected in all_relevant_obj:
			if len(detected) == 0:
				continue
			num_obj_freq[len(detected) - 1] += 1
		#average number of objects detected
		num_obj_freq = [round(o / len(all_relevant_obj), 3) for o in num_obj_freq]
		#format: outputs/eval-txt2img-four-objects-and
		category = " ".join(subfolder.split('-')[-3:-1])
		message += f"For {category}, the average frequencies of number of objects detected is {num_obj_freq} \n"
		

	print(message)	

	
if __name__ == '__main__':
	parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
	args = parser.parse_args()
	if args.output_dir:
		Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	sd_eval(args)
