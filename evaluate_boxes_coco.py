import torchvision
from torchvision.models import detection
import numpy as np
import argparse
import pickle
import torch
import cv2
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt 
import seaborn as sns
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

def show_boxes(img, prediction, labels):
	box = draw_bounding_boxes(img, boxes=prediction["boxes"],
						  labels=labels,
						  colors="red",
						  width=4, font_size=30)
	im = to_pil_image(box.detach())
	im.save("demo_detection.png")
	
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", type=str, required=True,
	help="path to the images")
args = vars(ap.parse_args())

# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load the list of categories in the COCO dataset and then generate a
# set of bounding box colors for each class
CLASSES = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'trafficlight', 'firehydrant', 'streetsign', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eyeglasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite', 'baseballbat', 'baseballglove', 'skateboard', 'surfboard', 'tennisracket', 'bottle', 'plate', 'wineglass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'mirror', 'diningtable', 'window', 'desk', 'toilet', 'door', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush', 'hairbrush']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
WEIGHTS = detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
preprocess = preprocess = WEIGHTS.transforms()

# load the model and set it to evaluation mode
model = detection.fasterrcnn_resnet50_fpn_v2(weights=WEIGHTS, 
							  progress=True, 
							  num_classes=len(CLASSES), 
							  weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT).to(DEVICE)
model.eval()
detected = {}


subfolders = [name for name in os.listdir(args['file']) if os.path.isdir(os.path.join(args['file'], name)) and "eval" in name]
order = {"one": 0, "two": 1, "three": 2, "four": 3}
subfolders.sort(key=lambda x: order[x.split('-')[-3]])



message = ""
for folder in tqdm(subfolders, total=len(subfolders), desc="Categories (by num obj in prompt)"):
	all_obj_detected = []
	all_boxes = []
	path = os.path.join(args['file'], folder)
	#subfolder for model version id
	if len(os.listdir(path)) == 1:
		path = os.path.join(path, os.listdir(path)[0])
		assert os.path.isdir(path)
	
	#iterate over all images in the folder
	for img in tqdm(os.listdir(path), total=len(os.listdir(path))):

		prompt = img.split('-')[-1].split('.')[0]

		words = prompt.split(' ')
		objects = []
		index = -2
		while (index < len(words)-4):
			index += 4
			objects.append(words[index])

	
		# load the image from disk
		image = read_image(path + '/' + img)
		batch = [preprocess(image).to(DEVICE)]
		
		prediction = model(batch)[0]
		labels = [WEIGHTS.meta["categories"][i] for i in prediction["labels"]]
		
		# Find the most confident prediction for the depicted object	
		detected = set()
		boxes = {ob: None for ob in objects}
		for i, label in enumerate(labels):
			if (label not in objects):
				continue 
			#the goal is to test if objects in the prompt exist in the image, not count the number 
			boxes[label] = prediction['boxes'][i].cpu().detach().numpy()
			detected.add(label)
		# show_boxes(image, prediction, labels)

		all_obj_detected.append(detected)
		all_boxes.append(boxes)


	#frequency from all images of number of objects detected 
	num_obj_freq = len(subfolders) * [0]
	for detected in all_obj_detected:

		if len(detected) == 0:
			continue
		num_obj_freq[len(detected) - 1] += 1
	#average number of objects detected
	num_obj_freq = [round(o / len(all_obj_detected), 3) for o in num_obj_freq]
	#format: outputs/eval-txt2img-four-objects-and
	category = " ".join(folder.split('-')[-3:-1])
	message += f"For {category}, the average frequencies of number of objects detected is {num_obj_freq} \n"
	

print(message)

	# if (not found):
	# 	print (name)

# with open('one-object-box-v%d.pickle' %(args['version']), 'wb') as f:
# 	pickle.dump(predictions, f, protocol=pickle.HIGHEST_PROTOCOL)


# box_size = []
# box_pos = []

# for img_id, box in predictions.items():
# 	box_size.append((box[2]-box[0])*(box[3]-box[1]))
# 	box_pos.append((int(box[0] + (box[2] - box[0])/2), int(box[1] + (box[3] - box[1])/2)))

# sns.kdeplot(box_size)
# plt.savefig('one-object-box-v%d-size' %(args['version']))
# plt.close()

# factor = 10
# center_map = np.zeros((int(512/factor), int(512/factor)))
# for p in box_pos:
# 	center_map[int(p[0]/factor), int(p[1]/factor)] += 1

# print (center_map.sum())

# plt.imshow(center_map/center_map.sum(), cmap='Reds')
# plt.savefig('one-object-box-v%d-center' %(args['version']))
# plt.close()











