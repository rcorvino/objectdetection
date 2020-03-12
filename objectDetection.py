from PIL import Image
import argparse
import math
import skimage

from os import listdir
from os import path
from xml.etree import ElementTree

import numpy as np
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from matplotlib import pyplot
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from numpy import expand_dims
from numpy import mean

from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image


from matplotlib import pyplot
from matplotlib.patches import Rectangle



def get_ids(class_name):
	base_id, class_id = 0, 1
	if class_name == 'smart-baby-bottle':
		class_id = 2
	elif class_name == 'toothbrush':
		class_id = 3
	elif class_name == 'wake-up-light':
		class_id = 4
	base_id = (class_id-1)*16
	return base_id, class_id


# class that defines and loads the shaver dataset
class PhilipsDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, class_name, is_train=True):
		#make class_id unique
		base_id, class_id = get_ids(class_name)
		# define one class
		self.add_class("dataset", class_id, class_name)
		# define data locations
		images_dir = 'philips_data/'+class_name + '/images/'
		annotations_dir = 'philips_data/'+class_name + '/annots/image'

		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = str(int(filename[5:-4]) + base_id)
			# skip all images after 14 if we are building the train set
			if is_train and int(image_id) >= (14 + base_id):
				continue
			# skip all images before 14 if we are building the test/val set
			if not is_train and int(image_id) < (14 + base_id):
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + filename[5:-4] + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for obj in root.findall('.//object'):
			class_name = obj.find('name').text
			box = obj.find('bndbox')
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height, class_name

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h, class_name = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index(class_name))
		return masks, asarray(class_ids, dtype='int32')


class PhilipsConfig(Config):
	# Give the configuration a recognizable name
	NAME = "philips_cfg"
	# Number of classes (background + shaver + bottle + lamp + toothbrush)
	NUM_CLASSES = 1 + 4
	# Number of training steps per epoch
	STEPS_PER_EPOCH = 100
	VALIDATION_STEPS = 5




# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "philips_cfg"
	# number of classes (background + shaver + bottle + lamp + toothbrush)
	NUM_CLASSES = 1 + 4
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

#extract a single object from a given image
def get_detected_objects(dataset, model, cfg, show=False):
	objects = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		#get the zone of the image with the detected objects
		y1, x1, y2, x2 = yhat[0]['rois'][0]
		if show:
			pyplot.imshow(image[y1:y2, x1:x2, :])
			pyplot.show()
		object =image[y1:y2, x1:x2, :]
		# store
		objects.append(object)

	return objects

class_names = ['background', 'shaver', 'smart-baby-bottle', 'toothbrush', 'wake-up-light ']
def display_all(dataset, model, cfg):
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		#display image, detected objects and classes
		display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

def predict(dataset, model, cfg):
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		#display image, detected objects and classes
		print(dataset.image_info[image_id]['path']+": "+class_names[r['class_ids'][0]])

def predict_validation(model, cfg):

	if path.exists('validation'):
		for img in listdir("validation"):
			dir = "validation/"+img
			# load image
			image = skimage.io.imread(dir)
			# convert pixel values (e.g. center)
			scaled_image = mold_image(image, cfg)
			# convert image into one sample
			sample = expand_dims(scaled_image, 0)
			# make prediction
			yhat = model.detect(sample, verbose=0)
			# extract results for first sample
			r = yhat[0]
			#display image, detected objects and classes
			if len(r['class_ids']) >0 :
				print(dir+": "+class_names [r['class_ids'][0]])
			else :
				print(dir+": NONE")
	else:
		print('missing directory validation')


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-t', action='store_true', help="train the model")
	parser.add_argument('-v', action='store_true', help="evaluate the model")
	parser.add_argument('-d', action='store_true', help="debug functions of the model")
	args = parser.parse_args()


	if args.t or args.v or args.d:
		# train set
		train_set = PhilipsDataset()
		train_set.load_dataset('shaver', is_train=True)
		train_set.load_dataset('smart-baby-bottle', is_train=True)
		train_set.load_dataset('toothbrush', is_train=True)
		train_set.load_dataset('wake-up-light', is_train=True)
		train_set.prepare()
		print('Train: %d' % len(train_set.image_ids))

		# test/val set
		test_set = PhilipsDataset()
		test_set.load_dataset('shaver', is_train=False)
		test_set.load_dataset('smart-baby-bottle', is_train=False)
		test_set.load_dataset('toothbrush', is_train=False)
		test_set.load_dataset('wake-up-light', is_train=False)
		test_set.prepare()
		print('Test: %d' % len(test_set.image_ids))

	if args.v:
		# create config
		cfg = PredictionConfig()
		# define the model
		model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
		# load model weights
		model.load_weights('mask_rcnn_philips_cfg_0004.h5', by_name=True)

		predict(test_set, model, cfg)

	elif args.d:
		train_set.extract_boxes("philips_data/shaver/annots/image4.xml")

	elif args.t:
		#train
		#prepare config
		config = PhilipsConfig()
		config.display()
		# define the model
		model = MaskRCNN(mode='training', model_dir='./', config=config)
		# load weights (mscoco) and exclude the output layers
		model.load_weights('mask_rcnn_philips_cfg_0004.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
		# train weights (output layers or 'heads')
		model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
	else:
		# create config
		cfg = PredictionConfig()
		# define the model
		model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
		# load model weights
		model.load_weights('mask_rcnn_philips_cfg_0004.h5', by_name=True)
		predict_validation(model, cfg)


if __name__ == "__main__":
	main()
