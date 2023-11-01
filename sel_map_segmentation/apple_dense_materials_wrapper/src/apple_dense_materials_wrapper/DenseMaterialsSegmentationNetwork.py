import cv2
import json
import math
import os
import random
import torch
import numpy as np
import torchvision.transforms as TTR

from PIL import Image
from scipy.io import loadmat
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder


## Useful helper functions
def padImageToBlocks(pil_image:Image, block_size:int) -> Image:
    """
    Helper function to pad the width and height to multiples of block_size

    Args:
        pil_image (Image): PIL Image object to pad
        block_size (int): The block size to ensure the Image is padded to multiples of.

    Returns:
        Image: A PIL Image object containing the image in the upper left corner,
        with the height and widths multiples of the desired block_size.
    """
    width, height = pil_image.size

    # Identify padding residuals
    width_pad = block_size - (width % block_size)
    height_pad = block_size - (height % block_size)

    # If we don't need to pad, return pil_image
    if width_pad + height_pad == 0:
        return pil_image
    
    # Otherwise pad and return
    new_image = Image.new(pil_image.mode, (width+width_pad, height+height_pad), 0)
    new_image.paste(pil_image, (0,0))
    return new_image


class DenseMaterialsSegmentationNetwork():
	def __init__(self, model="Apple_ResNet50_DMS_full.yaml", args=None, verbose=False):
		'''
		Initializes and runs the Apple Dense Materials Segmentation network for use in the
		terrain estimation mapping algorithm. Loads network weights, prepares input
		images for network, runs the segmentation network, and outputs visualizations
		to a file.

		Parameters
		------------
		args : obj, provides necessary arguements for network initialization

		Returns
		-----------
		'''

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.is_cuda = torch.cuda.is_available()
		random.seed(112)

		path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

		self.dms46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 
		   		 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 
				 51, 52, 53, 54, 55, 56]
		t = json.load(open(os.path.join(path, 'apple_dense_materials/taxonomy_edit.json'), 'rb'))
		srgb_colormap = [
			t['srgb_colormap'][i] for i in range(len(t['srgb_colormap'])) if i in self.dms46
		]
		srgb_colormap = np.array(srgb_colormap, dtype=np.uint8)

		jit_path = os.path.join(os.path.dirname(path), 'ckpt/DMS46_v1.pt')

		self.model = torch.jit.load(jit_path)

		if self.is_cuda:
			self.model = self.model.cuda()

		value_scale = 255
		mean = [0.485, 0.456, 0.406]
		self.mean = [item * value_scale for item in mean]
		std = [0.229, 0.224, 0.225]
		self.std = [item * value_scale for item in std]


	def runSegmentation(self, pil_image:Image, return_numpy=True, one_hot=False):
		'''
		Passes an image through the network and returns the pixelwise terrain class
		categorical probabilities.

		Parameters
		------------
		image : PIL.Image, image to input to network

		Returns
		-----------
		array : (w,h,k) shape array, pixelwise terrain class probability scores
		'''

		# Resize image to 512 x 512 (rescales)
		new_dim = 512
		w, h = pil_image.size[0:2]
		scale_x = float(new_dim) / float(w)
		scale_y = float(new_dim) / float(h)
		scale = min(scale_x, scale_y)
		new_h = math.ceil(scale * h)
		new_w = math.ceil(scale * w)
		pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

		cv_image = np.array(pil_image)

		# Convert to torch image and normalize
		image = torch.from_numpy(cv_image.transpose((2, 0, 1))).float()
		image = TTR.Normalize(self.mean, self.std)(image)

		if self.is_cuda:
			image = image.cuda()
		image = image.unsqueeze(0)

		with torch.no_grad():
			prediction = self.model(image)[0].data.cpu()[0, 0].numpy()

		# Reshape back to original size in order to match with depth image
		prediction = cv2.resize(prediction, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

		# Convert output from 1 x h x w (where each index is the class number for that pixel) to 59 x h x w where each pixel is a row where the class number index is 1
		one_hot_enc = OneHotEncoder(categories=[self.dms46], sparse=False)

		new_prediction = None
		for i in range(prediction.shape[0]):
			res = one_hot_enc.fit_transform(prediction[i].reshape(-1, 1)).T
			res = res.reshape(res.shape[0], 1, res.shape[1])
			if new_prediction is None:
				new_prediction = np.array(res)
			else:
				new_prediction = np.append(new_prediction, res, axis=1)

        # Return the scores
		if one_hot and return_numpy:
			return prediction
		elif one_hot and not return_numpy:
			return torch.tensor.convert_to_tensor(prediction).cuda()
		elif not one_hot and return_numpy:
			return new_prediction
		else:
			return torch.tensor(new_prediction).cuda()


# if __name__ == '__main__':
# 	image = Image.open('/home/anjashep-frog-lab/Desktop/00001.jpg').convert('RGB')

# 	network = DenseMaterialsSegmentationNetwork()
# 	scores = network.runSegmentation(image)