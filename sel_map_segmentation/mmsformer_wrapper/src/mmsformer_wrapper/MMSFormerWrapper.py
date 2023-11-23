"""
Wrapper for MMSFormer segmentation network: https://github.com/CSIPlab/MMSFormer
"""
import numpy as np
import os
import sys
import torch
from torchvision import transforms
from PIL import Image
from PIL import Image
from torch.nn import functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mmsformer'))

from semseg.models import *
from semseg.datasets import *

def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

## Class Code
class MMSFormerWrapper():
    def __init__(self, model:str = '', args:dict = {}):
        """
        Initialize the wrapper based on the model and args.

        Args:
            model (str): A string identifier to select the specific network or
                            architecture to use. Defaults to an empty string.
            args (dict): Extra arguments to pass onto the network in the form of
                            a dictionary. Defaults to an empty dictionary.
        """
        # Set the device.
        # !! Please make sure to set self.device to something !!
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # setup required torch transforms for PIL images into self.transforms
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '../ckpt/MMSFormer_MiT_B2_MCubeS_RGB.pth')

        # setup the network model
        self.model = MMSFormer('MMSFormer-B2', 20, ['image'])
        self.model.load_state_dict(torch.load(str(self.model_path), map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def runSegmentation(self, pil_image:Image, return_numpy:bool=True, one_hot:bool=True):
        """
        Pass the a PIL image through the segmentation network specified and return the pixelwise
        terrain class categorical probabilities. If one_hot is specified along with return_numpy,
        it will return the one-hot encoding in numpy.

        Args:
            pil_image (Image): A PIL object of the image to be segmented.
            return_numpy (bool, optional): Whether or not to return a numpy ndarray. Defaults to True.
            one_hot (bool, optional): Whether or not to perform one_hot encoding on the GPU. If
                                      return_numpy is False, then this has no effect. Defaults to False.

        Returns:
            3-dimensional ndarr or tensor: If return_numpy is specified as true, a 3-d numpy ndarr
            is returned with shape (num_classes, height, width) if one_hot is not specified, and
            (1, height, width) if one_hot is specified, where the first dimension is used for the
            one-hot classification. If return_numpy is false, then a tensor of shape (num_classes,
            height, width) is returned regardless of what one_hot is specified as.
        """
        # Store the image dimensions to truncate if needed
        width, height = pil_image.size
        
        ## Pad the image if needed
        ## The below will pad the width and height to multiples of 32.
        # padImageToBlocks(pil_image, 32)

        # Perform the segmentation
        img = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():

            output = self.model([img])

            output = output.softmax(dim=1)
        
            # Process the output for the single image
            scores = output.squeeze(0)
            
            # calculate the average probability across all the pixels for the predicted class
            values, indices = torch.softmax(scores, dim=0).max(axis=0)
            mean = torch.mean(values).cpu().item()
            
            if one_hot and return_numpy:
                # We save the one-hot process for the camera sensor model
                # instead if we're returning the torch tensor instead.
                # This prevents unnecessary early blocking.
                scores = indices
        
        # Crop and return the scores, returning it as a torch tensor
        # if return_numpy is false, or as a numpy ndarray otherwise
        if return_numpy:
            return scores.data.cpu().numpy()[:,:height,:width], mean
        else:
            return scores[:,:height,:width], mean


if __name__=='__main__':
    # test script
    img = Image.open('/home/anjashep-frog-lab/Desktop/00001.jpg')
    PALETTE = np.array([[ 255, 0, 0],
                        [ 0, 0, 255],
                        [ 100, 100, 100],
                        [ 100, 100, 100],
                        [ 100, 100, 100],
                        [ 100, 100, 100],
                        [ 100, 100, 100],
                        [ 100, 100, 100],
                        [ 100, 100, 100],
                        [ 255, 0, 255],
                        [ 100, 100, 100],
                        [ 100, 100, 100],
                        [ 100, 100, 100],
                        [ 100, 100, 100],
                        [ 0, 255, 0],
                        [ 100, 100, 100],
                        [ 100, 100, 100],
                        [ 100, 100, 100],
                        [ 100, 100, 100],
                        [ 100, 100, 100]])

    test = MMSFormerWrapper()
    ret = test.runSegmentation(img)

    ret = ret.squeeze(0)

    pic = ret.copy()
    pic = np.expand_dims(pic, axis=2)
    pic = np.repeat(pic, 3, axis=2)

    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            pic[i, j, :] = PALETTE[ret[i, j]]

    # plt.imshow(pic)
    # plt.show()