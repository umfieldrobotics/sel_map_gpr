import cv2
import numpy as np
import open3d as o3d
import time
import torch

from torchvision import transforms as T

from .cameraSensor import Pose
from .alexnet import AlexNet

class GPRSensor():
    def __init__(self, location=np.array([0,0,0]), rotation=np.zeros((3,3))):
        self.z_offset = 0.083 # m offset from GPR base to ground (more specifically the bottom of the wheels)

        self.cumulative_traces = np.zeros((200, 22))

        self.model = AlexNet(num_classes=4)
        self.model_path = '/home/anjashep-frog-lab/Research/gpr_mapping/gpr-clustering/out/alexnet_out/weights_500_16_1700161820.1177273.pt'
        self.one_hot_labels = ['asphalt', 'sidewalk', 'grass', 'sand']
        self.CUDA = 'cuda:0'

        # Load the trained model
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            self.model.load_state_dict(torch.load(self.model_path), map_location=torch.device('cpu'))

        self.device = torch.device(self.CUDA if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.model.eval()

    def runClassification(self, gpr_image):
        # Take gpr radargram image from numpy array to tensor
        gpr_image = cv2.merge([gpr_image, gpr_image, gpr_image]).transpose((2, 0, 1))
        gpr_image_torch = torch.from_numpy(gpr_image)
        gpr_image_torch = torch.unsqueeze(gpr_image_torch, 0)
        gpr_image_torch = T.Resize((227, 227))(gpr_image_torch)

        inputs = gpr_image_torch.to(self.device).float()

        # pass through model
        print(time.time())
        outputs = self.model(inputs)
        print(time.time())

        # softmax for probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        pred = torch.argmax(probabilities, dim=1) # convert one hot to array of classes
        # 0 = asphalt, 1 = sidewalk, 2 = grass, 3 = sand

        # ADE20K class to color mapping TODO add other network cmaps
        map_ade20k = [6, 11, 9, 46]

        pred_mapped = map_ade20k[pred.item()]

        print(pred_mapped)
    
        return pred_mapped
    
    def getProjectedPointCloudWithLabels(self, gpr_trace=None):
        gpr_trace = np.array(list(gpr_trace.trace))
        gpr_trace += 50
        gpr_trace *= 2.55

        self.cumulative_traces = np.append(self.cumulative_traces, np.expand_dims(gpr_trace, 1), axis=1)

        # once we have enough traces to form an image, pass image through network
        if self.cumulative_traces.shape[1] >= 32 :
            cv2.imwrite('/home/anjashep-frog-lab/Desktop/trace.png', np.array(self.cumulative_traces[70:102,-32:]))

            pred = self.runClassification(np.array(self.cumulative_traces[70:102,-32:]))
            # generate square point cloud
        else:
            return None

        # the image is 32 pixels long (32 seconds), so at a speed of, for example, 0.1 m/s:
        speed = 0.146 # m/s
        dist = 32 * speed
        x, y, z = np.mgrid[-dist:0:0.01, -1:1:0.01, 0:1:1] 
        var = np.empty(x.flatten().shape[0])
        var.fill(7)
        score = np.empty(x.flatten().shape[0])
        # label
        score.fill(pred)
        pc = np.vstack((x.flatten(), y.flatten(), z.flatten(), var, score)).T

        return pc