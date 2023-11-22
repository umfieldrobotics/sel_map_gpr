import cv2
import math
import numpy as np
import open3d as o3d
import time
import torch

from torchvision import transforms as T

from .cameraSensor import Pose
from .alexnet import AlexNet

from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

class GPRSensor():
    def __init__(self, location=np.array([0,0,0]), rotation=np.zeros((3,3))):
        self.z_offset = 0.083 # m offset from GPR base to ground (more specifically the bottom of the wheels)

        self.cumulative_traces = np.zeros((200, 22))

        self.model = AlexNet(num_classes=4)
        self.model_path = '/home/anjashep-frog-lab/Research/gpr_mapping/gpr-clustering/out/alexnet_out/weights_200_16_1700582696.6877162.pt'
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

        self.position_list = None
        self.z_rot_list = None

    # Return the predicted class of the image and the probability output from the network
    def runClassification(self, gpr_image):
        # Take gpr radargram image from numpy array to tensor
        gpr_image = cv2.merge([gpr_image, gpr_image, gpr_image]).transpose((2, 0, 1))
        gpr_image_torch = torch.from_numpy(gpr_image)
        gpr_image_torch = torch.unsqueeze(gpr_image_torch, 0)
        gpr_image_torch = T.Resize((227, 227))(gpr_image_torch)

        inputs = gpr_image_torch.to(self.device).float()

        # pass through model
        outputs = self.model(inputs)

        # softmax for probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        pred = torch.argmax(probabilities, dim=1) # convert one hot to array of classes
        # 0 = asphalt, 1 = sidewalk, 2 = grass, 3 = sand

        # ADE20K class to color mapping TODO add other network cmaps
        map_ade20k = [6, 11, 9, 46]

        pred_mapped = map_ade20k[pred.item()]
        pred_probability = probabilities[0][pred.item()].item()
    
        return pred_mapped, pred_probability
    
    def calculateGPRPointCloudShape(self):
        # note: one pose added to pose list at each GPR callback, so each trace corresponds to one pose

        # get last 32 poses (aka last 32 traces, aka one image) but if it's less than that like at the start of a scene then just take those
        # also subtract 
        if self.position_list.shape[0] >= 32:
            last_32 = self.position_list[-32:,:]# - self.position_list[-1:,:]
        else:
            last_32 = self.position_list# - self.position_list[-1:,:]

        # last_32 in odom frame. transform to GPR frame at current time step

        # t_odom_to_foot = -self.position_list[-1,:]
        # rot_odom_to_foot = Rotation.from_quat(poses[0].rotation).as_matrix().T
        # for i in range(last_32.shape[0]):
        #     last_32[i,:] =  rot_odom_to_foot @ last_32[i,:] + t_odom_to_foot

        # Fit cubic to position list
        # Parameter (e.g., arc length or simple range)
        t = np.arange(len(last_32))

        # Fit separate splines for x, y, and z
        spl_x = CubicSpline(t, last_32[:, 0])
        spl_y = CubicSpline(t, last_32[:, 1])
        spl_z = CubicSpline(t, last_32[:, 2])

        # Interpolated points
        t_interpolated = np.linspace(0, t[-1], 100)
        x_interpolated = spl_x(t_interpolated)
        y_interpolated = spl_y(t_interpolated)
        z_interpolated = spl_z(t_interpolated)

        x_interp_right = x_interpolated.copy()
        y_interp_right = y_interpolated.copy()
        x_interp_left = x_interpolated.copy()
        y_interp_left = y_interpolated.copy()


        for i in range(len(t_interpolated)):
            idx = round(i / len(t_interpolated) * (len(self.z_rot_list)-1))
            x_interp_right[i] += math.sin(-self.z_rot_list[idx])
            y_interp_right[i] += math.cos(-self.z_rot_list[idx])
            x_interp_left[i] -= math.sin(-self.z_rot_list[idx])
            y_interp_left[i] -= math.cos(-self.z_rot_list[idx])
        across = None

        for i in range(len(t_interpolated)):
            if i == 0:
                across = np.linspace([x_interp_left[i], y_interp_left[i], z_interpolated[i]], [x_interp_right[i], y_interp_right[i], z_interpolated[i]], 100)
            else:
                across = np.vstack((across, np.linspace([x_interp_left[i], y_interp_left[i], z_interpolated[i]], [x_interp_right[i], y_interp_right[i], z_interpolated[i]], 100)))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(last_32[:, 0], last_32[:, 1], last_32[:, 2], color='red', label='Original Points')
        ax.plot(x_interpolated, y_interpolated, z_interpolated, color='blue', label='Fitted Spline')
        ax.scatter(x_interp_right, y_interp_right, z_interpolated, color='black')
        ax.scatter(x_interp_left, y_interp_left, z_interpolated, color='black')
        # ax.scatter(across[:, 0], across[:, 1], across[:, 2], color='green')
        ax.legend()
        ax.set_zlim3d(-1, 1)

        fig.savefig('/home/anjashep-frog-lab/Desktop/test.png', format='png')

        return across

    
    def getProjectedPointCloudWithLabels(self, gpr_trace=None, pose_hist=None):
        # accumulate list of poses in odom frame and rotatios from base_footprint to odom frame
  
        if self.position_list is None:
            self.position_list = np.array(pose_hist[0].location)
            self.z_rot_list = np.array(Rotation.from_quat(pose_hist[0].rotation).as_euler('xyz')[2])
        else:
            self.position_list = np.vstack((self.position_list, np.array(pose_hist[-1].location)))
            self.z_rot_list = np.append(self.z_rot_list, Rotation.from_quat(pose_hist[-1].rotation).as_euler('xyz')[2])

        gpr_trace = np.array(list(gpr_trace.trace))
        gpr_trace += 50
        gpr_trace *= 2.55

        self.cumulative_traces = np.append(self.cumulative_traces, np.expand_dims(gpr_trace, 1), axis=1)

        # once we have enough traces to form an image, pass image through network
        if self.cumulative_traces.shape[1] >= 32 :
            cv2.imwrite('/home/anjashep-frog-lab/Desktop/trace.png', np.array(self.cumulative_traces[93:125,-32:]))

            pred, prob = self.runClassification(np.array(self.cumulative_traces[93:125,-32:]))
            # generate square point cloud
        else:
            return None
        
        # form the pointcloud shape according to the path of the robot (+/- 1 meter on each side)
        pc = self.calculateGPRPointCloudShape() # IN GLOBAL FRAME
 
        
        # add variance and class prediction
        var = np.empty((pc.shape[0], 1))
        var.fill(7)
        score = np.empty((pc.shape[0], 1))
        # label
        score.fill(pred)
        pc = np.hstack((pc, var, score))

        # # the image is 32 pixels long (32 seconds), so at a speed of, for example, 0.1 m/s:
        # speed = 0.146 # m/s
        # dist = 32 * speed
        # x, y, z = np.mgrid[-dist:0:0.01, -1:1:0.01, 0:1:1] 
        # var = np.empty(x.flatten().shape[0])
        # var.fill(7)
        # score = np.empty(x.flatten().shape[0])
        # # label
        # score.fill(pred)
        # pc = np.vstack((x.flatten(), y.flatten(), z.flatten(), var, score)).T

        return pc, prob