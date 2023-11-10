import numpy as np
import open3d as o3d

from .cameraSensor import Pose

class GPRSensor():
    def __init__(self, location=np.array([0,0,0]), rotation=np.zeros((3,3))):
        self.z_offset = 0.083 # m offset from GPR base to ground (more specifically the bottom of the wheels)
        # TODO where are these transforms getting set?

    def updateSensorMeasurements(self, trace):
        # image or just array from single trace?
        return
    
    def runClassification(self):
        return
    
    def projectMeasurementsIntoSensorFrame(self):
        return
    
    def computePointCovariance(self, point):
        return
    
    def getProjectedPointCloudWithLabels(self):
        # generate square point cloud

        x, y, z = np.mgrid[-3:3:0.05, -3:3:0.05, 1:2:1]
        var = np.empty(x.flatten().shape[0])
        var.fill(7)
        score = np.empty(x.flatten().shape[0])
        # label
        score.fill(11)
        pc = np.vstack((x.flatten(), y.flatten(), z.flatten(), var, score)).T

        return pc