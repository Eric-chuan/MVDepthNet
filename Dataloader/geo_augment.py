import torch
import torch.nn as nn
import torch.nn.functional as F

class Flip_h():
    def __init__(self):
        coordinate = np.zeros((256, 320, 2))
        for i in range(256):
            for j in range(320):
                coordinate[i, j, :] = [320-j, i]
        coordinate = coordinate.astype(np.float32)
        coordinate = torch.from_numpy(coordinate).cuda().unsqueeze(0)
        self.coordinate = coordinate
        self.coordinate[:, :, 0] = self.coordinate[:, :, 0] / (320 -1) * 2.0 - 1.0
        self.coordinate[:, :, 1] = self.coordinate[:, :, 1] / (256 -1) * 2.0 - 1.0
    def filp(self, x):
        return F.grid_sample(x, self.coordinate)

class Flip_v():
    def __init__(self):
        coordinate = np.zeros((256, 320, 2))
        for i in range(256):
            for j in range(320):
                coordinate[i, j, :] = [j, 256-i]
        coordinate = coordinate.astype(np.float32)
        coordinate = torch.from_numpy(coordinate).cuda().unsqueeze(0)
        self.coordinate = coordinate
        self.coordinate[:, :, 0] = self.coordinate[:, :, 0] / (320 -1) * 2.0 - 1.0
        self.coordinate[:, :, 1] = self.coordinate[:, :, 1] / (256 -1) * 2.0 - 1.0
    def filp(self, x):
        return F.grid_sample(x, self.coordinate)
