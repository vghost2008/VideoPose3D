from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as osp
import cv2
import argparse
import os
import pprint
import glob
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
import numpy as np

pdir_path = osp.dirname(osp.dirname(__file__))
torch.ops.load_library(osp.join(pdir_path,"boeweights/libhrnet_op.so"))



class KPDetection:
    def __init__(self) -> None:
        pt_path = osp.join(pdir_path,"boeweights/traced_cop.pt")
        self.model = torch.jit.load(pt_path)
        self.device = torch.device("cuda")
    
    def __call__(self, img):
        fimage = img.astype(np.float32)
        fimage = torch.Tensor(fimage).to(self.device)
        ans = self.model(fimage)
        ans = ans.cpu().detach().numpy()
        return ans
