from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torchvision.transforms as transforms
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
import onnxruntime as ort
from .yolox import *
import cv2

curdir_path = osp.dirname(__file__)

class PersonDetection:
    def __init__(self):
        self.model = YOLOXDetection()

    def __call__(self, img):
        '''
        img: BGR order
        '''
        output = self.model(img)
        mask = output[...,-1]==0
        output = output[mask]
        bboxes = output[...,:4]
        #labels = output[...,-1]
        probs = output[...,4]*output[...,5]

        wh = bboxes[...,2:]-bboxes[...,:2]
        wh_mask = wh>1
        size_mask = np.logical_and(wh[...,0],wh[...,1])

        return bboxes,probs

class KPDetection:
    def __init__(self) -> None:
        onnx_path = osp.join(curdir_path,"keypoints.onnx")
        self.model = ort.InferenceSession(onnx_path)
        self.input_name = self.model.get_inputs()[0].name
        self.person_det = PersonDetection()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def cut_and_resize(img,bboxes,size=(288,384)):
        res = []
        for bbox in bboxes:
            cur_img = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
            if cur_img.shape[0]>1 and cur_img.shape[1]>1:
                cur_img = cv2.resize(cur_img,size,interpolation=cv2.INTER_LINEAR)
            else:
                cur_img = np.zeros([size[1],size[0],3],dtype=np.float32)
            res.append(cur_img)
        return res

    @staticmethod
    def get_offset_and_scalar(bboxes,size=(288,384)):
        offset = bboxes[...,:2]
        offset = np.expand_dims(offset,axis=1)
        bboxes_size = bboxes[...,2:]-bboxes[...,:2]
        cur_size = np.array(size,np.float32)
        cur_size = np.resize(cur_size,[1,2])
        scalar = bboxes_size/cur_size
        scalar = np.expand_dims(scalar,axis=1)*4
        return offset,scalar

    @staticmethod
    def get_max_preds(batch_heatmaps):
        '''
        get predictions from score maps
        heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        '''
        assert isinstance(batch_heatmaps, np.ndarray), \
            'batch_heatmaps should be numpy.ndarray'
        assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'
        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width #x
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width) #y

        pred_mask = np.tile(np.greater(maxvals, 0.05), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds, maxvals

    @staticmethod
    def get_final_preds(batch_heatmaps):
        coords, maxvals = KPDetection.get_max_preds(batch_heatmaps)

        heatmap_height = batch_heatmaps.shape[2]
        heatmap_width = batch_heatmaps.shape[3]

        # post-processing
        if True:
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    hm = batch_heatmaps[n][p]
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                        diff = np.array(
                            [
                                hm[py][px + 1] - hm[py][px - 1],
                                hm[py + 1][px] - hm[py - 1][px]
                            ]
                        )
                        coords[n][p] += np.sign(diff) * .25

        preds = coords.copy()

        #return preds, maxvals
        return np.concatenate([preds,maxvals],axis=-1)

    def __call__(self, img):
        '''

        Args:
            img: RGB order

        Returns:
            ans: [N,17,3] (x,y,score,...)
        '''
        img = img[...,::-1]
        bboxes,probs = self.person_det(img)
        if len(probs) == 0:
            return np.zeros([0,17,3],dtype=np.float32)
        bboxes = bboxes.astype(np.int32)
        #print(bboxes)
        cv2.imwrite("/home/wj/ai/mldata/0day/x1/a.jpg",img)
        img = img[...,::-1]
        imgs = self.cut_and_resize(img,bboxes)
        cv2.imwrite("/home/wj/ai/mldata/0day/x1/b.jpg",imgs[0])
        imgs = [self.transform(x) for x in imgs]
        imgs = [x.cpu().numpy() for x in imgs]
        imgs = np.array(imgs)
        output = self.model.run(None, {self.input_name: imgs})[0]
        output = self.get_final_preds(output)
        bboxes = bboxes.astype(np.float32)
        offset,scalar = self.get_offset_and_scalar(bboxes)
        output[...,:2] = output[...,:2]*scalar+offset
        return output
