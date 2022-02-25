from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torchvision.transforms as transforms
import os.path as osp
import cv2
import argparse
import copy
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
import object_detection2.bboxes as odb
from .image_encode import ImageEncoder

curdir_path = osp.dirname(__file__)

class PersonDetection:
    def __init__(self):
        self.model = YOLOXDetection()

    def __call__(self, img):
        '''
        img: BGR order
        '''
        assert len(img.shape)==3,"Error img size"
        output = self.model(img)
        mask = output[...,-1]==0
        output = output[mask]
        bboxes = output[...,:4]
        #labels = output[...,-1]
        probs = output[...,4]*output[...,5]

        wh = bboxes[...,2:]-bboxes[...,:2]
        bboxes = odb.npscale_bboxes(bboxes,1.2,max_size=[img.shape[1],img.shape[0]])
        wh_mask = wh>1
        size_mask = np.logical_and(wh_mask[...,0],wh_mask[...,1])
        bboxes = bboxes[size_mask]
        probs = probs[size_mask]

        return bboxes,probs

class KPDetection:
    threshold = 0.3
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
        self.image_encoder = ImageEncoder()

    @staticmethod
    def cut_and_resizev0(img,bboxes,size=(288,384)):
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
    def cut_and_resize(img,bboxes,size=(288,384)):
        res = []
        res_bboxes = []
        for i,bbox in enumerate(bboxes):
            cur_img = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
            if cur_img.shape[0]>1 and cur_img.shape[1]>1:
                #cur_img = cv2.resize(cur_img,size,interpolation=cv2.INTER_LINEAR)
                cur_img,bbox = KPDetection.resize_img(cur_img,bbox,size)
            else:
                cur_img = np.zeros([size[1],size[0],3],dtype=np.float32)
            res.append(cur_img)
            res_bboxes.append(bbox)
        return res,np.array(res_bboxes,dtype=np.float32)
    
    @staticmethod
    def resize_img(img,bbox,target_size,pad_color=(127,127,127)):
        res = np.ndarray([target_size[1],target_size[0],3],dtype=np.uint8)
        res[:,:] = np.array(pad_color,dtype=np.uint8)
        ratio = target_size[0]/target_size[1]
        bbox_cx = (bbox[2]+bbox[0])/2
        bbox_cy = (bbox[3]+bbox[1])/2
        bbox_w = (bbox[2]-bbox[0])
        bbox_h = (bbox[3]-bbox[1])
        if img.shape[1]>ratio*img.shape[0]:
            nw = target_size[0]
            nh = int(target_size[0]*img.shape[0]/img.shape[1])
            bbox_h = bbox_w/ratio
        else:
            nh = target_size[1]
            nw = int(target_size[1]*img.shape[1]/img.shape[0])
            bbox_w = bbox_h*ratio
    
        img = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LINEAR)
        xoffset = (target_size[0]-nw)//2
        yoffset = (target_size[1]-nh)//2
        res[yoffset:yoffset+nh,xoffset:xoffset+nw] = img
        bbox = np.array([bbox_cx-bbox_w/2,bbox_cy-bbox_h/2,bbox_cx+bbox_w/2,bbox_cy+bbox_h/2],dtype=np.float32)
        return res,bbox

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

        pred_maskf = np.greater(maxvals, KPDetection.threshold).astype(np.float32)
        pred_mask = np.tile(np.greater(maxvals, KPDetection.threshold), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds, maxvals*pred_maskf

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

    def get_person_bboxes(self,img,return_ext_info=False):
        '''

        Args:
            img: RGB order

        Returns:
            ans: [N,17,3] (x,y,score,...)
        '''
        img = img[...,::-1]
        bboxes,probs = self.person_det(img)
        if len(probs) == 0:
            return np.zeros([0,4],dtype=np.float32),np.zeros([0,128],dtype=np.float32)
        if return_ext_info:
            img_patchs = self.cut_and_resizev0(img,bboxes.astype(np.int32),size=(64,128))
            img_patchs = np.array(img_patchs)
            img_patchs = img_patchs[...,::-1]
            embds = self.image_encoder(img_patchs)
            return bboxes,embds
        return bboxes

    def get_kps_by_bboxes(self,img,bboxes,return_fea=True):
        '''

        Args:
            img: RGB order

        Returns:
            ans: [N,17,3] (x,y,score,...)
        '''
        #print(bboxes)
        #cv2.imwrite("/home/wj/ai/mldata/0day/x1/a.jpg",img)
        imgs,bboxes = self.cut_and_resize(img,bboxes.astype(np.int32))
        #cv2.imwrite("/home/wj/ai/mldata/0day/x1/b.jpg",imgs[0])
        imgs = [self.transform(x) for x in imgs]
        imgs = [x.cpu().numpy() for x in imgs]
        imgs = np.ascontiguousarray(np.array(imgs))
        #print(imgs.shape)
        try:
            output = self.model.run(None, {self.input_name: imgs})[0]
        except Exception as e:
            print(f"ERROR")
            if return_fea:
                return np.zeros([imgs.shape[0],17,3],dtype=np.float32),None
            else:
                return np.zeros([imgs.shape[0],17,3],dtype=np.float32)
        output_fea = output 
        output = self.get_final_preds(output)
        offset,scalar = self.get_offset_and_scalar(bboxes)
        output[...,:2] = output[...,:2]*scalar+offset
        left_right_pairs = [[1,2],[3,4]]
        nr = output.shape[0]
        for i in range(nr):
            is_good = True
            for pair in left_right_pairs:
                l,r = pair
                if output[i,l,0]<output[i,r,0] or output[i,l,2]<KPDetection.threshold or output[i,r,2]<KPDetection.threshold:
                    is_good = False
                    break
            if not is_good:
                output[i,0] = 0.0
                for pair in left_right_pairs:
                    l,r = pair
                    output[i,l] = 0.0
                    output[i,r] = 0.0
        if return_fea:
            return output,output_fea,bboxes
        return output
    
    @staticmethod
    def trans_person_bboxes(bboxes,img_width=None,img_height=None):
        ratio = 288/384
        bboxes = odb.npto_cyxhw(bboxes)
        cx,cy,w,h = bboxes[...,0],bboxes[...,1],bboxes[...,2],bboxes[...,3]
        mask0 = w>(h*ratio)
        h0 = w/ratio
        mask1 = h>(w/ratio)
        w1 = h*ratio
        h = np.where(mask0,h0,h)
        w = np.where(mask1,w1,w)
        w = w*1.25
        h = h*1.25
        bboxes = np.stack([cx,cy,w,h],axis=-1)
        bboxes = odb.npto_yminxminymaxxmax(bboxes)
        bboxes = np.maximum(bboxes,0)
        if img_width is not None:
            bboxes[...,2] = np.minimum(bboxes[...,2],img_width-1)
        if img_height is not None:
            bboxes[...,3] = np.minimum(bboxes[...,3],img_height-1)
        return bboxes

    def __call__(self, img):
        '''

        Args:
            img: RGB order

        Returns:
            ans: [N,17,3] (x,y,score,...)
        '''
        bboxes = self.get_person_bboxes(img,return_ext_info=False)
        return self.get_kps_by_bboxes(img,bboxes)
