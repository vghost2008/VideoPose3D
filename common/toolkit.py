import numpy as np
from .camera import *
from .sem_dataset import *

def get_offset(kps2d,kps3d,cam):
    '''
    kps2d:[N,17,2],unormalized screen coordinate
    kps3d:[N,17,3]
    return:
    offset:[1,17,2], proj_2d-offset will be more correct
    '''
    kps3d = np.array(kps3d)
    kps3d[:,SemDataset.COCO_ID_VALID,:] = kps3d[:,SemDataset.H36M_ID_VALID,:]
    proj_2ds = npproject_to_2d(np.array([kps3d]),np.array([cam['intrinsic']]))[0]
    proj_2ds = unnormalize_screen_coordinates(proj_2ds,w=cam['res_w'],h=cam['res_h'])
    scores = np.ones(list(proj_2ds.shape[:-1])+[1])
    scores = scores*SemDataset.COCO_MASK[0]
    valid_len = min(proj_2ds.shape[0],kps2d.shape[0]) 
    proj_2ds = proj_2ds[:valid_len]
    kps2d = kps2d[:valid_len]
    inv_w = np.sum(scores,0,keepdims=True)+1e-1
    inv_w = 1.0/inv_w
    offset = np.sum((proj_2ds-kps2d)*scores,0,keepdims=True)*inv_w
    return offset

def get_mask_for_train(batch_size,seq_len,kps_nr):
    probs_l0 = 0.1
    probs_l1 = 0.3
    #total probs = probs_l0*probs_l1=0.03
    mask_p0 = np.random.rand(batch_size,seq_len)>probs_l0
    mask_p0 = np.expand_dims(mask_p0,axis=-1)
    mask_p0 = np.tile(mask_p0,[1,1,kps_nr])
    mask_p1 = np.random.rand(batch_size,seq_len,kps_nr)>probs_l1 
    mask_p = np.logical_or(mask_p0,mask_p1)
    return mask_p
