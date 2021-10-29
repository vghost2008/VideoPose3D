import numpy as np
from numpy.__config__ import show
from common.camera import normalize_screen_coordinates, image_coordinates
import os.path as osp
from predict_on_video.vis_error import update_camera
import wml_utils as wmlu
import img_utils as wmli
import cv2
from demo_toolkit import show_keypoints
from common.camera import *
from common.sem_dataset import *
from common.toolkit import get_offset

global_cam = {
   'id': '54138969',
        'center': [512.54150390625, 515.4514770507812],
        'focal_length': [1145.0494384765625, 1143.7811279296875],
        'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
        'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': 70, # Only used for visualization
        'orientation': [0.1337897777557373, -0.15692396461963654, -0.7571090459823608, 0.6198879480361938],
        'translation': [1935.4517822265625, 4950.24560546875, 1618.0838623046875],
}


def load_data(kp2d_path='data/data_2d_h36m_detectron_pt_coco.npz',kp3d_path='data/data_3d_h36m.npz'):
    #keypoints = np.load(kp2d_path, allow_pickle=True)
    keypoints = np.load('data/data_2d_h36m_gt.npz', allow_pickle=True)
    kps2d = keypoints['positions_2d'].item()
    kps3d = np.load(kp3d_path, allow_pickle=True)['positions_3d'].item()

    return kps2d,kps3d

def update_cam(cam):
    for k, v in cam.items():
        if k not in ['id', 'res_w', 'res_h']:
            cam[k] = np.array(v, dtype='float32')
                
                # Normalize camera frame
    cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype('float32')
    cam['focal_length'] = cam['focal_length']/cam['res_w']*2
    if 'translation' in cam:
        cam['translation'] = cam['translation']/1000 # mm to meters
    
    # Add intrinsic parameters vector
    cam['intrinsic'] = np.concatenate((cam['focal_length'], #2
                                       cam['center'], #2
                                       cam['radial_distortion'], #3
                                       cam['tangential_distortion'])) #2
    return cam

if __name__ == "__main__":
    subject = 'S6'
    action = 'Walking'
    #subject = 'S1'
    #action = 'Directions 1'
    save_dir = "/home/wj/ai/mldata/pose3d/tmp/vis_human3.6"
    video_path = "/home/wj/ai/mldata/human3.6/S6/Videos/Walking.58860488.mp4"
    video_path = "/home/wj/ai/mldata/human3.6/S6/Videos/Walking.54138969.mp4"
    keep_ids = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19,25,26,27]
    wmlu.create_empty_dir(save_dir,remove_if_exists=False)
    save_path = osp.join(save_dir,osp.basename(video_path))
    kps2d,kps3d = load_data()
    kps2d = kps2d[subject][action]
    kps3d = kps3d[subject][action]
    cam = update_cam(global_cam)
    #use first camera
    kps3d = world_to_camera(kps3d,R=cam['orientation'], t=cam['translation'])
    kps3d = kps3d[:,keep_ids,:]
    kps2d = kps2d[0][...,:2]
    #kps2d = kps2d[:,keep_ids,:]
    kps2d[:,SemDataset.COCO_ID_VALID,:] = kps2d[:,SemDataset.H36M_ID_VALID,:]


    video_writer = wmli.VideoWriter(save_path,fmt='BGR')
    reader = cv2.VideoCapture(video_path)
    idx = 0
    max_frames = 1000
    proj_2ds = npproject_to_2d(np.array([kps3d]),np.array([cam['intrinsic']]))[0]
    proj_2ds = unnormalize_screen_coordinates(proj_2ds,w=cam['res_w'],h=cam['res_h'])
    scores = np.ones(list(proj_2ds.shape[:-1])+[1])
    scores = scores*SemDataset.COCO_MASK[0]
    valid_len = min(proj_2ds.shape[0],kps2d.shape[0]) 
    proj_2ds = proj_2ds[:valid_len]
    kps2d = kps2d[:valid_len]
    offset = get_offset(kps2d,kps3d,cam)
    #kps2d[...,:2] = kps2d[...,:2]+offset
    proj_2ds[:,SemDataset.COCO_ID_VALID,:] = proj_2ds[:,SemDataset.H36M_ID_VALID,:]
    proj_2ds = proj_2ds-offset
    proj_2ds = np.concatenate([proj_2ds,scores],axis=-1)
    kps2d = np.concatenate([kps2d,scores],axis=-1)
    print(f"Offset {offset}.")
    error = np.abs(kps2d[...,:2][SemDataset.COCO_ID_VALID]-proj_2ds[...,:2][SemDataset.COCO_ID_VALID])
    error = np.mean(error)
    print(f"Error: {error}")

    while True:
        ret,frame = reader.read()
        if not ret or frame is None:
            break
        kp2d = kps2d[idx]
        kp3d = kps3d[idx]
        proj_2d = proj_2ds[idx]
        frame = show_keypoints(frame,kp2d)
        frame = show_keypoints(frame,proj_2d,color=(255,0,0))
        video_writer.write(frame)
        idx += 1
        if max_frames is not None and idx>max_frames:
            break
    video_writer.release()
    print(f"Save path {save_path}")
        
    

    
