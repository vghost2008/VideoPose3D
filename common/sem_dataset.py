import numpy as np
import copy
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates, image_coordinates
from common.h36m_dataset import h36m_skeleton
from .totalcapture_cameras import totalcapture_cameras
from collections import Iterable
import wml_utils as wmlu
import os.path as osp
import pickle
'''
COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}
 H36M_NAMES = ['']*32
  H36M_NAMES[0]  = 'Hip'
  H36M_NAMES[1]  = 'RHip'
  H36M_NAMES[2]  = 'RKnee'
  H36M_NAMES[3]  = 'RFoot'
  H36M_NAMES[4]  = 'RFootTip'
  H36M_NAMES[6]  = 'LHip'
  H36M_NAMES[7]  = 'LKnee'
  H36M_NAMES[8]  = 'LFoot'
  H36M_NAMES[12] = 'Spine'
  H36M_NAMES[13] = 'Thorax'
  H36M_NAMES[14] = 'Neck/Nose'
  H36M_NAMES[15] = 'Head'
  H36M_NAMES[17] = 'LShoulder'
  H36M_NAMES[18] = 'LElbow'
  H36M_NAMES[19] = 'LWrist'
  H36M_NAMES[25] = 'RShoulder'
  H36M_NAMES[26] = 'RElbow'
  H36M_NAMES[27] = 'RWrist'
'''
keep_joints=[0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19,25,26,27] 
id_map = {}
id_map[3] = 16
id_map[8] = 15
id_map[2] = 14
id_map[7] = 13
id_map[1] = 12
id_map[6] = 11
id_map[19] = 9
id_map[27] = 10
id_map[26] = 8
id_map[18] = 7
id_map[25] = 6
id_map[17] = 5
id_map[14] = 0
rid_map = [-1]*17

for k,v in id_map.items():
    idx = keep_joints.index(k)
    rid_map[idx] = v

joints_pair = [[5 , 6], [5 , 11],
[6 , 12], [11 , 12], [5 , 7], [7 , 9], [6 , 8], [8 , 10], [11 , 13], [13 , 15], [12 , 14], [14 , 16]]
joints_pair_a,joints_pair_b = [list(x) for x in zip(*joints_pair)]
good_id0 = []
good_id1 = []
unused_id0 = []
unused_id1 = []
for i,v in enumerate(rid_map):
    if v>=0 :
       good_id0.append(i)
       good_id1.append(v)
    else:
        unused_id0.append(i)
for i in range(len(rid_map)):
    if i not in good_id1:
        unused_id1.append(i)
coco_mask = np.ones([1,1,17,1],dtype=np.float32)
coco_mask[:,:,unused_id1] = 0

class SemDataset(MocapDataset):
    PAIRS_A = joints_pair_a
    PAIRS_B = joints_pair_b
    H36M_ID_VALID = good_id0
    COCO_ID_VALID = good_id1
    H36M_ID_UNVALID = unused_id0
    COCO_ID_UNVALID = unused_id1
    COCO_MASK = coco_mask
    def __init__(self, detections_path='/home/wj/ai/mldata/totalcapture', remove_static_joints=True):
        super().__init__(fps=None, skeleton=h36m_skeleton)        
        self.poses_2d = None
        self.cameras_data = None

        # Load serialized dataset
        
        self._cameras = copy.deepcopy(totalcapture_cameras)
        for i, cam in enumerate(self._cameras):
            if not isinstance(cam['radial_distortion'],Iterable):
                cam['radial_distortion'] = [cam['radial_distortion'],0.0,0.0]
            for k, v in cam.items():
                if k not in ['id', 'res_w', 'res_h']:
                    cam[k] = np.array(v, dtype='float32')
            if 'tangential_distortion' not in cam:
                cam['tangential_distortion'] = [0.0,0.0]
            
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

        self.load_data(detections_path)

        if remove_static_joints:
            # Rewire shoulders to the correct parents
            self._skeleton._parents[11] = 8
            self._skeleton._parents[14] = 8
            
    def supports_semi_supervised(self):
        return True 

    def load_data(self,data_dir,suffix=".npz"):
        self.poses_2d = []
        self.cameras_data = []
        
        all_files = wmlu.recurse_get_filepath_in_dir(data_dir,suffix)
        for file in all_files:
            with open(file,"rb") as f:
                kps = pickle.load(f)
            if len(kps)<100:
                continue
            cam = self.get_camera(file)
            kps[...,:2] = normalize_screen_coordinates(kps[...,:2], w=cam['res_w'], h=cam['res_h']).astype('float32')
            self.poses_2d.append(kps)
            self.cameras_data.append(cam['intrinsic'])

        print(f"Total load {len(self.poses_2d)} files.")
    
    def get_camera(self,path):
        basename = osp.basename(path)
        name = basename.split("_")
        cam_idx = int(name[3][3:])-1
        print(f"Cam for {basename} is {cam_idx}.")
        return self._cameras[cam_idx]


   