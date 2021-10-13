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

class SemDataset(MocapDataset):
    def __init__(self, detections_path='/home/wj/ai/mldata/totalcapture', remove_static_joints=True):
        super().__init__(fps=None, skeleton=h36m_skeleton)        
        self.poses_2d = None
        self.cameras_data = None

        # Load serialized dataset
        
        self._cameras = copy.deepcopy(totalcapture_cameras)
        for i, cam in enumerate(self._cameras):
            for k, v in cam.items():
                if not isinstance(cam['radial_distortion'],Iterable):
                    cam['radial_distortion'] = [cam['radial_distortion'],0.0,0.0]
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


   