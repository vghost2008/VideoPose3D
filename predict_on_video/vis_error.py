import argparse
import tensorflow as tf
from track_keypoints import *
from demo_toolkit import *
import tensorflow as tf
import numpy as np
import os.path as osp
import wml_utils as wmlu
from matplotlib.animation import FuncAnimation
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import os
from common.model import TemporalModel
from common.camera import *
import matplotlib.pyplot as plt
import pickle
from collections import Iterable
import copy
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
matplotlib.use('Agg')

tf.enable_eager_execution()
global_cam = {'id': '54138969',
'center': [512.54150390625, 515.4514770507812],
'focal_length': [1145.0494384765625, 1143.7811279296875],
'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
'res_w': 1000,
'res_h': 1002,
#'res_w': 1280,
#'res_h': 720,
'azimuth': 70, # Only used for visualizatio
'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
}
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


def process_cam(cam):
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
def update_camera(img,cam):
    if img is not None:
        #cam['res_h'] = img.shape[0]
        #cam['res_w'] = img.shape[1]
        pass
    process_cam(cam)

def parse_args():
    parser = argparse.ArgumentParser(description='Predict script')
    parser.add_argument('-v', '--video', default='/home/wj/ai/mldata/pose3d/tennis1.mp4', type=str, metavar='NAME',
    #parser.add_argument('-v', '--video', default='/home/wj/ai/mldata/pose3d/basketball2.mp4', type=str, metavar='NAME',
                        help='target dataset')  # h36m or humaneva
    parser.add_argument('-s', '--save_dir', default='/home/wj/ai/0day/b', type=str, metavar='NAME',
                        help='save data dir path')  # h36m or humaneva
    args = parser.parse_args()
    return args


tf.enable_eager_execution()


class VideoPose3DModel:
    def __init__(self,ckpt_pos,ckpt_traj=None,use_scores=False) -> None:
        self.device = torch.device("cuda:0")
        filter_widths = [3,3,3,3,3]
        filter_widths = [3,3,3,3]
        filter_widths = [3,3,3]
        if use_scores:
            in_features = 3
        else:
            in_features = 2
        self.model_pos = TemporalModel(num_joints_in=17,
                     in_features=in_features,num_joints_out=17,
                     filter_widths=filter_widths,
                     causal=False,
                     channels=1024,dense=False)
        checkpoint = torch.load(ckpt_pos)

        self.model_pos.load_state_dict(checkpoint['model_pos'])
        self.model_pos.to(self.device)
        self.model_pos.eval()
        self.pad = (self.model_pos.receptive_field()-1)//2
        if ckpt_traj is not None:
            checkpoint = torch.load(ckpt_traj)
        if 'model_traj' in checkpoint:
            self.model_traj = TemporalModel(num_joints_in=17,
                     in_features=in_features,num_joints_out=1,
                     filter_widths=filter_widths,
                     causal=False,
                     channels=1024,dense=False)
            self.model_traj.load_state_dict(checkpoint['model_traj'])
            self.model_traj.to(self.device)
            self.model_traj.eval()
        else:
            self.model_traj = None
        self.kps_left = [1, 3, 5, 7, 9, 11, 13, 15]
        self.kps_right = [2, 4, 6, 8, 10, 12, 14, 16]
        self.joints_left = [4, 5, 6, 11, 12, 13]
        self.joints_right = [1, 2, 3, 14, 15, 16]
        self.cam = copy.deepcopy(global_cam)
        self.use_scores = use_scores
        self.good_id0 = []
        self.good_id1 = []
        self.unused_id0 = []
        self.unused_id1 = []
        for i,v in enumerate(rid_map):
            if v>=0 :
                self.good_id0.append(i)
                self.good_id1.append(v)
            else:
                self.unused_id0.append(i)
        for i in range(len(rid_map)):
            if i not in self.good_id1:
                self.unused_id1.append(i)

    def update_camera(self,img):
        update_camera(img,self.cam)

    @staticmethod
    def normalize_screen_coordinates(X, w, h): 
        assert X.shape[-1] == 2
    
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        return X/w*2 - [1, h/w]

    @staticmethod
    def unnormalize_screen_coordinates(X, w, h): 
        assert X.shape[-1] == 2
    
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        return (X + [1, h/w])*w/2
    
    def __call__(self,kps,flip=True):
        '''
        kps: [N,17,2+x]
        '''
        kps = np.array(kps)
        scores = (np.array(kps)[...,2:]>0.015).astype(np.float32)
        scores_mask = np.array(scores)
        if self.use_scores:
            scores = np.expand_dims(scores,axis=0)
            scores = np.pad(scores,[[0,0],[self.pad,self.pad],[0,0],[0,0]],mode='edge')
        kps = kps[...,:2]
        org_kps = np.array(kps)

        data0 = [kps[0]]*self.pad
        data1 = [kps[-1]]*self.pad
        data_org = np.concatenate([np.array(data0),np.array(kps),np.array(data1)],axis=0)
        cam = self.cam

        if flip:
            data_agu = copy.deepcopy(data_org)
            data = np.stack([data_org,data_agu],axis=0)
            data = self.normalize_screen_coordinates(data,cam['res_w'],cam['res_h'])
            data_traj = data
        else:
            data = data_org
            data = self.normalize_screen_coordinates(data_org,cam['res_w'],cam['res_h'])
            data = np.expand_dims(data,axis=0)
            data_traj = data

        if flip:
            data[1,:,:,0] = data[1,:,:,0]*-1
            data[1,:,self.kps_left+self.kps_right] = data[1,:,self.kps_right+self.kps_left]
            data_traj[1,:,:,0] = data_traj[1,:,:,0]*-1
            data_traj[1,:,self.kps_left+self.kps_right] = data_traj[1,:,self.kps_right+self.kps_left]
            if self.use_scores:
                scores = np.tile(scores,[2,1,1,1])
        
        if self.use_scores:
            data = np.concatenate([data,scores],axis=-1)
            data_traj = np.concatenate([data_traj,scores],axis=-1)

        data = torch.tensor(data,dtype=torch.float32,device=self.device)
        data_traj = torch.tensor(data_traj,dtype=torch.float32,device=self.device)

        pos_3d = self.model_pos(data).cpu().detach().numpy()
        if flip:
            pos_3d[1,:,:,0] = pos_3d[1,:,:,0]*-1
            pos_3d[1,:,self.joints_right+self.joints_left,:] = pos_3d[1,:,self.joints_left+self.joints_right,:]

            pos_3d = np.mean(pos_3d,axis=0,keepdims=True)

        if self.model_traj is not None:
            offset = self.model_traj(data_traj)
            offset = offset.cpu().detach().numpy()
            if flip:
                offset[1,:,:,0] = offset[1,:,:,0]*-1
                offset = np.mean(offset,axis=0,keepdims=True)
            pos_3d = pos_3d+offset
        #projection_func = project_to_2d_linear 
        projection_func = npproject_to_2d
        pred_2d = projection_func(pos_3d, np.expand_dims(self.cam['intrinsic'],axis=0))
        pred_2d = np.squeeze(pred_2d,axis=0)
        pred_2d = self.unnormalize_screen_coordinates(pred_2d,cam['res_w'],cam['res_h'])

        pred_2d[:,self.good_id1] = pred_2d[:,self.good_id0]
        mask = np.ones([1,17,1],dtype=np.float32)
        mask[:,self.unused_id1] = 0
        shape = pred_2d.shape
        scores = np.ones([shape[0],shape[1],1],dtype=np.float32)
        scores[:,self.unused_id1] = 0

        dis1 = pred_2d[:,joints_pair_a]-pred_2d[:,joints_pair_b]
        dis2_mask = scores_mask[:,joints_pair_a]*scores_mask[:,joints_pair_b]
        dis2 = org_kps[:,joints_pair_a]-org_kps[:,joints_pair_b]
        dis2 = np.linalg.norm(dis2,2,axis=-1,keepdims=True)*dis2_mask
        dis1 = np.linalg.norm(dis1,2,axis=-1,keepdims=True)*dis2_mask
        dis1 = np.sum(dis1,axis=1,keepdims=True)
        dis2 = np.sum(dis2,axis=1,keepdims=True)
        r = dis2/dis1
        pred_2d = pred_2d*r

        r_mask = mask*scores_mask
        nr = np.sum(r_mask,axis=1,keepdims=True)
        base_offset = (pred_2d-org_kps)*mask*scores_mask
        base_offset = np.sum(base_offset,axis=1,keepdims=True)/nr
        pred_2d = pred_2d-base_offset


        error = np.mean(np.abs(pred_2d-org_kps)*r_mask)
        print(f"Error {error}")

        return np.squeeze(pos_3d,axis=0),np.concatenate([pred_2d,scores],axis=-1)

class RenderAnimation:
    def __init__(self,frames,pos_3d,keypoints,pred_keypoints) -> None:
        self.frames = frames
        self.pos_3d = np.array(pos_3d)
        self.keypoints = np.array(keypoints)
        self.pred_keypoints = np.array(pred_keypoints)
        self.joints_left = [4, 5, 6, 11, 12, 13]
        self.joints_right = [1, 2, 3, 14, 15, 16]
        self.cam = copy.deepcopy(global_cam)

    def update_camera(self,img):
        update_camera(img,self.cam)

    def render_animation(self, 
                         limit=-1, size=6, save_path=""):
        """
        TODO
        Render an animation. The supported output modes are:
         -- 'interactive': display an interactive figure
                           (also works on notebooks if associated with %matplotlib inline)
         -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
         -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
         -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
        """
        keypoints = self.keypoints
        all_frames = self.frames
        pos_3d = self.pos_3d
        rot = np.array(self.cam['orientation'],dtype=np.float32)
        pos_3d = camera_to_world(pos_3d, R=rot, t=0)
        # We don't have the trajectory, but at least we can rebase the height
        #prediction[:, :, 2] -= np.min(prediction[:, :, 2])
        #keypoints[...,:2] = image_coordinates(keypoints[..., :2], w=self.cam['res_w'], h=self.cam['res_h'])
        azim = self.cam['azimuth']
        viewport=(self.cam['res_w'], self.cam['res_h'])


        plt.ioff()
        fig = plt.figure(figsize=(size, size))
    
        ax_3d = []
        lines_3d = []
        trajectories = []
        radius = 1.7

        xmin,xmax = np.min(pos_3d[...,0]),np.max(pos_3d[...,0])
        ymin,ymax = np.min(pos_3d[...,1]),np.max(pos_3d[...,1])
        zmin,zmax = np.min(pos_3d[...,2]),np.max(pos_3d[...,2])

        for index in range(1):
            ax = fig.add_subplot(1, 1, index+1, projection='3d')
            ax.view_init(elev=15., azim=azim)

            '''ax.set_xlim3d([-radius/2, radius/2])
            ax.set_zlim3d([0, radius])
            ax.set_ylim3d([-radius/2, radius/2])'''

            ax.set_xlim3d([-radius/2+xmin, radius/2+xmax])
            ax.set_zlim3d([0+zmin, radius+zmax])
            ax.set_ylim3d([-radius/2+ymin, radius/2+ymax])

            try:
                ax.set_aspect('equal')
            except NotImplementedError:
                ax.set_aspect('auto')

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.dist = 7.5
            ax.set_title(f"Data") #, pad=35
            ax_3d.append(ax)
            lines_3d.append([])
    
        # Load video using ffmpeg
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]
            
        initialized = False
        
        if limit < 1:
            limit = len(all_frames)
        else:
            limit = min(limit, len(all_frames))
    
        parents = np.array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])

        def update_video(i):
            nonlocal initialized
    
            '''for n, ax in enumerate(ax_3d):
                ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
                ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])'''
    
            # Update 2D poses
            joints_right_2d = self.joints_right
            colors_2d = np.full(keypoints.shape[1], 'black')
            colors_2d[joints_right_2d] = 'red'
            if not initialized:
                
                for j, j_parent in enumerate(parents):
                    if j_parent == -1:
                        continue
                        
                    col = 'red' if j in self.joints_right else 'black'
                    pos = pos_3d[i]
                    for n, ax in enumerate(ax_3d):
                        lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                                   [pos[j, 1], pos[j_parent, 1]],
                                                   [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
    
                initialized = True
            else:

    
                for j, j_parent in enumerate(parents):
                    if j_parent == -1:
                        continue
                    
                    pos = pos_3d[i]
                    for n, ax in enumerate(ax_3d):
                        lines_3d[n][j-1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                        lines_3d[n][j-1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                        lines_3d[n][j-1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')
    
            
            print('{}/{}      '.format(i, limit), end='\r')
            
    
        fig.tight_layout()

        writer = None

        for i in range(len(keypoints)):
            update_video(i)
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = np.array(all_frames[i])
            show_keypoints(img,keypoints[i])
            #show_keypoints(img,self.pred_keypoints[i],color=(255,0,0))
            show_keypoints_diff(img,keypoints[i],self.pred_keypoints[i],color=(255,0,0))
            image_from_plot = resize_height(image_from_plot,h=img.shape[0])
            img = np.concatenate([img,image_from_plot],axis=1)
            if writer is None:
                writer = self.init_writer(save_path,img.shape[:2][::-1])
            writer.write(img)

        writer.release()
        
        plt.close()

 
    @staticmethod
    def init_writer(save_path,write_size):
        print(f"Save path {save_path}")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(save_path, fourcc, 30,write_size)
        return video_writer

if __name__ == "__main__":
    args = parse_args()
    video_path = args.video
    video_path = "/home/wj/ai/mldata/human3.6/S6/Videos/Walking.54138969.mp4"
    video_path = "/home/wj/ai/mldata/totalcapture/s1_freestyle1/freestyle1/TC_S1_freestyle1_cam1.mp4"
    use_scores = False
    if use_scores:
        suffix = "_v3"
    else:
        suffix = "_v1"
    save_dir = "/home/wj/ai/mldata/pose3d/tmp/vis_error"
    cache_dir = "/home/wj/ai/mldata/pose3d/tmp/cache"
    #ckpt_pos = 'weights/pretrained_h36m_detectron_coco.bin'
    ckpt_pos = 'weights/epoch_80.bin'
    ckpt_traj = 'weights/epoch_80.bin'
    if use_scores:
        ckpt_pos = 'weights_semv3/epoch_10.bin'
    else:
        ckpt_pos = 'weights_sem/epoch_30.bin'
        ckpt_pos = 'weights/epoch_80.bin'
    ckpt_traj = ckpt_pos
    name_suffix = osp.dirname(ckpt_pos)+"_"+osp.basename(ckpt_pos)+suffix
    #ckpt_traj = None
    video_pos_3d = VideoPose3DModel(ckpt_pos=ckpt_pos,ckpt_traj=ckpt_traj,use_scores=use_scores)

    wmlu.create_empty_dir(save_dir,False)
    wmlu.create_empty_dir(cache_dir,False)

    cache_path = osp.join(cache_dir,osp.basename(video_path))

    if osp.exists(cache_path):
        with open(cache_path,"rb") as f:
            all_data = pickle.load(f)
        sorted_kps = all_data['kps']
        all_frames = all_data['frames']
    else:
        track_model = TrackKeypoints(video_path)
        all_frames = track_model.track_keypoints(return_frames=True,max_frames_nr=None)
        sorted_kps = track_model.sorted_keypoints
        all_data = {}
        all_data['kps'] = sorted_kps
        all_data['frames'] = all_frames
        with open(cache_path,"wb") as f:
            pickle.dump(all_data,f)
        
    frames = all_frames
    video_pos_3d.update_camera(frames[0])

    for tid,keypoints in sorted_kps:
        idxs = np.array(list(keypoints.keys()))
        if len(idxs)<30:
            continue
        min_idx = np.min(idxs)
        max_idx = np.max(idxs)
        max_idx = min(min_idx+1000,max_idx)
        save_path = osp.join(save_dir,f"{name_suffix}_{min_idx}_{max_idx}_{tid}.mp4")

        all_keypoints = []
        all_frames = []
        for i in range(min_idx,max_idx):
            if i not in keypoints:
                continue
            cur_frame = frames[i]
            all_frames.append(cur_frame)
            all_keypoints.append(keypoints[i])
        pos_3d,pred_2d = video_pos_3d(all_keypoints,flip=False)
        render_ani = RenderAnimation(all_frames,pos_3d,all_keypoints,pred_2d)
        render_ani.update_camera(frames[0])
        render_ani.render_animation(save_path=save_path)
    print(f"Save path {save_dir}")
