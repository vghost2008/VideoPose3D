import argparse
import tensorflow as tf
from track_keypointsv2 import *
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
from data import tmp_kp_data1
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
matplotlib.use('Agg')

tf.enable_eager_execution()

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
        #filter_widths = [3,3,3]
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
        self.cam = {'id': '54138969',
        'center': [512.54150390625, 515.4514770507812],
        'focal_length': [1145.0494384765625, 1143.7811279296875],
        'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
        'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
        #'res_w': 1000,
        #'res_h': 1002,
        'res_w': 1280,
        'res_h': 720,
        'azimuth': 70, # Only used for visualizatio
        'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
        'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        }
        self.use_scores = use_scores

    def update_camera(self,img):
        self.cam['res_h'] = img.shape[0]
        self.cam['res_w'] = img.shape[1]

    @staticmethod
    def normalize_screen_coordinates(X, w, h): 
        assert X.shape[-1] == 2
    
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        return X/w*2 - [1, h/w]
    
    @staticmethod
    def get_offset(data):
        '''
        data: [N,17]
        '''
        data = data[:,0]
        window = 60
        if len(data)<=window:
            v = np.mean(data)
            offset = np.array([v]*len(data),dtype=np.float32)
            offset = np.expand_dims(offset,axis=-1)
            return offset
        else:
            v = np.mean(data[:window])
            offset = [v]*window
            for i in range(window,len(data)):
                v = v+(data[i]-data[i-window])/window
                offset.append(v)
            offset = np.array(offset,dtype=np.float32)
            offset = np.expand_dims(offset,axis=-1)
            return offset

    @staticmethod
    def trans_kps(kps):
        x = kps[...,0]
        y = kps[...,1]
        xoffset = VideoPose3DModel.get_offset(x)
        yoffset = VideoPose3DModel.get_offset(y)
        x = x-xoffset
        y = y-yoffset
        minx = np.min(x)
        maxx = np.max(x)
        miny = np.min(y)
        maxy = np.max(y)
        width = maxx-minx+1e-4
        height = maxy-miny+1e-4
        r = min(1.0/width,1.0/height)
        offset = np.array([-(maxx+minx)/2,-(maxy+miny)/2],dtype=np.float32)
        offset = np.reshape(offset,[1,1,2]) 
        print(f"Scalar: {r}")
        return r,offset,np.stack([x,y],axis=-1)


    def __call__(self,kps,scale_data=False,flip=True):
        '''
        kps: [N,17,2+x]
        '''
        kps = tmp_kp_data1
        kps = np.array(kps)
        if self.use_scores:
            scores = (np.array(kps)[...,2:]>0.015).astype(np.float32)
            scores = np.expand_dims(scores,axis=0)
            scores = np.pad(scores,[[0,0],[self.pad,self.pad],[0,0],[0,0]],mode='edge')
        kps = kps[...,:2]

        data0 = [kps[0]]*self.pad
        data1 = [kps[-1]]*self.pad
        data_org = np.concatenate([np.array(data0),np.array(kps),np.array(data1)],axis=0)
        cam = self.cam

        if flip:
            data_agu = copy.deepcopy(data_org)
            data = np.stack([data_org,data_agu],axis=0)
            data = self.normalize_screen_coordinates(data,cam['res_w'],cam['res_h'])
            data_traj = data
        elif scale_data:
            data = self.normalize_screen_coordinates(data_org,cam['res_w'],cam['res_h'])
            data_traj = np.array(data)
            r,offset,new_data = self.trans_kps(data)
            data[...,:2] = new_data
            data[...,:2] = (data[...,:2]+offset)*r
            data = np.expand_dims(data,axis=0)
            data_traj = np.expand_dims(data_traj,axis=0)
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

        if False and self.model_traj is not None:
            offset = self.model_traj(data_traj)
            offset = offset.cpu().detach().numpy()
            if flip:
                offset[1,:,:,0] = offset[1,:,:,0]*-1
                offset = np.mean(offset,axis=0,keepdims=True)
            pos_3d = pos_3d+offset

        res = np.squeeze(pos_3d,axis=0)
        xmin,xmax = np.min(res[...,0],axis=1),np.max(res[...,0],axis=1)
        ymin,ymax = np.min(res[...,1],axis=1),np.max(res[...,1],axis=1)
        zmin,zmax = np.min(res[...,2],axis=1),np.max(res[...,2],axis=1)
        bboxes = np.stack([xmax-xmin,ymax-ymin,zmax-zmin],axis=-1)
        axmin = np.min(xmin)
        axmax = np.max(xmax)
        aymin = np.min(ymin)
        aymax = np.max(ymax)
        azmin = np.min(zmin)
        azmax = np.max(zmax)
        wmlu.show_list(bboxes)
        print(axmin,axmax,aymin,aymax,azmin,azmax)
        return res

class RenderAnimation:
    def __init__(self,frames,pos_3d,keypoints) -> None:
        self.frames = frames
        self.pos_3d = np.array(pos_3d)
        self.keypoints = np.array(keypoints)
        self.joints_left = [4, 5, 6, 11, 12, 13]
        self.joints_right = [1, 2, 3, 14, 15, 16]
        self.cam = {'id': '54138969',
        'center': [512.54150390625, 515.4514770507812],
        'focal_length': [1145.0494384765625, 1143.7811279296875],
        'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
        'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
        #'res_w': 1000,
        #'res_h': 1002,
        'res_w': 1280,
        'res_h': 720,
        'azimuth': 70, # Only used for visualizatio
        'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
        'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        }

    def update_camera(self,img):
        self.cam['res_h'] = img.shape[0]
        self.cam['res_w'] = img.shape[1]

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
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
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

        for i in range(len(pos_3d)):
            update_video(i)
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = np.array(all_frames[i])
            show_keypoints(img,keypoints[i])
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
    video_path ='/home/wj/ai/mldata/pose3d/tennis1.mp4'
    #video_path = '/home/wj/ai/mldata/pose3d/basketball1.mp4'
    #video_path = '/home/wj/ai/mldata/pose3d/basketball2.mp4'
    #video_path = '/home/wj/ai/mldata/pose3d/basketball3.mp4'
    #video_path = "/home/wj/ai/mldata/human3.6/S6/Videos/Walking.58860488.mp4"
    use_scores = True
    if use_scores:
        suffix = "_v4"
    else:
        suffix = "_v1"
    save_dir = "/home/wj/ai/mldata/pose3d/tmp/predict_on_video_"+wmlu.base_name(video_path)+suffix
    cache_dir = "/home/wj/ai/mldata/pose3d/tmp/cache"
    #ckpt_pos = 'weights/pretrained_h36m_detectron_coco.bin'
    ckpt_pos = 'weights/epoch_80.bin'
    ckpt_traj = 'weights/epoch_80.bin'
    if use_scores:
        #ckpt_pos = 'weights_semv3/epoch_80.bin'
        ckpt_pos = 'weights_semv4/epoch_100.bin'
    else:
        ckpt_pos = 'weights_sem/epoch_30.bin'
        ckpt_pos = 'weights/epoch_80.bin'
    ckpt_traj = ckpt_pos
    #ckpt_traj = None
    video_pos_3d = VideoPose3DModel(ckpt_pos=ckpt_pos,ckpt_traj=ckpt_traj,use_scores=use_scores)

    if 'tmp' in save_dir:
        wmlu.create_empty_dir(save_dir,True,True)
    else:
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
        save_path = osp.join(save_dir,f"{min_idx}_{max_idx}_{tid}.mp4")

        all_keypoints = []
        all_frames = []
        for i in range(min_idx,max_idx):
            if i not in keypoints:
                continue
            cur_frame = frames[i]
            all_frames.append(cur_frame)
            all_keypoints.append(keypoints[i])
        pos_3d = video_pos_3d(all_keypoints,scale_data=True,flip=False)
        render_ani = RenderAnimation(all_frames,pos_3d,all_keypoints)
        render_ani.update_camera(frames[0])
        render_ani.render_animation(save_path=save_path)
    print(f"Save path {save_dir}")
