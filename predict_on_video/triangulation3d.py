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
import cv2
import slam.camera_toolkit as camt
import img_utils as wmli
from data import tmp_kp_data1
from iotoolkit.coco_toolkit import JOINTS_PAIR
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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
all_cameras = [
    {
        'id': '55011271',
        'center': [508.8486328125, 508.0649108886719],
        'focal_length': [1149.6756591796875, 1147.5916748046875],
        'radial_distortion': [-0.1942136287689209, 0.2404085397720337, 0.006819975562393665],
        'tangential_distortion': [-0.0016190266469493508, -0.0027408944442868233],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': -70, # Only used for visualization
    },
    {
        'id': '60457274',
        'center': [514.9682006835938, 501.88201904296875],
        'focal_length': [1145.5113525390625, 1144.77392578125],
        'radial_distortion': [-0.198384091258049, 0.21832367777824402, -0.008947807364165783],
        'tangential_distortion': [-0.0005872055771760643, -0.0018133620033040643],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': -110, # Only used for visualization
    }
]

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
        pos_3d = self.pos_3d.astype(np.float32)
        rot = np.array(self.cam['orientation'],dtype=np.float32)
        if pos_3d.shape[-1] == 4:
            score = pos_3d[...,-1:]
            pos_3d = camera_to_world(pos_3d[...,:-1], R=rot, t=0)
            pos_3d = np.concatenate([pos_3d,score],axis=-1)
        else:
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
        radius = 0.1

        xmin,xmax = np.min(pos_3d[...,0]),np.max(pos_3d[...,0])
        ymin,ymax = np.min(pos_3d[...,1]),np.max(pos_3d[...,1])
        zmin,zmax = np.min(pos_3d[...,2]),np.max(pos_3d[...,2])
        hwidth = max(max(xmax-xmin,ymax-ymin),zmax-zmin)/2
        cx = (xmax+xmin)/2
        xmin = cx-hwidth
        xmax = cx+hwidth
        cy = (ymax+ymin)/2
        ymin = cy-hwidth
        ymax = cy+hwidth
        cz = (zmax+zmin)/2
        zmin = cz-hwidth
        zmax = cz+hwidth

        for index in range(1):
            ax = fig.add_subplot(1, 1, index+1, projection='3d')
            ax.view_init(elev=15., azim=azim)

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
    
        def update_video(i):
            nonlocal initialized
    
            '''for n, ax in enumerate(ax_3d):
                ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
                ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])'''
    
            # Update 2D poses
            joints_right_2d = self.joints_right
            colors_2d = np.full(keypoints.shape[1], 'black')
            colors_2d[joints_right_2d] = 'red'

            def is_good_points(p0,p1):
                return p0[-1]>0.05 and p1[-1]>0.05

            if not initialized:
                
                pos = pos_3d[i]
                for j,pair in enumerate(JOINTS_PAIR):
                        
                    col = 'red' if pair[0]%2==1 else 'black'
                    p0 = pos[pair[0]]
                    p1 = pos[pair[1]]
                    if not is_good_points(p0,p1):
                        p0 = np.zeros_like(p0)
                        p1 = np.zeros_like(p1)

                    for n, ax in enumerate(ax_3d):
                        lines_3d[n].append(ax.plot([p0[0], p1[0]],
                                                   [p0[1], p1[1]],
                                                   [p0[2], p1[2]], zdir='z', c=col))
                                        
    
                initialized = True
            else:
                for j,pair in enumerate(JOINTS_PAIR):
                    pos = pos_3d[i]
                    p0 = pos[pair[0]]
                    p1 = pos[pair[1]]
                    if not is_good_points(p0,p1):
                        p0 = np.zeros_like(p0)
                        p1 = np.zeros_like(p1)
                    for n, ax in enumerate(ax_3d):
                        lines_3d[n][j][0].set_xdata(np.array([p0[0], p1[0]]))
                        lines_3d[n][j][0].set_ydata(np.array([p0[1], p1[1]]))
                        lines_3d[n][j][0].set_3d_properties([p0[2], p1[2]], zdir='z')
    
            
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
            img = np.concatenate([img[...,::-1],image_from_plot],axis=1)
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

def get_kps(data_dir,kpd):
    files = glob.glob(osp.join(data_dir,"*.jpg"))
    nr = len(files)
    re_imgs = []
    re_kps = []
    for i in range(nr):
        file = osp.join(data_dir,f"img_{i+1:05d}.jpg")
        img = wmli.imread(file)
        kps = kpd(img)[0][0] #process one person
        re_imgs.append(img)
        re_kps.append(kps)
    return re_imgs,re_kps

def get_cam_intrinsic_matrix(cam):
    K = np.eye(3,dtype=np.float32)
    K[0,0] = cam['focal_length'][0]
    K[1,1] = cam['focal_length'][1]
    K[0,2] = cam['center'][0]
    K[1,2] = cam['center'][1]
    return K

class Triangulation:
    def __init__(self,cam0,cam1):
        self.cam0 = cam0
        self.cam1 = cam1
        self.R = None
        self.t = None

    def __call__(self,kps0,kps1):
        B,NK,C = kps0.shape
        n_kps0 = np.reshape(kps0,[-1,C])
        n_kps1 = np.reshape(kps1,[-1,C])
        R,t,idxs,points0,points1 = self.get_RT(n_kps0,n_kps1)
        tmp_3d = self.triangulation(points0,points1,R,t)
        data = np.zeros([B*NK,4],dtype=np.float)
        data[idxs,:3] = tmp_3d
        data[idxs,3] = 1
        pts_3d = np.reshape(data,[B,NK,4])
        return np.array(pts_3d)

    def get_RT(self,kp0,kp1):
        points0 = []
        points1 = []
        idxs = []
        for i in range(kp0.shape[0]):
            if kp0[i,2]>0.1 and kp1[i,2]>0.1:
                points0.append(kp0[i])
                points1.append(kp1[i])
                idxs.append(i)
        
        K0 = get_cam_intrinsic_matrix(self.cam0)
        K1 = get_cam_intrinsic_matrix(self.cam1)

        points0 = np.array(points0,dtype=np.float32)[:,:2]
        points1 = np.array(points1,dtype=np.float32)[:,:2]
        org_points1 = np.array(points1,dtype=np.float32)

        points1 = camt.trans_cam_intrinsic(points1,K1,K0)

        E, mask = cv2.findEssentialMat(points0, points1, 
                    cameraMatrix=K0,
                    method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        pts, R, t, mask = cv2.recoverPose(E, points0, points1,cameraMatrix=K0)

        return R,t,idxs,points0,org_points1
    
    def triangulation(self,points0,points1,R,t):
        T1 = np.eye(3,4,dtype=np.float32)
        T2 = np.concatenate([R,t],axis=-1).astype(np.float32)
        K0 = get_cam_intrinsic_matrix(self.cam0)
        K1 = get_cam_intrinsic_matrix(self.cam1)
    
        cam_points0 = camt.pixel2cam(points0,K0)
        cam_points1 = camt.pixel2cam(points1,K1)
    
        pts_4d = cv2.triangulatePoints(T1,T2,cam_points0.T,cam_points1.T)
        pts_4d = np.transpose(pts_4d)
        pts_3d = camt.get_inhomogeneous_coordinates(pts_4d)
        print(cam_points0.shape,cam_points1.shape)
        self.show_diff(points1,pts_3d,self.cam1,R,t)

        return pts_3d

    def show_diff(self,pts_2d,pts_3d,cam,R=None,t=None):
        K = get_cam_intrinsic_matrix(cam)
        if R is not None and t is not None:
            pts_3d = np.transpose(np.matmul(R,pts_3d.T)+t)
    
        pt1_cam_2d = camt.get_inhomogeneous_coordinates(pts_3d)
        pt1_cam = camt.pixel2cam(pts_2d,K)
    
        print("from cam")
        print(pt1_cam)
        print("project")
        print(pt1_cam_2d)

if __name__ == "__main__":
    args = parse_args()
    video_path = args.video
    frames_path = ["/home/wj/ai/mldata1/camera_fusion/rawframes/Posing.55011271",
        "/home/wj/ai/mldata1/camera_fusion/rawframes/Posing.60457274"]

    kpd = KPDetection()

    save_dir = "/home/wj/ai/mldata/pose3d/tmp/predict_on_video_"+wmlu.base_name(frames_path[0])
    cache_dir = "/home/wj/ai/mldata/pose3d/tmp/cache"

    if 'tmp' in save_dir:
        wmlu.create_empty_dir(save_dir,True,True)
    else:
        wmlu.create_empty_dir(save_dir,False)
    wmlu.create_empty_dir(cache_dir,False)

    cache_path = osp.join(cache_dir,osp.basename(frames_path[0]))

    if osp.exists(cache_path):
        with open(cache_path,"rb") as f:
            all_data = pickle.load(f)
        imgs0,kps0,imgs1,kps1 = all_data
    else:
        imgs0,kps0 = get_kps(frames_path[0],kpd)
        imgs1,kps1 = get_kps(frames_path[1],kpd)
        with open(cache_path,"wb") as f:
            pickle.dump([imgs0,kps0,imgs1,kps1],f)
        
    nr = min(len(imgs0),len(imgs1))
    kps0 = np.array(kps0[:nr])
    kps1 = np.array(kps1[:nr])

    cam0 = all_cameras[0]
    cam1 = all_cameras[1]
    triangular = Triangulation(cam0,cam1)

    save_path = osp.join(save_dir,f"output.mp4")
    pos_3d = triangular(kps0,kps1)

    render_ani = RenderAnimation(imgs0,pos_3d,kps0)
    render_ani.update_camera(imgs0[0])
    render_ani.render_animation(save_path=save_path)
    print(f"Save path {save_dir}")
