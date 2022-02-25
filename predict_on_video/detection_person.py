from track_keypoints import *
from keypoints.get_keypoints import PersonDetection
import tensorflow as tf
import numpy as np
import os.path as osp
import img_utils as wmli
import wml_utils as wmlu
import os
import object_detection2.visualization as odv

os.environ['CUDA_VISIBLE_DEVICES'] = "3"


tf.enable_eager_execution()

def init_writer(save_path,write_size):
    print(f"Save path {save_path}")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(save_path, fourcc, 30,write_size)
    return video_writer

def color_fn(*args,**kwargs):
    return (255,0,0)

if __name__ == "__main__":
    video_path = osp.expanduser('~/ai/mldata/pose3d/basketball2.mp4')
    save_dir = "/home/wj/ai/mldata/0day/b1"
    wmlu.create_empty_dir(save_dir,False)
    save_path = osp.join(save_dir,osp.basename(video_path))
    model = PersonDetection()
    reader = wmli.VideoReader(video_path)
    writer = wmli.VideoWriter(save_path)
    

    for frame in reader:
        bboxes,probs = model(frame[...,::-1])
        xmin,ymin,xmax,ymax = bboxes[...,0],bboxes[...,1],bboxes[...,2],bboxes[...,3]
        bboxes = np.stack([ymin,xmin,ymax,xmax],axis=-1)
        img = odv.draw_bboxes(frame,classes=np.ones_like(probs,dtype=np.int32),scores=probs,
        bboxes=bboxes,is_relative_coordinate=False,color_fn=color_fn)
        writer.write(img)
    writer.release()

