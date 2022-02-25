from track_keypoints import *
from demo_toolkit import *
import tensorflow as tf
import numpy as np
import os.path as osp
import wml_utils as wmlu
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "3"


tf.enable_eager_execution()

def init_writer(save_path,write_size):
    print(f"Save path {save_path}")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(save_path, fourcc, 30,write_size)
    return video_writer

if __name__ == "__main__":
    video_path = "/home/wj/ai/mldata/0day/a/TC_S2_acting2_cam8.mp4"
    save_dir = "/home/wj/ai/mldata/0day/b"
    wmlu.create_empty_dir(save_dir,False)
    track_model = TrackKeypoints(video_path)
    frames = track_model.track_keypoints(return_frames=True,max_frames_nr=1000)

    for tid,keypoints in track_model.keypoints.items():
        idxs = np.array(list(keypoints.keys()))
        min_idx = np.min(idxs)
        max_idx = np.max(idxs)
        save_path = osp.join(save_dir,f"{min_idx}_{max_idx}_{tid}.mp4")
        writer = init_writer(save_path,track_model.write_size)
        for i in range(min_idx,max_idx):
            cur_frame = frames[i]
            if i in keypoints:
                cur_frame = show_keypoints(cur_frame,[keypoints[i]])
            writer.write(cur_frame)
        writer.release()




