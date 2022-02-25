import os
from posixpath import basename
import sys
import os.path as osp
import wml_utils as wmlu
import img_utils as wmli
import pickle
import copy
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

tf.enable_eager_execution()


cur_path = osp.dirname(__file__)
pdir_path = osp.dirname(cur_path)
sys.path.append(osp.join(pdir_path,"predict_on_video"))

from keypoints.get_keypoints import *
from demo_toolkit import *
from track_keypoints import TrackKeypoints

def predict_on_video(video_path):
    print(f"Process {video_path}")
    dir_name = osp.dirname(video_path)
    base_name = wmlu.base_name(video_path)

    save_path = osp.join(dir_name,base_name+f"_*.npz")
    tfiles = glob.glob(save_path)
    '''if len(tfiles)>0:
        print(f"File {tfiles} exists.")
        print(f"Skip {video_path}")
        return'''

    max_trackid_nr = 1
    min_frames_nr = 300
    track_model = TrackKeypoints(video_path)
    frames = track_model.track_keypoints(return_frames=False,max_frames_nr=None)
    sorted_kps = track_model.sorted_keypoints
        
    for tid,keypoints in sorted_kps[:max_trackid_nr]:
        idxs = np.array(list(keypoints.keys()))
        if len(idxs)<min_frames_nr:
            continue
        min_idx = np.min(idxs)
        max_idx = np.max(idxs)
        save_path = osp.join(dir_name,base_name+f"_{tid}.npz")

        all_keypoints = []
        all_frames = []
        for i in range(min_idx,max_idx):
            if i not in keypoints:
                continue
            if frames is not None:
                cur_frame = copy.deepcopy(frames[i])
                show_keypoints(cur_frame,keypoints[i])
                all_frames.append(cur_frame)
            all_keypoints.append(keypoints[i])
        if len(all_frames)>0:
            wmli.videowrite(wmlu.change_suffix(save_path,"avi"),all_frames)
        with open(save_path,"wb") as f:
            pickle.dump(np.array(all_keypoints),f)


if __name__ == "__main__":
    suffix = ".mp4"
    data_path = "/home/wj/ai/mldata/totalcapture/"
    all_files = wmlu.recurse_get_filepath_in_dir(data_path,suffix)
    #all_files = all_files[:1]
    for file in all_files:
        try:
            predict_on_video(file)
            sys.stdout.flush()
        except Exception as e:
            print(f"Process {file} faild.")

