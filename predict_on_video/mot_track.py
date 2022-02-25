# coding=utf-8
import object_detection2.config.config as config
from object_detection2.engine.train_loop import SimpleTrainer
from object_detection2.engine.defaults import default_argument_parser, get_config_file
from object_detection2.data.dataloader import *
from object_detection2.data.datasets.build import DATASETS_REGISTRY
from object_detection_tools.predictmodel import PredictModel
from object_detection2.mot_toolkit.fair_mot_tracker.multitracker import JDETracker
from object_detection2.mot_toolkit.fair_mot_tracker.multitracker_cpp import CPPTracker
from object_detection2.standard_names import *
from object_detection2.mot_toolkit.build import build_mot
import object_detection2.bboxes as odb
import tensorflow as tf
from keypoints.get_keypoints import KPDetection
import os
import os.path as osp
import wml_utils as wmlu
import img_utils as wmli
import cv2
from track_keypointsv2 import *
os.environ['CUDA_VISIBLE_DEVICES'] = "3"


class DetectionModel:
    def __init__(self) -> None:
        self.model = KPDetection()

    def __call__(self,imgs):
        shape = imgs.shape[1:3]
        _,bboxes,embds = self.model(imgs[0],return_ext_info=True)
        bs = bboxes.shape[0]
        bboxes = odb.npchangexyorder(bboxes)
        bboxes = odb.absolutely_boxes_to_relative_boxes(bboxes,width=shape[1],height=shape[0])
        return {RD_BOXES:bboxes,RD_PROBABILITY:np.ones([bs],dtype=np.float32),RD_ID:embds}

def main(_):
    model = DetectionModel()
    tracker = JDETracker(model)
    #tracker = CPPTracker(model)

    path = '/home/wj/ai/mldata/MOT/MOT20/test/MOT20-04'
    path = '/home/wj/ai/mldata/MOT/MOT20/test_1img'
    path = '/home/wj/ai/mldata/MOT/MOT15/test2/TUD-Crossing/img1'
    path = '/home/wj/ai/mldata/pose3d/basketball2.mp4'
    #path = '/home/wj/ai/mldata/pose3d/tennis1.mp4'
    #files = wmlu.recurse_get_filepath_in_dir(args.test_data_dir,suffix=".jpg")
    files = wmlu.recurse_get_filepath_in_dir(path,suffix=".jpg")
    #save_path = args.save_data_dir
    save_dir = '/home/wj/ai/mldata/MOT/output6'
    save_path = osp.join(save_dir,osp.basename(path))
    wmlu.create_empty_dir(save_dir,remove_if_exists=False,yes_to_all=True)
    writer = wmli.VideoWriter(save_path)
    reader = wmli.VideoReader(path)

    for img in reader:
        img = wmli.resize_width(img,960)
        objs = tracker.update(img)
        img = tracker.draw_tracks(img,objs)
        writer.write(img)
    writer.release()

if __name__ == "__main__":
    tf.app.run()
