from keypoints.get_keypoints import *
import cv2
import tensorflow as tf
import tfop
import copy
import math
import object_detection2.bboxes as odb
from itertools import count


class TrackKeypoints:
    def __init__(self,video_path,max_frame_cn=None):
        self.video_path = video_path
        self.max_frame_cn = max_frame_cn
        self.interval = 1
        self.kp_model = KPDetection()
        self.sorted_keypoints = []

    def track_keypoints(self,return_frames=False,max_frames_nr=None):
        self.video_reader = cv2.VideoCapture(self.video_path)
        self.frame_cnt = int(self.video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        self.keypoints = {}
        if self.max_frame_cn is not None and self.max_frame_cn > 1:
            self.frame_cnt = min(self.frame_cnt, self.max_frame_cn)
        width = int(self.video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.write_size = (width, height)
        trackid2size = {}
        frames = []

        idx = -1
        losted_id = {}

        while True:
            idx += 1
            if self.interval is not None and self.interval > 1:
                if idx % self.interval != 0:
                    continue
            if max_frames_nr is not None and max_frames_nr>1 and idx>max_frames_nr:
                break
            ret, frame = self.video_reader.read()
            if not ret:
                break
            if return_frames:
                frames.append(copy.deepcopy(frame))
            frame = frame[..., ::-1]
            frame_size = frame.shape
            bboxes,embds = self.kp_model.get_person_bboxes(frame,return_ext_info=True)
            print("bboxes:",bboxes)
            probs = np.ones([bboxes.shape[0]],dtype=np.float32)
            tracked_id, tracked_bboxes, tracked_idx = tfop.fair_mot(bboxes, probs,embds,
                                                                    is_first_frame=idx==0,
                                                                    det_thredh=0.0,
                                                                    frame_rate=25,
                                                                    track_buffer=10,
                                                                    assignment_thresh=[0.9,0.9,0.7],
                                                                    return_losted=False)
            new_bboxes = []
            tracked_id = tracked_id.numpy()
            tracked_bboxes = tracked_bboxes.numpy()
            tracked_idx = tracked_idx.numpy()

            for i,tid,tbbox,tidx in zip(count(),tracked_id,tracked_bboxes,tracked_idx):
                if tidx>=0:
                    bboxes[tidx] = odb.bbox_of_boxes([bboxes[tidx],tbbox])
                '''else:
                    if tid not in losted_id:
                        losted_id[tid] = 4
                    losted_id[tid] = losted_id[tid]-1
                    if losted_id[tid]>=0:
                        tracked_idx[i] = bboxes.shape[0]+len(new_bboxes)
                        new_bboxes.append(tbbox)'''
            
            if len(new_bboxes)>0:
                bboxes = np.concatenate([bboxes,new_bboxes],axis=0)
                bboxes = odb.npclip_bboxes(bboxes,[frame_size[1],frame_size[0]])
            print(bboxes,tracked_idx)

            ans = self.kp_model.get_kps_by_bboxes(frame,bboxes)

            for tid,tbbox,tidx in zip(tracked_id,tracked_bboxes,tracked_idx):
                if tidx<0:
                    print(f"Tidx is {tidx}.")
                if tid not in self.keypoints:
                    self.keypoints[tid] = {}
                    trackid2size[tid] = 0
                size = math.sqrt((tbbox[3]-tbbox[1])*(tbbox[2]-tbbox[0]))
                if trackid2size[tid]<size:
                    trackid2size[tid] = size

                self.keypoints[tid][idx] = ans[tidx]
        
        trackid2size = list(trackid2size.items())
        trackid2size.sort(key=lambda x:-x[1])

        self.sorted_keypoints = []
        for tid,_ in trackid2size:
            self.sorted_keypoints.append([tid,self.keypoints[tid]])

        self.video_reader.release()

        if return_frames:
            return frames
        else:
            return None