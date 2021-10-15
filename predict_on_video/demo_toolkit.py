import cv2
import numpy as np
import os
import sys
from vis_utils import *

__version__ = "1.1.2"

joints_pair = [[0 , 1], [1 , 2], [2 , 0], [1 , 3], [2 , 4], [3 , 5], [4 , 6], [5 , 6], [5 , 11],
[6 , 12], [11 , 12], [5 , 7], [7 , 9], [6 , 8], [8 , 10], [11 , 13], [13 , 15], [12 , 14], [14 , 16]]
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

def add_jointsv1(image, joints, color, r=5):

    def link(a, b, color):
        jointa = joints[a]
        jointb = joints[b]
        cv2.line(
                image,
                (int(jointa[0]), int(jointa[1])),
                (int(jointb[0]), int(jointb[1])),
                color, 2 )

    # add link
    for pair in joints_pair:
        link(pair[0], pair[1], color)

    # add joints
    for i, joint in enumerate(joints):
        cv2.circle(image, (int(joint[0]), int(joint[1])), r, colors_tableau[i], -1)

    return image

def add_jointsv2(image, joints, color, r=5):

    def link(a, b, color):
        jointa = joints[a]
        jointb = joints[b]
        if jointa[2] > 0.01 and jointb[2] > 0.01:
            cv2.line(
                image,
                (int(jointa[0]), int(jointa[1])),
                (int(jointb[0]), int(jointb[1])),
                color, 2 )

    # add link
    for pair in joints_pair:
        link(pair[0], pair[1], color)

    # add joints
    for i, joint in enumerate(joints):
        if joint[2] > 0.05 and joint[0] > 1 and joint[1] > 1:
            cv2.circle(image, (int(joint[0]), int(joint[1])), r, colors_tableau[i], -1)

    return image

def show_keypoints(image, joints, color=[0,255,0]):
    image = np.ascontiguousarray(image)
    joints = np.array(joints)
    if color is None:
        use_random_color=True
    else:
        use_random_color = False
    if len(joints.shape)==2:
        joints = [joints]
    else:
        assert len(joints.shape)==3,"keypoints need to be 3-dimensional."

    for person in joints:
        if use_random_color:
            color = np.random.randint(0, 255, size=3)
            color = [int(i) for i in color]
        if person.shape[-1] == 3:
            add_jointsv2(image, person, color=color)
        else:
            add_jointsv1(image, person, color=color)

    return image

def show_keypoints_diff(image, joints0, joints1,color=[0,255,0]):
    image = np.ascontiguousarray(image)
    joints0 = np.array(joints0)
    joints1 = np.array(joints1)
    if color is None:
        use_random_color=True
    else:
        use_random_color = False
    if len(joints0.shape)==2:
        points_nr = joints0.shape[0]
        joints0 = [joints0]
        joints1 = [joints1]
    else:
        points_nr = joints0.shape[1]
        assert len(joints0.shape)==3,"keypoints need to be 3-dimensional."

    for person0,person1 in zip(joints0,joints1):
        if use_random_color:
            color = np.random.randint(0, 255, size=3)
            color = [int(i) for i in color]
        for i in range(points_nr):
            jointa = person0[i]
            jointb = person1[i]
            if person0.shape[-1] == 3:
                if person0[i][-1]>0.015 and person1[i][-1]>0.015:
                    cv2.line(
                    image,
                    (int(jointa[0]), int(jointa[1])),
                    (int(jointb[0]), int(jointb[1])),
                     color, 2 )
            else:
                cv2.line(
                    image,
                    (int(jointa[0]), int(jointa[1])),
                    (int(jointb[0]), int(jointb[1])),
                     color, 2 )

    return image

class BufferTextPainter:
    def __init__(self,buffer_len=30):
        self.buffer_nr = buffer_len
        self.cur_text = ""
        self.cur_idx = 0
        self.font = os.path.join(os.path.dirname(__file__),"simhei.ttf")

    def putText(self,img,text,font_scale=20,text_color=(255,255,255)):
        img = np.ascontiguousarray(img)
        if text == "":
            if self.cur_idx == 0:
                return img
        else:
            self.cur_idx =self.buffer_nr
            self.cur_text = text

        self.cur_idx = self.cur_idx-1
        img = draw_text(img,(12,12),self.cur_text,
                        text_color=text_color,
                        font_size=font_scale,
                        font=self.font)
        return img

def resize_height(img,h,interpolation=cv2.INTER_LINEAR):
    shape = img.shape
    new_h = h
    new_w = int(shape[1]*new_h/shape[0])
    return cv2.resize(img,dsize=(new_w,new_h),interpolation=interpolation)

def resize_width(img,w,interpolation=cv2.INTER_LINEAR):
    shape = img.shape
    new_w = w
    new_h = int(shape[0]*new_w/shape[1])
    return cv2.resize(img,dsize=(new_w,new_h),interpolation=interpolation)

def resize_short_size(img,size,interpolation=cv2.INTER_LINEAR):
    shape = img.shape
    if shape[0]<shape[1]:
        return resize_height(img,size,interpolation)
    else:
        return resize_width(img,size,interpolation)

class VideoDemo:
    def __init__(self,model,fps=30,save_path=None,buffer_size=0,show_video=True,max_frame_cn=None,interval=None) -> None:
        self.model = model
        self.fps = fps
        self.save_path = save_path
        self.buffer_size = buffer_size
        self.buffer = []
        self.write_size = None
        self.video_reader = None
        self.video_writer = None
        self.show_video = show_video
        self.preprocess = None
        self.max_frame_cn = max_frame_cn
        self.interval = interval
        print(f"Demo toolkit version {__version__}.")
    
    def __del__(self):
        self.close()
    
    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()


    def init_writer(self):
        save_path = self.save_path
        if save_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(save_path,fourcc,self.fps,self.write_size)
    
    def init_reader(self):
        if self.video_path is not None and os.path.exists(self.video_path):
            print(f"Use video file {self.video_path}")
            self.video_reader = cv2.VideoCapture(self.video_path)
            self.frame_cnt = int(self.video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.max_frame_cn is not None and self.max_frame_cn>1:
                self.frame_cnt = min(self.frame_cnt,self.max_frame_cn)
        else:
            if self.video_path is not None:
                vc = int(self.video_path)
            else:
                vc = 0
            print(f"Use camera {vc}")
            self.video_reader = cv2.VideoCapture(vc)
            self.frame_cnt = -1

        width = int(self.video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.preprocess is None:
            self.write_size = (width,height)
        else:
            tmp_img = np.zeros([height,width,3],dtype=np.uint8)
            tmp_img = self.preprocess(tmp_img)
            if isinstance(tmp_img,dict):
                tmp_img = self.get_last_img([tmp_img])
            self.write_size = (tmp_img.shape[1],tmp_img.shape[0])
        self.fps = self.video_reader.get(cv2.CAP_PROP_FPS)

    def inference_loop(self,video_path=None):
        self.video_path = video_path
        self.init_reader()
        self.init_writer()
        idx = 0

        print(f"Press Esc to escape.")

        while True:
            idx += 1
            if self.interval is not None and self.interval>1:
                if idx%self.interval != 0:
                    continue
            ret,frame = self.video_reader.read()
            if not ret:
                break
            frame = frame[...,::-1]
            if self.preprocess is not None:
                frame = self.preprocess(frame)
            img = self.inference(frame)
            if self.show_video:
                cv2.imshow("video",img[...,::-1])
                if cv2.waitKey(30)&0xFF == 27:
                    break
            if self.frame_cnt > 1:
                sys.stdout.write(f"{idx}/{self.frame_cnt}  {idx*100/self.frame_cnt:.3f}%.\r")
                if idx>self.frame_cnt:
                    break

    def inference(self,img):
        if self.buffer_size <= 1:
            r_img = self.inference_single_img(img)
        else:
            r_img = self.inference_buffer_img(img)
        if self.video_writer is not None:
            self.video_writer.write(r_img[...,::-1])
        return r_img

    def inference_single_img(self,img):
        return self.model(img)

    def inference_buffer_img(self,img):
        self.buffer.append(img)
        if len(self.buffer)>self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
        return self.model(self.buffer)

    @staticmethod
    def get_last_img(imgs):
        img = imgs[-1]
        if isinstance(img,dict):
            if 'raw_image' in img:
                return img['raw_image']
            return img['image']
        else:
            return img

    @staticmethod
    def resize_h_and_save_raw_image_preprocess(img,h=224):
        r_img = resize_height(img,h).astype(np.uint8)
        return {'image':r_img,"raw_image":img}

class IntervalMode:
    def __init__(self,interval=30):
        self.interval = interval
        self.idx = 0

    def add(self):
        self.idx += 1

    def need_pred(self):
        self.add()
        return (self.idx%self.interval)==0
