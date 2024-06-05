import os
import cv2
import json
import torch
import random
import logging
import tempfile
import numpy as np
from copy import copy
from PIL import Image
from torch.utils.data import Dataset
from utils.registry_class import DATASETS


@DATASETS.register_class()
class VideoDataset(Dataset):
    def __init__(self, 
            data_list,
            data_dir_list,
            origin_data_dir,
            max_words=1000,
            resolution=(384, 256),
            vit_resolution=(224, 224),
            max_frames=8,
            sample_fps=8,
            transforms=None,
            vit_transforms=None,
            get_first_frame=False,
            **kwargs):

        self.max_words = max_words
        self.max_frames = max_frames
        self.resolution = resolution
        self.vit_resolution = vit_resolution
        self.sample_fps = sample_fps
        self.transforms = transforms
        self.vit_transforms = vit_transforms
        self.get_first_frame = get_first_frame
        self.origin_data_dir = origin_data_dir
        
        image_list = []
        for item_path, data_dir in zip(data_list, data_dir_list):
            lines = open(item_path, 'r').readlines()
            lines = [[data_dir, item] for item in lines]
            image_list.extend(lines)
        self.image_list = image_list


    def __getitem__(self, index):
        data_dir, file_path = self.image_list[index]
        video_key = file_path.split('|||')[0]
        #try:
        ref_frame, vit_frame, video_data, caption, origin_video_data = self._get_video_data(data_dir, file_path)
        # except Exception as e:
        #     logging.info('{} get frames failed... with error: {}'.format(video_key, e))
        #     caption = ''
        #     video_key = '' 
        #     ref_frame = torch.zeros(3, self.resolution[1], self.resolution[0])
        #     vit_frame = torch.zeros(3, self.vit_resolution[1], self.vit_resolution[0])
        #     video_data = torch.zeros(self.max_frames, 3, self.resolution[1], self.resolution[0]) 
        #     origin_video_data = torch.zeros(self.max_frames, 3, self.resolution[1], self.resolution[0])       
        return ref_frame, vit_frame, video_data, caption, video_key, origin_video_data
    
    
    def _get_video_data(self, data_dir, file_path):
        video_key, caption = file_path.split('|||')
        file_path = os.path.join(data_dir, video_key)
        #print(file_path)
        origin_video_key = video_key[:-6] + '_0.mp4'
        #print(data_dir, self.origin_data_dir)
        origin_file_path = os.path.join(self.origin_data_dir, origin_video_key)

        for _ in range(5):
            try:
                capture = cv2.VideoCapture(file_path)
                capture_ori = cv2.VideoCapture(origin_file_path)
                _fps = capture.get(cv2.CAP_PROP_FPS)
                _total_frame_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                stride = round(_fps / self.sample_fps)
                stride = 1
                #print(_fps, self.sample_fps, stride, _total_frame_num)
                cover_frame_num = (stride * self.max_frames)
                if _total_frame_num < cover_frame_num + 5:
                    start_frame = 0
                    end_frame = _total_frame_num
                else:
                    start_frame = random.randint(0, _total_frame_num-cover_frame_num-5)
                    end_frame = start_frame + cover_frame_num
                #print(start_frame, end_frame)
                start_frame, end_frame = 0, 16
                pointer, frame_list,origin_frame_list = 0, [], []
                while(True):
                    ret, frame = capture.read()
                    ret1, frame_ori = capture_ori.read()
                    #print(ret, ret1)
                    if (not ret) or (frame is None): break
                    if pointer < start_frame: continue
                    if pointer >= end_frame: break
                    #print(pointer-start_frame)
                    if (pointer - start_frame) % stride == 0:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = Image.fromarray(frame)
                        frame_list.append(frame)
                        frame_ori = cv2.cvtColor(frame_ori, cv2.COLOR_BGR2RGB)
                        frame_ori = Image.fromarray(frame_ori)
                        origin_frame_list.append(frame_ori)
                    pointer +=1 
                    
                break
            except Exception as e:
                logging.info('{} read video frame failed with error: {}'.format(video_key, e))
                continue

        video_data = torch.zeros(self.max_frames, 3,  self.resolution[1], self.resolution[0])
        origin_video_data = torch.zeros(self.max_frames, 3,  self.resolution[1], self.resolution[0])
        #print(len(frame_list))
        if self.get_first_frame:
            ref_idx = 0
        else:
            ref_idx = int(len(frame_list)/2)
        try:
            if len(frame_list)>0:
                #print(len(frame_list), self.max_frames)
                mid_frame = copy(frame_list[ref_idx])
                vit_frame = self.vit_transforms(mid_frame)
                frames = self.transforms(frame_list)
                video_data[:len(frame_list), ...] = frames
                ori_frames = self.transforms(origin_frame_list)
                origin_video_data[:len(ori_frames), ...] = ori_frames
            else:
                vit_frame = torch.zeros(3, self.vit_resolution[1], self.vit_resolution[0])
        except:
            vit_frame = torch.zeros(3, self.vit_resolution[1], self.vit_resolution[0])
        ref_frame = copy(frames[ref_idx])
        
        return ref_frame, vit_frame, video_data, caption, origin_video_data

    def __len__(self):
        return len(self.image_list)



if __name__ =='__main__':
    v_dataset = VideoDataset(['/mnt/data/zhenghao/i2vgen-xl/data/eval_list.txt', ],['/mnt/data/zhenghao/i2vgen-xl/data/eval/edit_video', ],'/mnt/data/zhenghao/i2vgen-xl/data/eval/ori_video')
    print(len(v_dataset.__getitem__(1)))
    