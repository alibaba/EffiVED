from augmenter import ImageToSeqAugmenter
import cv2
from PIL import Image
import numpy as np
import imageio
import glob
import os

augmenter = ImageToSeqAugmenter(perspective=True, affine=True, motion_blur=True,
                                             rotation_range=(-5, 5), perspective_magnitude=0.08,
                                             hue_saturation_range=(-5, 5), brightness_range=(-40, 40),
                                             motion_blur_prob=0.25, motion_blur_kernel_sizes=(9, 11),
                                             translate_range=(-0.1, 0.1))



def get_pair_img(img_url1, img_url2):
    

    img = img_url1
    img1 = img_url2
    im_trafo,im_trafo1 = augmenter(np.asarray(img),np.asarray(img1),None)
    return im_trafo,im_trafo1 

def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)
        
def get_pair_video(img_url1, img_url2, n_frames, save_name):
    video1 = [np.array(img_url1)]
    video2 = [np.array(img_url2)]
    for t in range(n_frames-1):
        img1, img2 = get_pair_img(img_url1, img_url2)
        video1.append(img1)
        video2.append(img2)
    export_to_video(video1, './ori_video/{}_0.mp4'.format(save_name), n_frames//2)
    export_to_video(video2, './edit_video/{}_1.mp4'.format(save_name), n_frames//2)
    imageio.mimwrite('./ori_gif/{}_0.gif'.format(save_name), video1, duration=150, loop=0)
    imageio.mimwrite('./edit_gif/{}_1.gif'.format(save_name), video2, duration=150, loop=0)


if __name__ == "__main__":
    import json 
    from datasets import load_dataset
    cnt = 0
    for path in glob.glob('./generate_videos/magicbrush/*.parquet'):
        dataset = load_dataset("parquet", data_files=path)
        for item in dataset['train']:
            img_name = item['img_id']
            prompt = item['instruction']
            if os.path.exists('./ori_video/{}_0.mp4'.format(img_name)):
                continue
            try:
                get_pair_video(item['source_img'], item['target_img'], 8, img_name)
                with open('./edit_video/{}_1.txt'.format(img_name),'w') as f:
                    f.write(prompt.strip())
            except:
                continue
            cnt+=1
            print(cnt)
  