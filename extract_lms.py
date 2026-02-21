import cv2
import numpy as np
import face_alignment
from skimage import io
from skimage.transform import resize
import torch
import torch.nn.functional as F
import json
import os
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import argparse
from tqdm import tqdm
ori_imgs_dir = '/data/wav2lip/Wav2Lip/MEAD_preprocessed/new_formats/'
save_file = '/data/wav2lip/Wav2Lip/landmark_file'


try:
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
except:
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
for file_path in sorted(os.listdir(ori_imgs_dir))[:2000]:
        if not os.path.exists(os.path.join(save_file,file_path)):
             os.makedirs(os.path.join(save_file,file_path))
        for image_path in tqdm(os.listdir(os.path.join(ori_imgs_dir,file_path))):
            if image_path.endswith('.wav'):
                continue
            if not os.path.exists(os.path.join(save_file,file_path,image_path)):
                 os.makedirs(os.path.join(save_file,file_path,image_path))
            save_path = os.path.join(save_file,file_path,image_path)
            for image in os.listdir(os.path.join(ori_imgs_dir,file_path,image_path)):
                if image.endswith('.jpg'):
                    input = cv2.imread(os.path.join(ori_imgs_dir,file_path,image_path, image), cv2.IMREAD_UNCHANGED) # [H, W, 3]
                    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
                    # input = io.imread(os.path.join(ori_imgs_dir,file_path,image_path, image))[:, :, :3]
                    try:
                        preds = fa.get_landmarks(input)
                        if len(preds) > 0:
                            lands = preds[0].reshape(-1, 2)[:,:2]
                            np.savetxt(os.path.join(save_path, image[:-3] + 'lms'), lands, '%f')
                    except:
                        print(os.path.join(ori_imgs_dir,file_path,image_path,image))                    


