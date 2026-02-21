
import os 
import random


file_path = 'MEAD_preprocessed/new_formats'
f_o = open('train.txt', "w")
f_test = open('val.txt',"w")

file_list = []
file_name = []
# for video_name in os.listdir(file_path):
#     video_path = os.path.join(file_path,video_name)
#     for file in os.listdir(video_path):
#         if 'wav' in file:
#             continue
#         write_line = str(video_name)+'/'+str(file)
#         file_list.append(write_line)

for video_name in os.listdir(file_path):
    file_name.append(video_name.split('_')[0])
    file_list.append(video_name)
num = 6
test_name = random.sample(file_name,num)
print(test_name)
test_list = []
for file in file_list:
    name = file.split('_')[0]
    if name in test_name:
        test_list.append(file)
train_list = list(set(file_list) ^ set(test_list))
for video_name in test_list:
    video_path = os.path.join(file_path,video_name)
    for file in os.listdir(video_path):
        if 'wav' in file:
            continue
        write_line = str(video_name)+'/'+str(file)+'\n'
        f_test.write(write_line)
f_test.close()
for video_name in train_list:
    video_path = os.path.join(file_path,video_name)
    for file in os.listdir(video_path):
        if 'wav' in file:
            continue
        write_line = str(video_name)+'/'+str(file)+'\n'
        f_o.write(write_line)
f_o.close()


# for name in train_list:
#     write_line = '25_fps/'+str(name)+'\n'
#     f_train.write(write_line)
# f_train.close()

# import os
# import glob
# import cv2
# import shutil
# file_path = '/data/wav2lip/Wav2Lip/MEAD_preprocessed/FFOutput/'
# crop_face_video_dir = '/data/wav2lip/Wav2Lip/MEAD_preprocessed/new_formats/'
# for video_name in os.listdir(file_path):
#     frame_dir = os.path.join(file_path,video_name)
#     temp_crop_face_video_dir = os.path.join(crop_face_video_dir,video_name)
#     if not os.path.exists(temp_crop_face_video_dir):
#         os.makedirs(temp_crop_face_video_dir)
#     frame_length = len(glob.glob(os.path.join(frame_dir, '*.jpg')))
#     end_frame_index = list(range(35, frame_length, 35))
#     video_clip_num = len(end_frame_index)
#     for i in range(video_clip_num):
#         res_face_clip_dir = os.path.join(temp_crop_face_video_dir, str(i).zfill(6))

#         if not os.path.exists(res_face_clip_dir):
#             os.mkdir(res_face_clip_dir)
#         for frame_index in range(end_frame_index[i]- 35,end_frame_index[i]):
#             source_frame_path = os.path.join(frame_dir,str(frame_index)+'.jpg')

#             res_crop_face_frame_path = os.path.join(res_face_clip_dir, str(frame_index) + '.jpg')
#             if os.path.exists(res_crop_face_frame_path):
#                 os.remove(res_crop_face_frame_path)
#             shutil.copy(source_frame_path,res_crop_face_frame_path)


# file_path = '/data/wav2lip/Wav2Lip/MEAD_preprocessed/FFOutput/'
# crop_face_video_dir = '/data/wav2lip/Wav2Lip/MEAD_preprocessed/new_formats/'
# for video_name in os.listdir(file_path):
#     wav_path = os.path.join(file_path,video_name,"audio.wav")
#     new_wav_path = os.path.join(crop_face_video_dir,video_name,'audio.wav')
#     shutil.copy(wav_path,new_wav_path)


# import shutil
# import os
# from glob import glob
# source_dir = '/data/wav2lip/LRS2/LRW/lipread_mp4/'
# target_dir = '/data/wav2lip/Wav2Lip/LRW_preprocessed/val'

# file_names = os.listdir(source_dir)

# for file in file_names:
#     video_file =os.path.join(source_dir,file,'val')
#     filelist = glob(os.path.join(video_file, '*.mp4'))
#     for video in filelist:
#         shutil.copy(video,os.path.join(target_dir,os.path.basename(video)))