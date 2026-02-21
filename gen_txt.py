
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
