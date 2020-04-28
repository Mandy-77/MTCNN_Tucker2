# import os, sys
import numpy as np

# file1path = 'list_bbox_celeba.txt'
# file2path = 'list_landmarks_align_celeba.txt'

# file_1 = open(file1path,'r')
# file_2 = open(file2path,'r')

# list1 = []
# for line in file_1.readlines():
#     ss = line.strip()
#     list1.append(ss)
# file_1.close()

# list2 = []
# for line in file_2.readlines():
#     ss = line.strip()
#     if ss[7:10] == 'jpg':
#         ss = ss[11:]
#     list2.append(ss)
# file_2.close()

# file_new = open('result.txt','w')
# for i in range(len(list1)):
#     sline = list1[i] + ' ' + list2[i]
#     file_new.write(sline+'\n')
# file_new.close()

file_new = open('trainImageList.txt','r')
annotations = file_new.readlines()

num = len(annotations)
print("%d total images" % num)

l_idx =0
idx = 0
# image_path bbox landmark(5*2)

for annotation in annotations:
    # print imgPath

    annotation = annotation.strip().split()
    gt_box = map(float, annotation[1:5])
        # gt_box = [gt_box[0], gt_box[2], gt_box[1], gt_box[3]]


    gt_box = np.array(gt_box, dtype=np.int32)

    landmark = map(float, annotation[5:])
    landmark = np.array(landmark, dtype=np.float)

    x1, y1, x2, y2 = gt_box

