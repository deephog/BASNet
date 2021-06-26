import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
from shutil import copyfile


data_fold = '/home/hypevr/Desktop/data/projects/data/DUTS/DUTS-TE/'
src = data_fold+'image/'
img_ext = '.jpg'

mask_src = 'mask/'
mask_ext = '.png'

dst_image = 'nonhuman_img/'
dst_mask = 'nonhuman_mask/'

if not os.path.isdir(data_fold + dst_mask):
    os.mkdir(data_fold + dst_mask)
if not os.path.isdir(data_fold + dst_image):
    os.mkdir(data_fold + dst_image)

mtcnn = MTCNN(image_size=300)
path, dirs, files = next(os.walk(src))

for f in files:
    if f.split('.')[-1] == img_ext:
        img = Image.open(src + f)
        array_img = np.asarray(img)
        if len(array_img.shape) > 2:
            detected, prob = mtcnn.detect(img)
            print(f, 'prob:', prob)
            if not prob[0]:
                #if max(prob) < 0.5:
                copyfile(src + f, data_fold + dst_image + f)
                copyfile(data_fold + mask_src + f.split('.')[0] + mask_ext, data_fold + dst_mask + f.split('.')[0] + mask_ext)
            #else:

