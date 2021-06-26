import os
import random
import shutil

src_root = '/home/hypevr/data/projects/data/human2/'
path, dirs, files = next(os.walk(src_root + 'image/'))

num_img = len(files)
random.shuffle(files)

train_pct = 0.9

train_list = files[:int(num_img*train_pct)]
val_list = files[int(num_img*train_pct):]

for f in train_list:
    fname = f.split('.')[0]
    shutil.copy(src_root + 'image/' + f, src_root + 'train/image/' + f)
    shutil.copy(src_root + 'mask/' + fname + '.jpg', src_root + 'train/mask/' + fname + '.jpg')

for f in val_list:
    fname = f.split('.')[0]
    shutil.copy(src_root + 'image/' + f, src_root + 'val/image/' + f)
    shutil.copy(src_root + 'mask/' + fname + '.jpg', src_root + 'val/mask/' + fname + '.jpg')
