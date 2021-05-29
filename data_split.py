import os
import random
import shutil

src_root = 'HUMAN/'
path, dirs, files = next(os.walk(src_root + 'image/'))

num_img = len(files)
random.shuffle(files)

train_pct = 0.8

train_list = files[:int(num_img*train_pct)]
val_list = files[int(num_img*train_pct):]

for f in train_list:
    fname = f.split('.')[0]
    shutil.copy(src_root + 'image/' + f, src_root + 'train/image/' + f)
    shutil.copy(src_root + 'mask/' + fname + '.png', src_root + 'train/mask/' + fname + '.png')

for f in val_list:
    fname = f.split('.')[0]
    shutil.copy(src_root + 'image/' + f, src_root + 'val/image/' + f)
    shutil.copy(src_root + 'mask/' + fname + '.png', src_root + 'val/mask/' + fname + '.png')
