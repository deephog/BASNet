from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
from shutil import copyfile


data_fold = 'train_data/MSRA/'
src = data_fold+'image_src/'

mtcnn = MTCNN(image_size=160)
path, dirs, files = next(os.walk(src))

for f in files:
    img = Image.open(src + f)
    detected, prob = mtcnn.detect(img)
    print(f, 'prob:', prob)
    if prob[0]:
        if max(prob) > 0.995:
            copyfile(src + f, data_fold + 'human_img/' + f)
            copyfile(data_fold + 'image_src/' + f.split('.')[0] + '.png', data_fold + 'human_mask/' + f.split('.')[0] + '.png')