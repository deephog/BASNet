import os
from skimage import io, transform, color
from skimage.viewer import ImageViewer
import numpy as np

src = 'train_data/HUMAN/train/mask/'
path, dirs, files = next(os.walk(src))
images = []
for f in files:
    image = io.imread(path+f, as_gray=True)
    image = np.asarray(image)
    viewer = ImageViewer(image)
    viewer.show()
    print(np.unique(image))

    image[image<1.] = 0
    viewer = ImageViewer(image)
    viewer.show()
    input('wait')
    image = transform.resize(image, (256, 256), mode='constant', order=0, preserve_range=True)
    images.append(image)

images = np.asarray(images)
print(np.unique(images))

