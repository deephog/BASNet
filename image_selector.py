import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
from torchvision import models
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
import time

from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import BASNet
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(image_name,pred,d_dir):

	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()
	
	io.imsave('inter_img.png', predict_np*255)

	im = Image.fromarray(predict_np*255).convert('RGB')
	img_name = image_name.split("/")[-1]
	image = io.imread(image_name)
	imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

	pb_np = np.array(imo)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	imo.save(d_dir+imidx+'.png')


if __name__ == '__main__':
	# --------- 1. get image path and name ---------
	
	image_dir = './test_data/test_images/'
	prediction_dir = './test_data/test_results/'
	#model_dir = './saved_models/basnet_bsi/basnet_time.pth'
	
	img_name_list = glob.glob(image_dir + '*.jpg')
	
	# --------- 2. dataloader ---------
	#1. dataload
	test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],transform=transforms.Compose([RescaleT(224),ToTensorLab(flag=0)]))
	test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=4,shuffle=False,num_workers=1)
	
	# --------- 3. model define ---------
	print("...load BASNet...")
	net = models.resnext101_32x8d(pretrained=True)
	if torch.cuda.is_available():
		net.cuda()
	net.eval()
	
	# --------- 4. inference for each image ---------
	for i_test, data_test in enumerate(test_salobj_dataloader):

		inputs_test = data_test['image']
		inputs_test = inputs_test.type(torch.FloatTensor)
	
		if torch.cuda.is_available():
			inputs_test = Variable(inputs_test.cuda())
		else:
			inputs_test = Variable(inputs_test)

		d1 = net(inputs_test)#,d2,d3,d4,d5,d6,d7,d8
		#print(d1)
		torch.cuda.synchronize()
		#d1 = d1.cpu() 
		print(time.time()-start)
		
		#print(d1.shape)
		#traced_script_module = torch.jit.trace(net, inputs_test)
		#traced_script_module.save("traced_model_BASNet.pt")
	
		# normalization
		#d1 = torch.nn.functional.sigmoid(d1)
		#pred = d1[:,0,:,:]
		#pred = normPRED(d1)
	
		# save results to test_results folder
		#save_output(img_name_list[i_test],pred,prediction_dir)
	
		#del d1,d2,d3,d4,d5,d6,d7,d8
