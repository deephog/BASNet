import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
import cv2
# import torch.optim as optim
from data import test_dataset
import numpy as np
import onnx
from PIL import Image
import glob
import time
from data import get_loader

from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab, OtherTrans
from data_loader import SalObjDataset

from model import BASNet
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)
	dn = (d-mi)/(ma-mi)
	return dn

def overlay(image, mask):
	mask_3 = np.tile(np.expand_dims(mask, axis=-1), (1, 1, 3))
	olay = np.multiply(image, mask_3)
	return olay


def save_output(image_name, pred, d_dir, o_dir):
	pred = pred.squeeze()
	pred = pred.cpu().data.numpy()
	th = 0.1
	pred[pred > th] = 1
	pred[pred <= th] = 0

	img_name = image_name.split("/")[-1]
	image = io.imread(image_name)

	mask = transform.resize(pred, (image.shape[0],image.shape[1]), anti_aliasing=False, mode = 'constant', order=0)
	mask = np.tile(np.expand_dims(mask, axis=-1), (1, 1, 3))
	#kernel = np.ones((3, 3), np.uint8)
	#mask = cv2.erode(mask, kernel, iterations=4)
	olay = image * mask


	#pb_np = np.array(imo)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	io.imsave(o_dir+imidx+'.jpg', olay)
	io.imsave(d_dir + imidx + '.jpg', mask)


if __name__ == '__main__':
	# --------- 1. get image path and name ---------
	
	image_dir = '/home/hypevr/Desktop/data_0616/xy/2/image/'#'/media/hypevr/KEY/tonaci_selected/'#'./test_data/test_images/'
	prediction_dir = '/home/hypevr/Desktop/data_0616/xy/2/mask/'#'/media/hypev/KEY/tonaci_selected_masks/'
	olay_dir = '/home/hypevr/Desktop/data_0616/xy/2/olay/'#'/media/hypevr/KEY/tonaci_selected_olay/'
	model_dir = './saved_models/basnet_bsi_human2_fr0.2_pb_0.2/basnet_209.pth' #refine/
	plate_dir = '/home/hypevr/Desktop/data_0616/xy/2/back'
	
	img_name_list = glob.glob(image_dir + '*.jpg')
	
	# --------- 2. dataloader ---------
	#1. dataload
	##test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],transform=transforms.Compose([RescaleT(352), ToTensorLab(flag=0)])) #,OtherTrans()
	#test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,shuffle=False,num_workers=1)
	# test_salobj_dataloader = get_loader(image_dir, prediction_dir, batchsize=1,
	# 								  trainsize=416)
	test_loader = test_dataset(image_dir, image_dir, 352, True)
	# --------- 3. model define ---------
	print("...load BASNet...")
	net = BASNet(3, 1)
	#net = nn.DataParallel(net)
	net.load_state_dict(torch.load(model_dir)) #, map_location='cuda:0'
	net.cuda()
	net.eval()
	scriptedmodel = torch.jit.script(net)
	torch.jit.save(scriptedmodel, 'scripted_BASNet_57.pt')
	x = torch.ones((1, 3, 352, 352)).cuda()
	torch.onnx.export(net, x, "basnet.onnx", opset_version=11)
	onnx_model = onnx.load("basnet.onnx")
	onnx.checker.check_model(onnx_model)
	#net.eval()
	
	#example = torch.rand(1, 3, 256, 256).cuda()
	#traced_script_module = torch.jit.trace(net, example)
	#traced_script_module.save("traced_model_BASNet.pt")
	net = torch.load('scripted_BASNet_57.pt')
	net.eval()
	
	# --------- 4. inference for each image ---------
	for i in range(test_loader.size):
		image_orig, inputs_test, gt, name = test_loader.load_data()
		##inputs_test = data_test[0]
		
		inputs_test = inputs_test.type(torch.FloatTensor)

		image_resized = inputs_test.numpy()[0, :, :, :].transpose((1, 2, 0))

		#io.imsave('after_resize.png', inputs_test.numpy()[0, :, :, :].transpose((1, 2, 0)))
	
		if torch.cuda.is_available():
			inputs_test = Variable(inputs_test.cuda())
		else:
			inputs_test = Variable(inputs_test)
	
		start = time.time()
		d1 = net(inputs_test)#,d2,d3,d4,d5,d6,d7,d8 ,, d2, d3, d4, d5, d6, d7, d8
		#print(d1)
		torch.cuda.synchronize()
		#d1 = d1.cpu() 
		print(time.time()-start)


		pred = normPRED(d1)
		#pred = overlay(image_resized, pred.squeeze().cpu().data.numpy())
		# save results to test_results folder
		save_output(image_dir+name, pred,prediction_dir, olay_dir)
	
		#del d1,d2,d3,d4,d5,d6,d7,d8
