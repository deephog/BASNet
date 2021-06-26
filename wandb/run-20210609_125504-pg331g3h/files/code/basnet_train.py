import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import wandb
import torchvision.transforms as standard_transforms

import numpy as np
import sys
import glob
import time

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop, OtherTrans
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from skimage import io

from model import BASNet

import pytorch_ssim
import pytorch_iou
import os
# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

hyperparameter_defaults = {
              "gpu": '0',
              "learning_rate": 1e-4,
              "lr_decay": 0,
              "epochs": 1000,
              "batch_size": 16,
              "checkpoint": False,
              "load_pretrained": True,
              "model_dir": "./saved_models/basnet_bsi_refine/"
}

run = wandb.init(project='basnet_refine', config=hyperparameter_defaults, save_code='on', mode='online', reinit=True)
config = run.config
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
def bce_ssim_loss(pred,target):

    bce_out = bce_loss(pred,target)
    ssim_out = 1 - ssim_loss(pred,target)
    iou_out = iou_loss(pred,target)

    loss = bce_out + ssim_out + iou_out

    return loss

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v):

    loss0 = bce_ssim_loss(d0,labels_v)
    loss1 = bce_ssim_loss(d1,labels_v)
    loss2 = bce_ssim_loss(d2,labels_v)
    loss3 = bce_ssim_loss(d3,labels_v)
    loss4 = bce_ssim_loss(d4,labels_v)
    loss5 = bce_ssim_loss(d5,labels_v)
    loss6 = bce_ssim_loss(d6,labels_v)
    loss7 = bce_ssim_loss(d7,labels_v)
    #ssim0 = 1 - ssim_loss(d0,labels_v)

    # iou0 = iou_loss(d0,labels_v)
    #loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7#+ 5.0*lossa
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data[0],loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],loss6.data[0]))
    # print("BCE: l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, la:%3f, all:%3f\n"%(loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],lossa.data[0],loss.data[0]))
    #print("\r l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f" % (loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()))
    return loss0, loss


# ------- 2. set the directory of training dataset --------

data_dir = '/home/hypevr/data/projects/data/combined_human/'
tra_image_dir = 'train/image/'
tra_label_dir = 'train/mask/'

te_image_dir = 'val/image/'
te_label_dir = 'val/mask/'

# tra_image_dir = 'dummy_img/'
# tra_label_dir = 'dummy_gt/'
#
# te_image_dir = 'dummy_img/'
# te_label_dir = 'dummy_gt/'

image_ext = '.jpg'
label_ext = '.jpg'

model_dir = config.model_dir

##############################
checkpoint = config.checkpoint
load_pretrained = config.load_pretrained
#############################

if checkpoint:
    checkpoint_dir = model_dir + 'basnet_' + str(checkpoint) + '.pth'

if load_pretrained:
    checkpoint_dir = './saved_models/basnet_bsi/basnet.pth'

epoch_num = config.epochs
batch_size_train = config.batch_size
batch_size_val = config.batch_size
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
te_img_name_list = glob.glob(data_dir + te_image_dir + '*' + image_ext)

tra_lbl_name_list = []
te_lbl_name_list = []




for img_path in tra_img_name_list:
    img_name = img_path.split("/")[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

for img_path in te_img_name_list:
    img_name = img_path.split("/")[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    te_lbl_name_list.append(data_dir + te_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("test images: ", len(te_img_name_list))
print("test labels: ", len(te_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)
val_num = len(te_img_name_list)


salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    img_transform=transforms.Compose([OtherTrans()]),
    transform=transforms.Compose([
        RescaleT(512),
        RandomCrop(352),
        ToTensorLab(flag=0),
    ]))

salobj_dataset_te = SalObjDataset(
    img_name_list=te_img_name_list,
    lbl_name_list=te_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(352),
        #RandomCrop(224),
        ToTensorLab(flag=0),
    ]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)
salobj_dataloader_te = DataLoader(salobj_dataset_te, batch_size=1, shuffle=False, num_workers=1)

# ------- 3. define model --------
# define the net
net = BASNet(3, 1)
if torch.cuda.is_available():
    net.cuda()

if checkpoint or load_pretrained:
    net.load_state_dict(torch.load(checkpoint_dir))


# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.lr_decay)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0

for epoch in range(0, epoch_num):
    if checkpoint:
        epoch += checkpoint

    net.train()
    start_time = time.time()

    for i, data in enumerate(salobj_dataloader):
        if not i == 0:
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")

        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        # print(inputs.shape)
        #
        # io.imsave('temp.jpg', inputs[0, 0, :, :]*255)
        # io.imsave('temp.png', labels[0, 0, :, :]*255)
        # input('wait')

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6, d7 = net(inputs_v)#
        loss0, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v) #loss2

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.item()
        running_tar_loss += loss0.item()

        # del temporary outputs and loss
        del d0, loss#d1, d2, d3, d4, d5, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f , time_lapse: %3f" % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
            running_tar_loss / ite_num4val, time.time()-start_time))

    wandb.log({'epochs': epoch,
               'train_loss': float(running_tar_loss / ite_num4val),
               })

    if epoch % 2 == 1:  # save model every 2000 iterations
        # basnet_bsi_itr_%d_train_%3f_tar_%3f.pth basnet_time.pth
        # torch.save(net.state_dict(), model_dir + "basnet_bsi_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        net.eval()
        ind_v = 0
        running_loss_v = 0.0
        running_tar_loss_v = 0.0
        for i, data in enumerate(salobj_dataloader_te):
            if not i == 0:
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")
            ind_v += 1
            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            d0, d1, d2, d3, d4, d5, d6, d7 = net(inputs_v)#
            loss0, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v)#d1, d2, d3, d4, d5,

            running_loss_v += loss.item()
            running_tar_loss_v += loss0.item()

            print("(Validation Phase) [epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_val, val_num, ind_v, running_loss_v / ind_v,
                running_tar_loss_v / ind_v))
            # # print statistics

            del d0, loss #d1, d2, d3, d4, d5, loss2,

        torch.save(net.state_dict(), model_dir + "basnet_%d.pth" % (epoch))
        #running_loss = 0.0
        #running_tar_loss = 0.0
        net.train()  # resume train
        #ite_num4val = 0
        wandb.log({
                   'val_loss': float(running_tar_loss_v / ind_v)
                   })

    # sys.stdout.write("\033[F")
    # sys.stdout.write("\033[F")
    # sys.stdout.write("\033[F")


print('-------------Congratulations! Training Done!!!-------------')
