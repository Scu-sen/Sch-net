import torch
import os
import numpy as np

from time import time
import torch.nn as nn
from  torch.nn import CrossEntropyLoss
import sys
sys.path.append('../')

import warnings
warnings.filterwarnings("ignore")
from time import time
import csv
import argparse


from torch.utils.data import DataLoader
from src.models.aux_skip_attention import AuxSkipAttention
from src.datasets import FreesoundDatasetys, FreesoundNoisyDataset, RandomDataset
from src.mixers import RandomMixer, AddMixer, SigmoidConcatMixer, UseMixerWithProb
from src.transforms import get_transforms
from src import config
import pandas as pd

import torch.backends.cudnn as cudnn
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default='hh', type=str)
args = parser.parse_args()

BATCH_SIZE = 16
CROP_SIZE = 256
DATASET_SIZE = 128 * 256
NOISY_PROB = 0.01
CORR_NOISY_PROB = 0.42
MIXER_PROB = 0.8
WRAP_PAD_PROB = 0.5
CORRECTIONS = True



train_transfrom = get_transforms(train=True,
                                     size=CROP_SIZE,
                                     wrap_pad_prob=WRAP_PAD_PROB,
                                     resize_scale=(0.8, 1.0),
                                     resize_ratio=(1.7, 2.3),
                                     resize_prob=0.33,
                                     spec_num_mask=2,
                                     spec_freq_masking=0.15,
                                     spec_time_masking=0.20,
                                     spec_prob=0.5)

mixer = RandomMixer([
    SigmoidConcatMixer(sigmoid_range=(3, 12)),
    AddMixer(alpha_dist='uniform')
], p=[0.6, 0.4])
mixer = UseMixerWithProb(mixer, prob=MIXER_PROB)





train_csv = pd.read_csv('./train_fold'+str(1)+'.csv')
val_csv = pd.read_csv('./val_fold'+str(1)+'.csv')

train_ds = FreesoundDatasetys(train_csv,
                                   transform=train_transfrom,
                                   mixer=None)
val_ds = FreesoundDatasetys(val_csv,get_transforms(False, CROP_SIZE))


net=AuxSkipAttention().cuda()


on_server = True
save_model_path = "./result/"  # save Pytorch models

cudnn.benchmark = True
Epoch = 30
leaing_rate_base = 3e-4

batch_size = 16

num_workers = 8
pin_memory =True


train_dl = DataLoader(train_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)
val_dl = DataLoader(val_ds, 1, True, num_workers=num_workers, pin_memory=pin_memory)

print(len(train_dl.dataset))

print(len(val_dl.dataset))


loss_fun=CrossEntropyLoss()




opt = torch.optim.Adam(params=net.parameters(), lr=leaing_rate_base)

lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [10,20])
best_pth=30
train_loss=[]
train_acc=[]

val_loss=[]
val_acc=[]

def validation(model, optimizer, test_loader,best_pth,isval=True):
    # set model as testing mode
    model.eval()
    pre_all=[]
    tre_all=[]

    test_loss = 0
    all_y = []
    all_y_pred = []

    prob_all=[]
    num = 0
    with torch.no_grad():
        for X, y  in test_loader:
            all_y.append(y.cpu().numpy())
            tre=y
            y = y.type(torch.LongTensor).cuda()
            num=num+1

            # distribute data to device
            X= X.cuda()

            outputs = model(X)


            loss= loss_fun(outputs, y)

            mask=outputs.cpu().numpy()

            tmp_prob=mask[:,1]

            prob_all.extend(tmp_prob)

            tmp=np.argmax(mask, 1)
            # print(tmp)
            all_y_pred.append(tmp)

            test_loss += loss.item()    # sum up batch loss

    test_loss /= num
    test_loss=test_loss

    y_true=np.array(all_y)
    y_pre=np.array(all_y_pred)

    acc_sk=accuracy_score(y_true,y_pre)
    precis=precision_score(y_true,y_pre)
    recall=recall_score(y_true,y_pre)
    f1=f1_score(y_true,y_pre)


    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 1- test_loss))

    print([acc_sk,precis,recall,recall,f1])



    return [acc_sk,precis,recall,f1]

# 训练网络

dirs = save_model_path
log_path=dirs
out_dir=dirs

if not os.path.exists(dirs):
    os.makedirs(dirs)
save_model_path = dirs


start = time()
for epoch in range(Epoch):

    net.train()

    lr_decay.step()

    mean_loss = []

    for step, (ct, seg) in enumerate(train_dl):

        ct = ct.cuda()
        # print(ct.size())
        seg=seg.type(torch.LongTensor).cuda()


        outputs = net(ct)
        loss4 = loss_fun(outputs, seg)
        loss=loss4


        mean_loss.append(loss4.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 2 is 0:
            print('epoch:{}, step:{}, loss1:{:.3f}'
                  .format(epoch, step,  loss.item(), (time() - start) / 60))
    val_acc1 = validation(net, opt, val_dl, best_pth, isval=True)


    print(val_acc1)

    net.train()
    mean_loss = sum(mean_loss) / len(mean_loss)