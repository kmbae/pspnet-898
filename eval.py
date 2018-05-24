import datetime
import os
from math import sqrt
import cv2
from tqdm import *
import scipy.io as sio
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import PIL.Image as Image
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from gt_colormap import gt_cmap
from torch import optim
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
from torch.utils import data
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d
from psp_net import PSPNet
import torch.nn as nn
import ipdb

num_classes = 21
args = {
    'train_batch_size': 4,
    'lr': 1e-2 / sqrt(16 / 4),
    'lr_decay': 0.9,
    'max_iter': 3e4,
    'longer_size': 512,
    'crop_size': 473,
    'stride_rate': 2 / 3.,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot': '',
    'print_freq': 10,
    'val_save_to_img_file': True,
    'val_img_sample_rate': 0.01,  # randomly sample some validation results to display,
    'val_img_display_size': 384,
}

def make_dataset(mode):
    assert mode in ['train', 'val', 'test']
    items = []
    root = 'VOC'
    if mode == 'train':
        img_path = os.path.join('VOC', 'benchmark_RELEASE', 'dataset', 'img')
        mask_path = os.path.join('VOC', 'benchmark_RELEASE', 'dataset', 'cls')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'benchmark_RELEASE', 'dataset', 'train.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.mat'))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join('VOC', 'benchmark_RELEASE', 'dataset', 'img')
        mask_path = os.path.join('VOC', 'benchmark_RELEASE', 'dataset', 'cls')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'benchmark_RELEASE', 'dataset', 'val.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.mat'))
            items.append(item)
    return items

class VOC(data.Dataset):
    def __init__(self, mode):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'val':
            img_path, mask_path = self.imgs[index]
            img = Image.open(img_path)
            img = transforms.Resize((473,473))(img)
            img_orig = np.copy(img)
            img = transforms.ToTensor()(img)
            img_orig = transforms.ToTensor()(img_orig)
            mean = [torch.mean(img[0]), torch.mean(img[1]), torch.mean(img[2])]
            std = [torch.std(img[0]), torch.std(img[1]), torch.std(img[2])]
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            #img = transforms.Normalize(mean, std)(img)
            mask = Image.fromarray(sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0])
            mask = transforms.Resize((473,473), Image.NEAREST)(mask)
            gt = np.array(mask)
            return img, torch.from_numpy(gt), img_orig

        elif self.mode == 'train':
            img_path, mask_path = self.imgs[index]
            img = cv2.imread(img_path).astype(float)
            img = cv2.resize(img,(473,473)).astype(float)
            img = img.transpose([2,0,1])
            mask = sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0]
            mask = cv2.resize(mask,(473,473),interpolation = cv2.INTER_NEAREST).astype(float)
            #scale = random.uniform(0.5, 2)
            return torch.from_numpy(img).float(), torch.from_numpy(np.array(mask), )

    def __len__(self):
        return len(self.imgs)

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def get_iou(pred,gt):
    if pred.shape!= gt.shape:
        print('pred shape',pred.shape, 'gt shape', gt.shape)
    assert(pred.shape == gt.shape)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    max_label = int(num_classes)-1  # labels from 0,1, ... 20(for VOC)
    count = np.zeros((max_label+1,))
    for j in range(max_label+1):
        x = np.where(pred==j)
        p_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
        x = np.where(gt==j)
        GT_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
        #pdb.set_trace()
        n_jj = set.intersection(p_idx_j,GT_idx_j)
        u_jj = set.union(p_idx_j,GT_idx_j)


        if len(GT_idx_j)!=0:
            count[j] = float(len(n_jj))/float(len(u_jj))

    result_class = count
    Aiou = np.sum(result_class[:])/float(len(np.unique(gt)))

    return Aiou, result_class

def eval_main(net, writer, epoch):
    #net = PSPNet(num_classes=num_classes).cuda()
    #net.load_state_dict(torch.load('snapshots2/VOC12_pyramid_31000.pth'))
    hist = np.zeros((21, 21))
    net.eval()
    train_set = VOC('val')
    data_len = train_set.__len__()
    train_loader = DataLoader(train_set, batch_size=1, num_workers=8, shuffle=False)
    pytorch_list = []
    i = 0
    #miou = np.zeros(21)
    loss = 0.
    criterion = CrossEntropyLoss2d(size_average=True, ignore_index=255).cuda()
    tmp_orig = []
    tmp_gt = []
    tmp_pred=[]
    for data in train_loader:
        img, gt, img_orig = data
        #img.cuda()
        #gt.cuda()
        img = Variable(img.cuda())
        gt = Variable(gt.cuda().long())
        outputs = net(img)
        main_loss = criterion(outputs, (gt))
        loss += main_loss.data
        _, indices = outputs.max(1)
        #ipdb.set_trace()
        #indices = indices.cpu()
        pred = np.array(indices.data)
        gts = np.array(gt.data)
        pred = pred.transpose(1,2,0)
        gts = gts.transpose(1,2,0)
        #a, b = get_iou(pred, gts)
        iou_pytorch, _ = get_iou(pred,gts)
        pytorch_list.append(iou_pytorch)
        hist += fast_hist(gts.flatten(), pred.flatten(), 21)
        i += 1
        if i%600==0:
            tmp_gt.append(gt)
            tmp_orig.append(img_orig)
            tmp_pred.append(indices)
        if i%1000==0:
            print('Testing process {}/{}'.format(i, data_len))
    miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    # Summary add
    writer.add_scalar('test/miou', np.mean(miou[1:]), epoch)
    writer.add_scalar('test/loss', loss/len(train_loader), epoch)
    indices = torch.cat(tmp_pred,0)
    indices_data = gt_cmap(indices)
    writer.add_image('test/pred', indices_data, epoch)
    img_orig = torch.cat(tmp_orig)
    writer.add_image('test/gt_img', img_orig, epoch)
    gt = torch.cat(tmp_gt)
    gts = gt_cmap(gt)
    writer.add_image('test/gt', gts, epoch)
    print("Mean iou = {},\niou per class = {}".format(np.mean(miou[1:]), miou))
    # Showing images
    #plt.subplot(3,1,1)
    #plt.imshow(np.squeeze(img_orig))
    #plt.subplot(3,1,2)
    #plt.imshow(np.squeeze(gts))
    #plt.subplot(3,1,3)
    #plt.imshow(np.squeeze(pred))
    #plt.show()


if __name__=='__main__':
    net = PSPNet(num_classes=num_classes).cuda()
    net.load_state_dict(torch.load('snapshots2/VOC12_pyramid_31000.pth'))
    hist = np.zeros((21, 21))
    net.eval()
    train_set = VOC('val')
    data_len = train_set.__len__()
    train_loader = DataLoader(train_set, batch_size=1, num_workers=8, shuffle=False)
    pytorch_list = []
    i = 0
    #miou = np.zeros(21)
    criterion = CrossEntropyLoss2d(size_average=True, ignore_index=255).cuda()
    for data in train_loader:
        img, gt, img_orig = data
        #img.cuda()
        #gt.cuda()
        img = Variable(img.cuda())
        outputs = net(img)
        _, indices = outputs.max(1)
        indices = indices.cpu()
        pred = np.array(indices.data)
        gts = gt.numpy()
        pred = pred.transpose(1,2,0)
        gts = gts.transpose(1,2,0)
        #a, b = get_iou(pred, gts)
        iou_pytorch, _ = get_iou(pred,gts)
        pytorch_list.append(iou_pytorch)
        hist += fast_hist(gts.flatten(), pred.flatten(), 21)
        i += 1
        if i%100==0:
            print('Done {}/{}'.format(i, data_len))

    miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print("Mean iou = {}, iou per class = {}".format(np.mean(miou[1:]), miou))
    # Showing images
    plt.subplot(3,1,1)
    plt.imshow(np.squeeze(img_orig))
    plt.subplot(3,1,2)
    plt.imshow(np.squeeze(gts))
    plt.subplot(3,1,3)
    plt.imshow(np.squeeze(pred))
    plt.show()

