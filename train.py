import datetime
import os
from math import sqrt
import cv2
from tqdm import *
import PIL.Image as Image
import scipy.io as sio
import numpy as np
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
from torch.utils import data
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d
from utils.transforms import RandomGaussianBlur
from psp_net import PSPNet
from gt_colormap import gt_cmap
import torch.nn as nn
import ipdb
import datetime
from eval import eval_main
num_classes = 21
args = {
    'train_batch_size': 16,
    'lr': 1e-2 ,#/ sqrt(16 / 4),
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
        if self.mode == 'test':
            img_path, mask_path = self.imgs[index]
            img = cv2.imread(img_path).astype(float)
            mask = sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0]
            return torch.from_numpy(img).float(), torch.from_numpy(np.array(mask))
        elif self.mode == 'train':
            img_path, mask_path = self.imgs[index]
            img = Image.open(img_path)

            mask = Image.fromarray(sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0])
            scale_size = (img.size*(1.5*np.random.rand(1) + 0.5)).astype(np.long)
            random_degree = 20*(np.random.rand(1)-0.5)
            # Data Augmentation
            trans = transforms.Compose([
                transforms.Resize(scale_size, Image.NEAREST),
                transforms.CenterCrop(img.size),
                transforms.RandomRotation((random_degree,random_degree), expand=False),
                transforms.Resize((473,473), Image.NEAREST),
                ])
            img = trans(img)
            gt = trans(mask)
            if np.random.rand(1)>0.5:
                img = transforms.RandomHorizontalFlip(1)(img)
                gt = transforms.RandomHorizontalFlip(1)(gt)
            if np.random.rand(1)>0.5:
                img = transforms.RandomVerticalFlip(1)(img)
                gt = transforms.RandomVerticalFlip(1)(gt)
            img_org = np.copy(img)
            img = RandomGaussianBlur()(img)
            img = transforms.ToTensor()(img)
            img_orig =  transforms.ToTensor()(img_org)
            gt = np.array(gt)
            #img = self.normalize(img)
            mean = [torch.mean(img[0]), torch.mean(img[1]), torch.mean(img[2])]
            std = [torch.std(img[0]), torch.std(img[1]), torch.std(img[2])]
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            #img = transforms.Normalize((123.68, 116.779, 103.939), (1, 1, 1))(img)

            return img, torch.from_numpy(gt), img_orig

    def __len__(self):
        return len(self.imgs)


if __name__=='__main__':
    # Network define
    net = PSPNet(num_classes=num_classes)#.cuda()


    '''optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'], nesterov=True)'''


    # Summary writer
    net.train()
    time_current = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    writer = SummaryWriter('logs/'+time_current)
    #dummy_input = Variable(torch.rand(2, 3, 473, 473))
    #writer.add_graph(net, (dummy_input, ))

    # Net train
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.train()
    train_set = VOC('train')
    train_loader = DataLoader(train_set, batch_size=8, num_workers=8, shuffle=True)
    train_loss = []
    train_accu = []
    i = 0


    # Loss function
    criterion = CrossEntropyLoss2d(size_average=True, ignore_index=255).cuda()
    lr = 0.01
    while(1):
        for data in train_loader:
            lr_tmp = lr*(1 - i/args['max_iter'])**0.9
            optimizer = optim.SGD(net.parameters(), lr_tmp, momentum=0.9, nesterov=True, weight_decay=0.0001)
            net.train()
            img, gt, img_org = data
            #img.cuda()
            #gt.cuda()
            #ipdb.set_trace()
            img = Variable(img.cuda())
            #ipdb.set_trace()
            gt = Variable(gt.cuda().long())
            optimizer.zero_grad()
            outputs, aux = net(img)
            main_loss = criterion(outputs, torch.squeeze(gt))#nn.NLLLoss2d(output, gt)
            aux_loss = criterion(aux, torch.squeeze(gt))#nn.NLLLoss2d(aux, gt)    # calc gradients

            loss = main_loss + 0.4 * aux_loss
            train_loss.append(loss.data[0])
            loss.backward()
            optimizer.step()   # update gradients
            if i % 100 == 0:
                #train_accu.append(accuracy)
                print('Train Step: {}  Loss: {}'.format(i, loss.data[0]))
                writer.add_scalar('lr', lr_tmp, i)
                writer.add_scalar('train/loss', loss.data[0], i)
                #writer.add_image('train/img', img.data, i)
                writer.add_image('train/img_orig', img_org, i)
                gts = gt_cmap(gt)
                writer.add_image('train/gt', gts,i)
                #writer.add_image('train/gt', gt.data, i)
            i += 1
            if i==args['max_iter']:
                break
            if i % 10000 == 0:# and i!=0:
                print('taking snapshot ...')
                torch.save(net.module.state_dict(),'logs/'+time_current+'/'+'VOC12_pyramid_'+str(i)+'.pth')
                eval_main(net, writer, i)
    eval_main(net,writer,i)

