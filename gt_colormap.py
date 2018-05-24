import numpy as np
import torch
from torch.autograd import Variable
import ipdb
def make_one_hot(labels, C=21):
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    target = Variable(target)

    return target

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
            cmap[i] = np.array([r, g, b])
    cmap = cmap/255 if normalized else cmap
    return cmap

cmap = color_map()[:21]
cmap = Variable(torch.from_numpy(cmap)).cuda()

def gt_cmap(gts):
    #target = make_one_hot(gts, C=21)
    gts_stack = torch.stack([gts,gts,gts],1)
    out = torch.zeros_like(gts_stack).float()
    for i in range(21):
        tmp = gts_stack==i
        out += tmp.float()*cmap[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(0).float()
    #target = target.transpose(1,-1)
    #ipdb.set_trace()
    #out = target*cmap
    return out.byte()

