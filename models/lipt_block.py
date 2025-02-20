import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import numbers
import random
from torch.nn.utils import weight_norm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.transforms import functional as TF
from .diversebranchblock import DiverseBranchBlock


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class HRM(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type='relu',act_num=3,deploy = False):
        super(HRM, self).__init__()    
        self.exp_ratio = exp_ratio
        self.act_type  = act_type
        self.deploy = deploy

        if self.deploy==True:
            self.conv0 = nn.Conv2d(inp_channels, out_channels*exp_ratio,3,1,1)
            self.conv1 = nn.Conv2d(out_channels*exp_ratio, out_channels,3,1,1)
        else:
            self.conv0 = DiverseBranchBlock(inp_channels, out_channels*exp_ratio,3,1,1)
            self.conv1 = DiverseBranchBlock(out_channels*exp_ratio, out_channels,3,1,1)
        
        if self.act_type == 'linear':
            self.act = None
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError('unsupport type of activation')

    def forward(self, x):
        y = self.conv0(x)
        y = self.act(y)
        y = self.conv1(y) 
        
        return y


class MASK(nn.Module):
    def __init__(self, mask_type,pad_type='ref_pad'):
        super(MASK, self).__init__()  
        self.padding_mode = pad_type
        self.mask_type = mask_type
        if self.mask_type =='dense':
            self.mask_1 = torch.tensor([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],dtype=bool)
            self.mask_2 = torch.tensor([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],dtype=bool)  
            self.mask_3 = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=bool)
            self.mask_4 = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=bool) 
        elif self.mask_type == 'sparse':
            self.mask_1 = torch.tensor([1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1],dtype=bool)
            self.mask_2 = torch.tensor([1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1],dtype=bool)  
            self.mask_3 = torch.tensor([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],dtype=bool)
            self.mask_4 = torch.tensor([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],dtype=bool) 
        elif self.mask_type == 'sector':
            self.mask_1 = torch.tensor([0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1],dtype=bool)
            self.mask_2 = torch.tensor([0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1],dtype=bool)  
            self.mask_3 = torch.tensor([0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],dtype=bool)
            self.mask_4 = torch.tensor([0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],dtype=bool) 
        elif self.mask_type == 'random':
            mask = random.sample(range(0,16),16)
            mask1 = [0]*16
            for i in mask[:8]:
                mask1[i]=1 
            self.mask_1 = torch.tensor(mask1,dtype=bool)
            self.mask_2 = self.mask_1

    def seq_refl_win_pad(self, x,win=8, back=False):
        if win == 1: return x
        x = TF.pad(x, (0,0,win,win)) if not back else TF.pad(x, (win,win,0,0))
        #import pdb;pdb.set_trace()
        if self.padding_mode == 'zero_pad':
            return x
        if not back:
            if self.padding_mode == 'sparse':
                (start_h, start_w), (end_h, end_w) = to_2tuple(2*win), to_2tuple(win)
                # pad lower
                x[:,:,-(win):,:] = x[:,:,0:end_h,:].contiguous()
                # pad right
                x[:,:,:,-(win):] = x[:,:,:,0:end_w].contiguous()
            else:
                (start_h, start_w), (end_h, end_w) = to_2tuple(-2*win), to_2tuple(-win)
                # pad lower
                x[:,:,-(win):,:] = x[:,:,start_h:end_h,:].contiguous()
                # pad right
                x[:,:,:,-(win):] = x[:,:,:,start_w:end_w].contiguous()
        else:
            (start_h, start_w), (end_h, end_w) = to_2tuple(win), to_2tuple(2*win)
            # pad upper
            x[:,:,:win,:] = x[:,:,start_h:end_h,:].contiguous()
            # pad left
            x[:,:,:,:win] = x[:,:,:,start_w:end_w].contiguous()
            
        return x
    def forward(self, x, window_size):
        x_pad = self.seq_refl_win_pad(x,window_size, False)
        x_un = x_pad.unfold(3, 2*window_size, window_size).unfold(2, 2*window_size, window_size)
        if window_size==8:
            mask1=self.mask_1
            mask2=self.mask_2
        else:
            mask1=self.mask_3
            mask2=self.mask_4
        gla_X = x_un[:,:,:,:,mask1,:]
        x_ex = gla_X[:,:,:,:,:,mask2]
        x = rearrange(
            x_ex, 'b c hh hw h w ->b c (hh w) (hw h)'
        )
        return x

class NVSM_SA(nn.Module):
    def __init__(self, channels, shifts=4, window_sizes=[8, 8, 8], calc_attn=True,deploy = False):
        super(NVSM_SA, self).__init__()    
        self.channels = channels
        self.shifts   = shifts
        self.window_sizes = window_sizes
        self.calc_attn = calc_attn
        self.deploy = deploy


        #self.mask_dense = MASK('dense')
        self.mask_sparse = MASK(mask_type='sparse',pad_type='sparse')
        # self.mask_sector = MASK(mask_type='sector',pad_type='sparse')
        # self.mask_random = MASK(mask_type='random')

        if self.calc_attn:
            self.split_chns  = [channels*2//2,channels*2//2]
            if self.deploy==False:
                self.project_inp = nn.Sequential(
                    nn.Conv2d(self.channels, self.channels*2, kernel_size=1), #3,stride=1,padding=1
                    nn.BatchNorm2d(self.channels*2)
                )
            else:
                 self.project_inp = nn.Conv2d(self.channels, self.channels*2, kernel_size=1)
            self.project_out = nn.Conv2d(channels, channels, kernel_size=1)
        else:
            self.split_chns  = [channels//2,channels//2]
            if self.deploy==False:
                self.project_inp = nn.Sequential(
                    nn.Conv2d(self.channels, self.channels, kernel_size=1), 
                    nn.BatchNorm2d(self.channels)
                )
            else:
                self.project_inp = nn.Conv2d(self.channels, self.channels, kernel_size=1)
            self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, prev_atns = None):
        b,c,h,w = x.shape
        #x = self.mask(x,self.window_sizes[0])
        x = self.project_inp(x)
        xs = torch.split(x, self.split_chns, dim=1)
        ys = []
        atns = []
        if prev_atns is None:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize//2, -wsize//2), dims=(2,3))
                #x_ = self.mask_random(x_,wsize)
                if idx == 1:
                    x_ = self.mask_sparse(x_,wsize)
                # elif idx == 2:
                #     x_ = self.mask_sector(x_,wsize)
                q, v = rearrange(
                    x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c', 
                    qv=2, dh=wsize, dw=wsize
                )
                atn = (q @ q.transpose(-2, -1))
                atn = atn.softmax(dim=-1)
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)', 
                    h=h//wsize, w=w//wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize//2, wsize//2), dims=(2, 3))
                ys.append(y_)
                atns.append(atn)
            y = torch.cat(ys, dim=1)            
            y = self.project_out(y)
            return y, atns
        else:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize//2, -wsize//2), dims=(2,3))
                atn = prev_atns[idx]
                v = rearrange(
                    x_, 'b (c) (h dh) (w dw) -> (b h w) (dh dw) c', 
                    dh=wsize, dw=wsize
                )
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)', 
                    h=h//wsize, w=w//wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize//2, wsize//2), dims=(2, 3))
                ys.append(y_)
            y = torch.cat(ys, dim=1)            
            y = self.project_out(y)
            return y, prev_atns

class LIPTB(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, shifts=0, window_sizes=[4, 8, 12], shared_depth=1,deploy=False):
        super(LIPTB, self).__init__()
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.shared_depth = shared_depth
        self.deploy = deploy


        
        modules_hrm_a = {}
        modules_hrm_c = {}
        modules_nvsm_sa = {}
        modules_hrm_b = {}
        modules_hrm_a['lcs_a_0'] = HRM(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio, deploy=deploy)
        modules_hrm_c['lcs_c_0'] = HRM(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio, deploy=deploy)
        modules_nvsm_sa['smmsa_0'] = NVSM_SA(channels=inp_channels, shifts=shifts, window_sizes=window_sizes, calc_attn=True, deploy=deploy)
        modules_hrm_b['lcs_b_0'] = HRM(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio, deploy=deploy)
        for i in range(shared_depth):
            modules_hrm_a['lcs_a_{}'.format(i+1)] = HRM(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio, deploy=deploy)
            modules_hrm_c['lcs_c_{}'.format(i+1)] = HRM(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio, deploy=deploy)
            modules_nvsm_sa['smmsa_{}'.format(i+1)] = NVSM_SA(channels=inp_channels, shifts=shifts, window_sizes=window_sizes, calc_attn=False, deploy=deploy)
            modules_hrm_b['lcs_b_{}'.format(i+1)] = HRM(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio, deploy=deploy)
        self.modules_hrm_a = nn.ModuleDict(modules_hrm_a)
        self.modules_hrm_c = nn.ModuleDict(modules_hrm_c)
        self.modules_nvsm_sa = nn.ModuleDict(modules_nvsm_sa)
        self.modules_hrm_b = nn.ModuleDict(modules_hrm_b)

    def forward(self, x):
        atn = None
        for i in range(1 + self.shared_depth):
            if i == 0: ## only calculate attention for the 1-st module
                x = self.modules_hrm_a['lcs_a_{}'.format(i)](x) + x
                x = self.modules_hrm_c['lcs_c_{}'.format(i)](x) + x
                y, atn = self.modules_nvsm_sa['smmsa_{}'.format(i)](x, None)
                x = y + x
                x = self.modules_hrm_b['lcs_b_{}'.format(i)](x) + x
            else:
                x = self.modules_hrm_a['lcs_a_{}'.format(i)](x) + x
                x = self.modules_hrm_c['lcs_c_{}'.format(i)](x) + x
                y, atn = self.modules_nvsm_sa['smmsa_{}'.format(i)](x, atn)
                x = y + x
                x = self.modules_hrm_b['lcs_b_{}'.format(i)](x) + x
        return x