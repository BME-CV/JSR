import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as nnf
# from share import Encoder
from torch.distributions.normal import Normal
import numpy as np
from torch.autograd import Variable
class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)

        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )
    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class CConv(nn.Module):
    def __init__(self, channel,outchannel):
        super(CConv, self).__init__()

        c = channel
        cout = outchannel

        self.conv = nn.Sequential(
            ConvInsBlock(c, cout, 3, 1),
        )

    def forward(self, float_fm, fixed_fm, d_fm):
        concat_fm = torch.cat([float_fm, fixed_fm, d_fm], dim=1)

        x = self.conv(concat_fm)
        return x

class DeMostFromFeats(nn.Module):
    def __init__(self, inshape=(160, 192, 160), channels=16):
        super().__init__()
        self.channels = channels
        self.inshape  = inshape
        c = channels

        # 不再需要 encoder，因为输入就是特征图
        self.upsample       = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear',
                                           align_corners=True)

        # 与原代码保持一致的多尺度池化
        self.pool5 = nn.Sequential(nn.AvgPool3d(16, 16), nn.Conv3d(1, 32*c, 1))
        self.pool4 = nn.Sequential(nn.AvgPool3d(8, 8),  nn.Conv3d(1, 16*c, 1))
        self.pool3 = nn.Sequential(nn.AvgPool3d(4, 4),  nn.Conv3d(1, 8*c, 1))
        self.pool2 = nn.Conv3d(1, 4*c, 3, 2, 1)
        self.pool1 = nn.Conv3d(1, 2*c, 1, 1)

        # warp / diff
        self.warp = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(5):
            self.warp.append(SpatialTransformer([s // 2**i for s in inshape]))
            self.diff.append(VecInt([s // 2**i for s in inshape]))

        # --------- 以下结构与原 De_most 保持一致 ---------
        self.cconv_5 = ConvInsBlock(16*2*c, 4*8*c)
        self.defconv5 = nn.Conv3d(4*8*c, 3, 3, 1, 1)
        self.defconv5.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv5.weight.shape))
        self.defconv5.bias   = nn.Parameter(torch.zeros(self.defconv5.bias.shape))
        self.dconv5  = ConvInsBlock(3*32*c, 4*8*c)

        self.upconv4 = UpConvBlock(8*4*c, 4*4*c, 4, 2)
        self.cconv_4 = ConvInsBlock(8*3*2*c, 8*2*c)
        self.defconv4 = nn.Conv3d(8*2*c, 3, 3, 1, 1)
        self.defconv4.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv4.weight.shape))
        self.defconv4.bias   = nn.Parameter(torch.zeros(self.defconv4.bias.shape))

        self.upconv3 = UpConvBlock(8*2*c, 4*2*c, 4, 2)
        self.cconv_3 = CConv(8*3*c, 2*4*c)
        self.defconv3 = nn.Conv3d(4*2*c, 3, 3, 1, 1)
        self.defconv3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv3.weight.shape))
        self.defconv3.bias   = nn.Parameter(torch.zeros(self.defconv3.bias.shape))

        self.upconv2 = UpConvBlock(4*2*c, 2*2*c, 4, 2)
        self.cconv_2 = CConv(4*3*c, 4*c)
        self.defconv2 = nn.Conv3d(2*2*c, 3, 3, 1, 1)
        self.defconv2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv2.weight.shape))
        self.defconv2.bias   = nn.Parameter(torch.zeros(self.defconv2.bias.shape))

        self.upconv1 = UpConvBlock(2*2*c, 2*c, 4, 2)
        self.cconv_1 = CConv(3*2*c, 2*c)
        self.defconv1 = nn.Conv3d(2*c, 3, 3, 1, 1)
        self.defconv1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv1.weight.shape))
        self.defconv1.bias   = nn.Parameter(torch.zeros(self.defconv1.bias.shape))

    # 与原 compute_err 一致
    def comput_err(self, g, t):
        abs_e = torch.abs(g - t)
        sq_e  = (g - t) ** 2
        e = 0.5 * abs_e + 0.5 * sq_e
        e_min = e.amin(dim=[2, 3, 4], keepdim=True)
        e_max = e.amax(dim=[2, 3, 4], keepdim=True)
        return (e - e_min) / (e_max - e_min + 1e-8)

    # **关键修改：forward 现在接收 5 层特征图 + 原始 moving/fixed**
    def forward(self, moving, fixed, feats_m, feats_f):
        # feats_m / feats_f 均为 tuple (f0,f1,f2,f3,f4)
        M1, M2, M3, M4, M5 = feats_m
        F1, F2, F3, F4, F5 = feats_f

        er = ar = br = cr = dr = 1  # 可按需调大

        # level 5
        C5 = torch.cat([F5, M5], dim=1)
        C5 = self.cconv_5(C5)
        flow = self.defconv5(C5)
        flow_all = self.diff[4](flow)

        flowx5 = F.interpolate(16*flow_all, scale_factor=16, mode='trilinear', align_corners=True)
        deformM5 = self.warp[0](moving, flowx5)
        diff_map5 = self.comput_err(deformM5, fixed)
        sig_diff_map5 = self.pool5(diff_map5)

        for _ in range(er):
            warped = self.warp[4](M5, flow_all)
            warped = warped * sig_diff_map5
            C5 = self.dconv5(torch.cat([F5, warped, C5], dim=1))
            v = self.defconv5(C5)
            w = self.diff[4](v)
            flow_all = self.warp[4](flow_all, w) + w

        # level 4
        flowx4 = F.interpolate(16*flow_all, scale_factor=16, mode='trilinear', align_corners=True)
        deformM4 = self.warp[0](moving, flowx4)
        diff_map4 = self.comput_err(deformM4, fixed)
        sig_diff_map4 = self.pool4(diff_map4)

        flow_all = F.interpolate(2*flow_all, scale_factor=2, mode='trilinear', align_corners=True)
        for _ in range(ar):
            D4 = self.upconv4(C5)
            warped = self.warp[3](M4, flow_all)
            warped = warped * sig_diff_map4
            C4 = self.cconv_4(torch.cat([F4, warped, D4], dim=1))
            v = self.defconv4(C4)
            w = self.diff[3](v)
            flow_all = self.warp[3](flow_all, w) + w

        # level 3
        flowx3 = F.interpolate(8*flow_all, scale_factor=8, mode='trilinear', align_corners=True)
        deformM3 = self.warp[0](moving, flowx3)
        diff_map3 = self.comput_err(deformM3, fixed)
        sig_diff_map3 = self.pool3(diff_map3)

        flow_all = F.interpolate(2*flow_all, scale_factor=2, mode='trilinear', align_corners=True)
        for _ in range(br):
            D3 = self.upconv3(C4)
            warped = self.warp[2](M3, flow_all)
            warped = warped * sig_diff_map3
            C3 = self.cconv_3(F3, warped, D3)
            v = self.defconv3(C3)
            w = self.diff[2](v)
            flow_all = self.warp[2](flow_all, w) + w

        # level 2
        flowx2 = F.interpolate(4*flow_all, scale_factor=4, mode='trilinear', align_corners=True)
        deformM2 = self.warp[0](moving, flowx2)
        diff_map2 = self.comput_err(deformM2, fixed)
        sig_diff_map2 = self.pool2(diff_map2)

        flow_all = F.interpolate(2*flow_all, scale_factor=2, mode='trilinear', align_corners=True)
        for _ in range(cr):
            D2 = self.upconv2(C3)
            warped = self.warp[1](M2, flow_all)
            warped = warped * sig_diff_map2
            C2 = self.cconv_2(F2, warped, D2)
            v = self.defconv2(C2)
            w = self.diff[1](v)
            flow_all = self.warp[1](flow_all, w) + w

        # level 1
        flowx1 = F.interpolate(2*flow_all, scale_factor=2, mode='trilinear', align_corners=True)
        deformM1 = self.warp[0](moving, flowx1)
        diff_map1 = self.comput_err(deformM1, fixed)
        sig_diff_map1 = self.pool1(diff_map1)

        flow_all = F.interpolate(2*flow_all, scale_factor=2, mode='trilinear', align_corners=True)
        for _ in range(dr):
            D1 = self.upconv1(C2)
            warped = self.warp[0](M1, flow_all)
            warped = warped * sig_diff_map1
            C1 = self.cconv_1(F1, warped, D1)
            v = self.defconv1(C1)
            w = self.diff[0](v)
            flow_all = self.warp[0](flow_all, w) + w

        y_moved = self.warp[0](moving, flow_all)
        field = [flowx4, flowx3, flowx2, flowx1, flow_all]
        return y_moved, flow_all, field

class Reg_Decoder(nn.Module):
    def __init__(self, inshape=(160, 192, 160), channels=16):
        super(Reg_Decoder, self).__init__()
        self.channels = channels
        self.step = 7
        self.inshape = inshape

        c = self.channels
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear',
                                           align_corners=True) 

        # self.pool5 = nn.Sequential(
        #     nn.AvgPool3d(kernel_size=16,stride=16),
        #     nn.Conv3d(1, 32*c, kernel_size=1,stride=1))
        # self.pool4 = nn.Sequential(
        #     nn.AvgPool3d(kernel_size=8, stride=8),
        #     nn.Conv3d(1, 16*c, kernel_size=1,stride=1))
        # self.pool3 = nn.Sequential(
        #     nn.AvgPool3d(kernel_size=4, stride=4),
        #     nn.Conv3d(1, 8*c, kernel_size=1,stride=1))
        # self.pool2 = (
        #     nn.Conv3d(1, 4*c, kernel_size=3,stride=2,padding=1))
        # self.pool1 = nn.Conv3d(1, 2*c, kernel_size=1,stride=1)

        self.warp = nn.ModuleList()
        self.warp_label = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(5):
            self.warp.append(SpatialTransformer([s // 2 ** i for s in inshape]))
            self.warp_label.append(SpatialTransformer([s // 2 ** i for s in inshape],mode='nearest'))
            self.diff.append(VecInt([s // 2 ** i for s in inshape]))

        # bottleNeck
        self.cconv_5 = nn.Sequential(
            ConvInsBlock(16*2* c, 16*c, 3, 1),  
        )
        # warp scale 2
        self.defconv5 = nn.Conv3d(16* c, 3, 3, 1, 1)
        self.defconv5.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv5.weight.shape))
        self.defconv5.bias = nn.Parameter(torch.zeros(self.defconv5.bias.shape))
        
        self.dconv5 = nn.Sequential(
            ConvInsBlock(16*3* c, 16 * c),
        )

        self.upconv4 = UpConvBlock(16*c, 8*c, 4, 2)
      


        self.cconv_4 = nn.Sequential(
            ConvInsBlock(8*3* c, 8*c, 3, 1),
            
        )
    
        self.defconv4 = nn.Conv3d(8* c, 3, 3, 1, 1)
        self.defconv4.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv4.weight.shape))
        self.defconv4.bias = nn.Parameter(torch.zeros(self.defconv4.bias.shape))
       

        self.upconv3 = UpConvBlock(8* c, 4* c, 4, 2)
        self.cconv_3 = CConv(4* 3 * c,4*c)

     
        self.defconv3 = nn.Conv3d(4*c, 3, 3, 1, 1)
        self.defconv3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv3.weight.shape))
        self.defconv3.bias = nn.Parameter(torch.zeros(self.defconv3.bias.shape))
        

        self.upconv2 = UpConvBlock(4* c, 2* c, 4, 2)
        self.cconv_2 = CConv(2*3*c,2* c)

       
        self.defconv2 = nn.Conv3d(2*c, 3, 3, 1, 1)
        self.defconv2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv2.weight.shape))
        self.defconv2.bias = nn.Parameter(torch.zeros(self.defconv2.bias.shape))
        

        self.upconv1 = UpConvBlock(2*c, c, 4, 2)
        self.cconv_1 = CConv(3*c,c)

       
        self.defconv1 = nn.Conv3d(c, 3, 3, 1, 1)
        self.defconv1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv1.weight.shape))
        self.defconv1.bias = nn.Parameter(torch.zeros(self.defconv1.bias.shape))
       
        # self.dconv1 = ConvInsBlock(3 * c, c)
        # self.compdiff = self.comput_err

    def comput_err(self,generated,target):
        abs_error = torch.abs(generated - target)
        sq_error = (generated - target) ** 2
        # 可根据需要调整权重
        error_map = 0.5 * abs_error + 0.5 * sq_error

        # 2. 归一化误差图，将数值映射到0~1范围
        # 注意这里的归一化可以防止除以0的问题
        error_map_min = error_map.amin(dim=[2, 3, 4], keepdim=True)
        error_map_max = error_map.amax(dim=[2, 3, 4], keepdim=True)
        error_map_norm = (error_map - error_map_min) / (error_map_max - error_map_min + 1e-8)
        return error_map_norm

    def forward(self,moving,FM,FF): #moving,fixed,
        # encode stage
        M1, M2, M3, M4, M5 = FM[0], FM[1], FM[2],FM[3],FM[4]
        F1, F2, F3, F4, F5 = FF[0], FF[1], FF[2],FF[3],FF[4]
        # c=8, 2c, 4c, 8c,16c,32c  # 160, 80, 40, 20, 5
        # print(F5.shape)

        er = 1
        ar = 1
        br = 1
        cr = 1
        dr = 1
        # first dec layer

        C5 = torch.cat([F5, M5], dim=1)  #16c*2=32
        # print(C5.shape)
        # print(C5.shape)
        C5 = self.cconv_5(C5)  # (1,128,20,24,20)
        flow = self.defconv5(C5)  # (1,3,20,24,20) #第一个变形场
        flow_all = self.diff[4](flow)  # 进行平滑
        #--------------------------------------------------------------------------------------
        # flowx5 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)(16 * flow_all)
        # deformM5 = self.warp[0](moving, flowx5)
        # diff_map5 = self.compdiff(deformM5,fixed)
        # # diff_map5 = torch.sigmoid(diff_map5)
        # sig_diff_map5 = self.pool5(diff_map5)

        for ee in range(er): #32c
            # print(M5.shape)
            warped = self.warp[4](M5, flow_all)  # 复合变形场  32c
            # warped = warped * sig_diff_map5
            # C5 = self.dconv5(torch.cat([F5, warped, C5,sig_diff_map5], dim=1))  #32c
            C5 = self.dconv5(torch.cat([F5, warped, C5], dim=1))  #32c

            v = self.defconv5(C5)  # (1,3,20,24,20)
            w = self.diff[4](v)
            flow_all = self.warp[4](flow_all, w) + w  #复合
        #----------------------------------------------------------------------
        # flowx4 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)(16 * flow_all)
        # deformM4 = self.warp[0](moving, flowx4)
        # diff_map4 = self.compdiff(deformM4,fixed)  # 这里的图像尺度和当前的配准尺度是不匹配的,以及需要进行绝对值吗？
        # sig_diff_map4 = self.pool4(diff_map4)

        flow_all = self.upsample_trilin(2 * flow_all)
        for aa in range(ar): #8c
            D4 = self.upconv4(C5)   #16c
            warped = self.warp[3](M4, flow_all)  # 复合变形场
            # warped = warped * sig_diff_map4
            C4 = self.cconv_4(torch.cat([F4, warped, D4], dim=1))

            v = self.defconv4(C4)  # (1,3,20,24,20)  #为什么要输入三个图像呢
            w = self.diff[3](v)
            flow_all = self.warp[3](flow_all, w) + w
        #---------------------------------------------------------------------------------------
        # flowx3 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)(8 * flow_all)
        # deformM3 = self.warp[0](moving, flowx3)
        # diff_map3 = self.compdiff(deformM3,fixed)
        # sig_diff_map3 = self.pool3(diff_map3)

        flow_all = self.upsample_trilin(2 * flow_all)
        for bb in range(br):#4c
            D3 = self.upconv3(C4)  # (1, 64, 40, 48, 40) C是f，m，def的产物，用来生成单个尺度的最终变形场
            warped = self.warp[2](M3, flow_all)  # (1, 64, 40, 48, 40)
            # warped = warped * sig_diff_map3
            C3 = self.cconv_3(F3, warped, D3)  # (1, 3 * 64, 40, 48, 40)

            v = self.defconv3(C3)
            w = self.diff[2](v)
            flow_all = self.warp[2](flow_all, w) + w
        #-------------------------------------------------------------------------------------------
        # flowx2 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)(4 * flow_all)
        # deformM2 = self.warp[0](moving, flowx2)
        # diff_map2 = self.compdiff(deformM2,fixed)
        # sig_diff_map2 = self.pool2(diff_map2)

        flow_all = self.upsample_trilin(2 * flow_all)
        for cc in range(cr):#2c
            D2 = self.upconv2(C3)
            warped = self.warp[1](M2, flow_all)
            # warped = warped * sig_diff_map2
            C2 = self.cconv_2(F2, warped, D2)

            v = self.defconv2(C2)  # (1,3,80,96,80)
            w = self.diff[1](v)
            flow_all = self.warp[1](flow_all, w) + w
        #=-----------------------------------------------------------------------------
        # flowx1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)(2 * flow_all)
        # deformM1 = self.warp[0](moving, flowx1)
        # diff_map1= self.compdiff(deformM1,fixed)
        # sig_diff_map1 = self.pool1(diff_map1)
        # # print(sig_diff_map1.shape)
        flow_all = self.upsample_trilin(2 * flow_all)
        for dd in range(dr): #1c
            D1 = self.upconv1(C2)  # (1,16,160,196,160)
            warped = self.warp[0](M1, flow_all)  # （1,16,160,196,160)
            # warped = warped * sig_diff_map1
            # print(F1.shape)
            C1 = self.cconv_1(F1, warped, D1)  # （1,48,160,196,160)

            v = self.defconv1(C1)
            w = self.diff[0](v)
            flow_all = self.warp[0](flow_all, w) + w  # （1,3,160,196,160)

        y_moved = self.warp[0](moving, flow_all)
        #field = [flowx4,flowx3,flowx2,flowx1,flow_all]
        return y_moved, flow_all#,field #, diff_map1


class multi_scale_Reg_Decoder(nn.Module):
    def __init__(self, inshape=(160, 192, 160), channels=16):
        super(multi_scale_Reg_Decoder, self).__init__()
        self.channels = channels
        self.step = 7
        self.inshape = inshape

        c = self.channels
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear',
                                           align_corners=True)
        self.warp = nn.ModuleList()
        # self.warp_label = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(5):
            self.warp.append(SpatialTransformer([s // 2 ** i for s in inshape]))
            # self.warp_label.append(SpatialTransformer([s // 2 ** i for s in inshape],mode='nearest'))
            self.diff.append(VecInt([s // 2 ** i for s in inshape]))
        # bottleNeck
        self.cconv_5 = nn.Sequential(
            ConvInsBlock(16 * 2 * c, 16 * c, 3, 1),
        )
        # warp scale 2
        self.defconv5 = nn.Conv3d(16 * c, 3, 3, 1, 1)
        self.defconv5.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv5.weight.shape))
        self.defconv5.bias = nn.Parameter(torch.zeros(self.defconv5.bias.shape))

        self.dconv5 = nn.Sequential(
            ConvInsBlock(16 * 3 * c, 16 * c),
        )

        self.upconv4 = UpConvBlock(16 * c, 8 * c, 4, 2)

        self.cconv_4 = nn.Sequential(
            ConvInsBlock(8 * 3 * c, 8 * c, 3, 1),

        )
        self.defconv4 = nn.Conv3d(8 * c, 3, 3, 1, 1)
        self.defconv4.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv4.weight.shape))
        self.defconv4.bias = nn.Parameter(torch.zeros(self.defconv4.bias.shape))

        self.upconv3 = UpConvBlock(8 * c, 4 * c, 4, 2)
        self.cconv_3 = CConv(4 * 3 * c, 4 * c)

        self.defconv3 = nn.Conv3d(4 * c, 3, 3, 1, 1)
        self.defconv3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv3.weight.shape))
        self.defconv3.bias = nn.Parameter(torch.zeros(self.defconv3.bias.shape))

        self.upconv2 = UpConvBlock(4 * c, 2 * c, 4, 2)
        self.cconv_2 = CConv(2 * 3 * c, 2 * c)

        self.defconv2 = nn.Conv3d(2 * c, 3, 3, 1, 1)
        self.defconv2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv2.weight.shape))
        self.defconv2.bias = nn.Parameter(torch.zeros(self.defconv2.bias.shape))

        self.upconv1 = UpConvBlock(2 * c, c, 4, 2)
        self.cconv_1 = CConv(3 * c, c)

        self.defconv1 = nn.Conv3d(c, 3, 3, 1, 1)
        self.defconv1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv1.weight.shape))
        self.defconv1.bias = nn.Parameter(torch.zeros(self.defconv1.bias.shape))

        # self.dconv1 = ConvInsBlock(3 * c, c)
        # self.compdiff = self.comput_err
        # self.dice = self.dice

    def dice(self,outputs, labels, num_classes=5):
        # print(outputs.shape)
        """Calculate Dice score for each class"""
        # outputs = torch.argmax(outputs, dim=1)
        # print(outputs.shape)
        # labels = torch.argmax(labels, dim=1)
        dice_scores = []

        for cls in range(1, num_classes):  # Start from 1 to skip background
            outputs_cls = (outputs == cls).float()
            # labels_cls = (labels.squeeze(1) == cls).float()
            labels_cls = (labels == cls).float()
            # print(outputs_cls.shape)
            intersection = (outputs_cls * labels_cls).sum()
            union = outputs_cls.sum() + labels_cls.sum()

            if union > 0:
                dice = (2. * intersection) / (union + 1e-8)
                dice_scores.append(dice.item())

        return np.mean(dice_scores) if dice_scores else 0.0

    def forward(self, moving,mov_seg,fix_seg, FM, FF):  # moving,fixed,
        # encode stage
        M1, M2, M3, M4, M5 = FM[0], FM[1], FM[2], FM[3], FM[4]
        F1, F2, F3, F4, F5 = FF[0], FF[1], FF[2], FF[3], FF[4]
        # c=8, 2c, 4c, 8c,16c,32c  # 160, 80, 40, 20, 5
        er = 5
        ar = 5
        br = 5
        cr = 5
        dr = 5
        # first dec layer

        C5 = torch.cat([F5, M5], dim=1)  # 16c*2=32
        # print(C5.shape)
        # print(C5.shape)
        C5 = self.cconv_5(C5)  # (1,128,20,24,20)
        flow = self.defconv5(C5)  # (1,3,20,24,20) #第一个变形场
        flow_all = self.diff[4](flow)  # 进行平滑
        # --------------------------------------------------------------------------------------
        # warp_seg = self.warp[4](mov_seg[0],flow_all)
        # print(F5.shape)
        # print(mov_seg[0].shape)
        # print(warp_seg.shape)
        # print(fix_seg[0].shape)
        # pre_sim = self.dice(warp_seg,fix_seg[0])
        warp_seg = self.warp[4](mov_seg[4], flow_all)
        # print(mov_seg[4].shape)
        pre_sim = self.dice(warp_seg, fix_seg[4])
        for ee in range(er):  # 32c
            # print(M5.shape)
            warped = self.warp[4](M5, flow_all)  # 复合变形场  32c
            # warped = warped * sig_diff_map5
            # C5 = self.dconv5(torch.cat([F5, warped, C5,sig_diff_map5], dim=1))  #32c
            C5 = self.dconv5(torch.cat([F5, warped, C5], dim=1))  # 32c

            v = self.defconv5(C5)  # (1,3,20,24,20)
            w = self.diff[4](v)
            flow_all = self.warp[4](flow_all, w) + w  # 复合

            warp_seg = self.warp[4](mov_seg[4],flow)  #可以在此处考虑是flowall参与迭代还是只有当前的folow参与迭代
            cur_sim = self.dice(warp_seg,fix_seg[4])
            if cur_sim > pre_sim:
                pre_sim = cur_sim
                flow_all = flow
            else:
                break
        # --------------------------------------------------------------------
        flow_all = self.upsample_trilin(2 * flow_all)
        warp_seg = self.warp[3](mov_seg[3], flow_all)
        # print(mov_seg[4].shape)
        pre_sim = self.dice(warp_seg, fix_seg[3])

        for aa in range(ar):  # 8c
            D4 = self.upconv4(C5)  # 16c
            warped = self.warp[3](M4, flow_all)  # 复合变形场
            # warped = warped * sig_diff_map4
            C4 = self.cconv_4(torch.cat([F4, warped, D4], dim=1))

            v = self.defconv4(C4)  # (1,3,20,24,20)  #为什么要输入三个图像呢
            w = self.diff[3](v)
            flow = self.warp[3](flow_all, w) + w

            warp_seg = self.warp[3](mov_seg[3], flow)  # 可以在此处考虑是flowall参与迭代还是只有当前的folow参与迭代
            cur_sim = self.dice(warp_seg, fix_seg[3])
            if cur_sim > pre_sim:
                pre_sim = cur_sim
                flow_all = flow
            else:
                flow_all = flow
                break
        # ---------------------------------------------------------------------------------------
        flow_all = self.upsample_trilin(2 * flow_all)
        warp_seg = self.warp[2](mov_seg[2], flow_all)
        pre_sim = self.dice(warp_seg, fix_seg[2])
        for bb in range(br):  # 4c
            D3 = self.upconv3(C4)  # (1, 64, 40, 48, 40) C是f，m，def的产物，用来生成单个尺度的最终变形场
            warped = self.warp[2](M3, flow_all)  # (1, 64, 40, 48, 40)
            # warped = warped * sig_diff_map3
            C3 = self.cconv_3(F3, warped, D3)  # (1, 3 * 64, 40, 48, 40)

            v = self.defconv3(C3)
            w = self.diff[2](v)
            flow = self.warp[2](flow_all, w) + w

            warp_seg = self.warp[2](mov_seg[2], flow)  # 可以在此处考虑是flowall参与迭代还是只有当前的folow参与迭代
            cur_sim = self.dice(warp_seg, fix_seg[2])
            if cur_sim > pre_sim:
                pre_sim = cur_sim
                flow_all = flow
            else:
                flow_all = flow
                break
        # -------------------------------------------------------------------------------------------
        flow_all = self.upsample_trilin(2 * flow_all)
        warp_seg = self.warp[1](mov_seg[1], flow_all)
        pre_sim = self.dice(warp_seg, fix_seg[1])

        for cc in range(cr):  # 2c
            D2 = self.upconv2(C3)
            warped = self.warp[1](M2, flow_all)
            # warped = warped * sig_diff_map2
            C2 = self.cconv_2(F2, warped, D2)

            v = self.defconv2(C2)  # (1,3,80,96,80)
            w = self.diff[1](v)
            flow = self.warp[1](flow_all, w) + w

            warp_seg = self.warp[1](mov_seg[1], flow)  # 可以在此处考虑是flowall参与迭代还是只有当前的folow参与迭代
            cur_sim = self.dice(warp_seg, fix_seg[1])
            if cur_sim > pre_sim:
                pre_sim = cur_sim
                flow_all = flow
            else:
                flow_all = flow
                break
        # =-----------------------------------------------------------------------------
        flow_all = self.upsample_trilin(2 * flow_all)
        warp_seg = self.warp[0](mov_seg[0], flow_all)
        pre_sim = self.dice(warp_seg, fix_seg[0])

        for dd in range(dr):  # 1c
            D1 = self.upconv1(C2)  # (1,16,160,196,160)
            warped = self.warp[0](M1, flow_all)  # （1,16,160,196,160)
            # warped = warped * sig_diff_map1
            # print(F1.shape)
            C1 = self.cconv_1(F1, warped, D1)  # （1,48,160,196,160)

            v = self.defconv1(C1)
            w = self.diff[0](v)
            flow = self.warp[0](flow_all, w) + w  # （1,3,160,196,160)

            warp_seg = self.warp[0](mov_seg[0], flow)  # 可以在此处考虑是flowall参与迭代还是只有当前的folow参与迭代
            cur_sim = self.dice(warp_seg, fix_seg[0])
            if cur_sim > pre_sim:
                pre_sim = cur_sim
                flow_all = flow
            else:
                flow_all = flow
                break

        y_moved = self.warp[0](moving, flow_all)
        # field = [flowx4,flowx3,flowx2,flowx1,flow_all]
        return y_moved, flow_all  # ,field #, diff_map1

class multi_scale_Reg_Decoder_nearest(nn.Module):
    def __init__(self, inshape=(160, 192, 160), channels=16):
        super(multi_scale_Reg_Decoder_nearest, self).__init__()
        self.channels = channels
        self.step = 7
        self.inshape = inshape

        c = self.channels
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear',
                                           align_corners=True)
        self.warp = nn.ModuleList()
        self.warp_label = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(5):
            self.warp.append(SpatialTransformer([s // 2 ** i for s in inshape]))
            self.warp_label.append(SpatialTransformer([s // 2 ** i for s in inshape],mode='nearest'))
            self.diff.append(VecInt([s // 2 ** i for s in inshape]))
        # bottleNeck
        self.cconv_5 = nn.Sequential(
            ConvInsBlock(16 * 2 * c, 16 * c, 3, 1),
        )
        # warp scale 2
        self.defconv5 = nn.Conv3d(16 * c, 3, 3, 1, 1)
        self.defconv5.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv5.weight.shape))
        self.defconv5.bias = nn.Parameter(torch.zeros(self.defconv5.bias.shape))

        self.dconv5 = nn.Sequential(
            ConvInsBlock(16 * 3 * c, 16 * c),
        )

        self.upconv4 = UpConvBlock(16 * c, 8 * c, 4, 2)

        self.cconv_4 = nn.Sequential(
            ConvInsBlock(8 * 3 * c, 8 * c, 3, 1),

        )
        self.defconv4 = nn.Conv3d(8 * c, 3, 3, 1, 1)
        self.defconv4.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv4.weight.shape))
        self.defconv4.bias = nn.Parameter(torch.zeros(self.defconv4.bias.shape))

        self.upconv3 = UpConvBlock(8 * c, 4 * c, 4, 2)
        self.cconv_3 = CConv(4 * 3 * c, 4 * c)

        self.defconv3 = nn.Conv3d(4 * c, 3, 3, 1, 1)
        self.defconv3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv3.weight.shape))
        self.defconv3.bias = nn.Parameter(torch.zeros(self.defconv3.bias.shape))

        self.upconv2 = UpConvBlock(4 * c, 2 * c, 4, 2)
        self.cconv_2 = CConv(2 * 3 * c, 2 * c)

        self.defconv2 = nn.Conv3d(2 * c, 3, 3, 1, 1)
        self.defconv2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv2.weight.shape))
        self.defconv2.bias = nn.Parameter(torch.zeros(self.defconv2.bias.shape))

        self.upconv1 = UpConvBlock(2 * c, c, 4, 2)
        self.cconv_1 = CConv(3 * c, c)

        self.defconv1 = nn.Conv3d(c, 3, 3, 1, 1)
        self.defconv1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv1.weight.shape))
        self.defconv1.bias = nn.Parameter(torch.zeros(self.defconv1.bias.shape))

        # self.dconv1 = ConvInsBlock(3 * c, c)
        # self.compdiff = self.comput_err
        # self.dice = self.dice

    def dice(self,outputs, labels, num_classes=5):
        # print(outputs.shape)
        """Calculate Dice score for each class"""
        # outputs = torch.argmax(outputs, dim=1)
        # print(outputs.shape)
        # labels = torch.argmax(labels, dim=1)
        dice_scores = []

        for cls in range(1, num_classes):  # Start from 1 to skip background
            outputs_cls = (outputs == cls).float()
            # labels_cls = (labels.squeeze(1) == cls).float()
            labels_cls = (labels == cls).float()
            # print(outputs_cls.shape)
            intersection = (outputs_cls * labels_cls).sum()
            union = outputs_cls.sum() + labels_cls.sum()

            if union > 0:
                dice = (2. * intersection) / (union + 1e-8)
                dice_scores.append(dice.item())

        return np.mean(dice_scores) if dice_scores else 0.0

    def forward(self, moving,mov_seg,fix_seg, FM, FF):  # moving,fixed,
        # encode stage
        M1, M2, M3, M4, M5 = FM[0], FM[1], FM[2], FM[3], FM[4]
        F1, F2, F3, F4, F5 = FF[0], FF[1], FF[2], FF[3], FF[4]
        # c=8, 2c, 4c, 8c,16c,32c  # 160, 80, 40, 20, 5
        er = 5
        ar = 5
        br = 5
        cr = 5
        dr = 5
        # first dec layer

        C5 = torch.cat([F5, M5], dim=1)  # 16c*2=32
        C5 = self.cconv_5(C5)  # (1,128,20,24,20)
        flow = self.defconv5(C5)  # (1,3,20,24,20) #第一个变形场
        flow_all = self.diff[4](flow)  # 进行平滑
        # --------------------------------------------------------------------------------------
        flowx5 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)(16 * flow_all)
        # warp_seg = self.warp[4](mov_seg[0],flow_all)
        # pre_sim = self.dice(warp_seg,fix_seg[0])
        warp_seg = self.warp_label[0](mov_seg, flowx5)
        pre_sim = self.dice(warp_seg, fix_seg)
        for ee in range(er):  # 32c
            # print(M5.shape)
            warped = self.warp[4](M5, flow_all)  # 复合变形场  32c
            # warped = warped * sig_diff_map5
            # C5 = self.dconv5(torch.cat([F5, warped, C5,sig_diff_map5], dim=1))  #32c
            C5 = self.dconv5(torch.cat([F5, warped, C5], dim=1))  # 32c

            v = self.defconv5(C5)  # (1,3,20,24,20)
            w = self.diff[4](v)
            flow_all = self.warp[4](flow_all, w) + w  # 复合

            flowx5 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)(16 * flow)
            warp_seg = self.warp_label[0](mov_seg,flowx5)  #可以在此处考虑是flowall参与迭代还是只有当前的folow参与迭代
            cur_sim = self.dice(warp_seg,fix_seg)
            if cur_sim > pre_sim:
                pre_sim = cur_sim
                flow_all = flow
            else:
                break
        # --------------------------------------------------------------------
        flowx4 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)(16 * flow_all)
        warp_seg = self.warp_label[0](mov_seg, flowx4)
        pre_sim = self.dice(warp_seg, fix_seg)

        flow_all = self.upsample_trilin(2 * flow_all)
        for aa in range(ar):  # 8c
            D4 = self.upconv4(C5)  # 16c
            warped = self.warp[3](M4, flow_all)  # 复合变形场
            # warped = warped * sig_diff_map4
            C4 = self.cconv_4(torch.cat([F4, warped, D4], dim=1))

            v = self.defconv4(C4)  # (1,3,20,24,20)  #为什么要输入三个图像呢
            w = self.diff[3](v)
            flow = self.warp[3](flow_all, w) + w

            flowx4 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)(8 * flow)
            warp_seg = self.warp_label[0](mov_seg, flowx4)  # 可以在此处考虑是flowall参与迭代还是只有当前的folow参与迭代
            cur_sim = self.dice(warp_seg, fix_seg)
            if cur_sim > pre_sim:
                pre_sim = cur_sim
                flow_all = flow
            else:
                flow_all = flow
                break
        # ---------------------------------------------------------------------------------------
        flowx3 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)(8 * flow_all)
        flow_all = self.upsample_trilin(2 * flow_all)

        warp_seg = self.warp_label[0](mov_seg, flowx3)
        pre_sim = self.dice(warp_seg, fix_seg)
        for bb in range(br):  # 4c
            D3 = self.upconv3(C4)  # (1, 64, 40, 48, 40) C是f，m，def的产物，用来生成单个尺度的最终变形场
            warped = self.warp[2](M3, flow_all)  # (1, 64, 40, 48, 40)
            # warped = warped * sig_diff_map3
            C3 = self.cconv_3(F3, warped, D3)  # (1, 3 * 64, 40, 48, 40)

            v = self.defconv3(C3)
            w = self.diff[2](v)
            flow = self.warp[2](flow_all, w) + w

            flowx3 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)(4 * flow_all)
            warp_seg = self.warp_label[0](mov_seg, flowx3)  # 可以在此处考虑是flowall参与迭代还是只有当前的folow参与迭代
            cur_sim = self.dice(warp_seg, fix_seg)
            if cur_sim > pre_sim:
                pre_sim = cur_sim
                flow_all = flow
            else:
                flow_all = flow
                break
        # -------------------------------------------------------------------------------------------
        flowx2 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)(4 * flow_all)
        flow_all = self.upsample_trilin(2 * flow_all)
        warp_seg = self.warp_label[0](mov_seg, flowx2)
        pre_sim = self.dice(warp_seg, fix_seg)

        for cc in range(cr):  # 2c
            D2 = self.upconv2(C3)
            warped = self.warp[1](M2, flow_all)
            # warped = warped * sig_diff_map2
            C2 = self.cconv_2(F2, warped, D2)

            v = self.defconv2(C2)  # (1,3,80,96,80)
            w = self.diff[1](v)
            flow = self.warp[1](flow_all, w) + w

            flowx2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)(2 * flow)
            warp_seg = self.warp_label[0](mov_seg, flowx2)  # 可以在此处考虑是flowall参与迭代还是只有当前的folow参与迭代
            cur_sim = self.dice(warp_seg, fix_seg)
            if cur_sim > pre_sim:
                pre_sim = cur_sim
                flow_all = flow
            else:
                flow_all = flow
                break
        # =-----------------------------------------------------------------------------
        flowx1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)(2 * flow_all)
        flow_all = self.upsample_trilin(2 * flow_all)

        warp_seg = self.warp_label[0](mov_seg, flowx1)
        pre_sim = self.dice(warp_seg, fix_seg)

        for dd in range(dr):  # 1c
            D1 = self.upconv1(C2)  # (1,16,160,196,160)
            warped = self.warp[0](M1, flow_all)  # （1,16,160,196,160)
            # warped = warped * sig_diff_map1
            # print(F1.shape)
            C1 = self.cconv_1(F1, warped, D1)  # （1,48,160,196,160)

            v = self.defconv1(C1)
            w = self.diff[0](v)
            flow = self.warp[0](flow_all, w) + w  # （1,3,160,196,160)

            warp_seg = self.warp_label[0](mov_seg, flow)  # 可以在此处考虑是flowall参与迭代还是只有当前的folow参与迭代
            cur_sim = self.dice(warp_seg, fix_seg)
            if cur_sim > pre_sim:
                pre_sim = cur_sim
                flow_all = flow
            else:
                flow_all = flow
                break

        y_moved = self.warp[0](moving, flow_all)
        # field = [flowx4,flowx3,flowx2,flowx1,flow_all]
        return y_moved, flow_all  # ,field #, diff_map1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Reg_Decoder(inshape=(160, 192, 160), channels=16).to(device)

    # 构造 dummy 数据
    moving = torch.randn(1, 1, 160, 192, 160).to(device)
    fixed  = torch.randn(1, 1, 160, 192, 160).to(device)

    # 构造 Encoder 输出（与原 Encoder 保持一致）
    c = 16
    f0 = torch.randn(1, 1*c, 160, 192, 160)
    f1 = torch.randn(1, 2*c,  80,  96,  80)
    f2 = torch.randn(1, 4*c,  40,  48,  40)
    f3 = torch.randn(1, 8*c, 20,  24,  20)
    f4 = torch.randn(1, 16*c, 10,  12,  10)
    feats_m = tuple(f.to(device) for f in (f0, f1, f2, f3, f4))
    feats_f = tuple(f.to(device) for f in (f0, f1, f2, f3, f4))

    with torch.no_grad():
        moved, final_flow= model(moving, fixed, feats_m, feats_f)

    print("✅ 前向成功！")
    print(f"moving:   {tuple(moving.shape)}")
    print(f"fixed:    {tuple(fixed.shape)}")
    print(f"moved:    {tuple(moved.shape)}")
    print(f"flow:     {tuple(final_flow.shape)}")
