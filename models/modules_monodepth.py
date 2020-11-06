## The souce code is from: https://github.com/OniroAI/MonoDepth-PyTorch
from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tf
from .modules_camconv import CamConvModule
import torchsnooper

class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1)/2))
        p2d = (p, p, p, p)
        x = self.conv_base(tf.pad(x, p2d))
        x = self.normalize(x)
        return tf.elu(x, inplace=True)

### relu version, for pose decoder(same as in monodepth2)
class conv_relu(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv_relu, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        # self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1)/2))
        p2d = (p, p, p, p)
        x = self.conv_base(tf.pad(x, p2d))
        # x = self.normalize(x)
        return tf.relu(x, inplace=True)

class convblock(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size):
        super(convblock, self).__init__()
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, kernel_size, 2)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


class maxpool(nn.Module):
    def __init__(self, kernel_size):
        super(maxpool, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1) / 2))
        p2d = (p, p, p, p)
        return tf.max_pool2d(tf.pad(x, p2d), self.kernel_size, stride=2)


class resconv_basic(nn.Module):
    # for resnet18
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv_basic, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 3, stride)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, 1)
        self.conv3 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        if do_proj:
            shortcut = self.conv3(x)
        else:
            shortcut = x
        return tf.elu(self.normalize(x_out + shortcut), inplace=True)


def resblock_basic(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv_basic(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks):
        layers.append(resconv_basic(num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv1(x)


class get_disp_1ch(nn.Module):
    def __init__(self, num_in_layers):
        super(get_disp_1ch, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, 1, kernel_size=3, stride=1)
        self.normalize = nn.BatchNorm2d(1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(tf.pad(x, p2d))
        x = self.normalize(x)
        return 0.3 * self.sigmoid(x)

### multi predictor (disp and 3d flow)
class get_disp_3dflow(nn.Module):
    def __init__(self, num_in_layers):
        super(get_disp_3dflow, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, 1, kernel_size=3, stride=1)
        # self.normalize_disp = nn.BatchNorm2d(1)
        self.sigmoid = torch.nn.Sigmoid()

        self.conv3 = nn.Conv2d(num_in_layers, 3, kernel_size=3, stride=1)
        # self.normalize_flow = nn.BatchNorm2d(3)

        self.conv_mask = nn.Conv2d(num_in_layers, 1, kernel_size=3, stride=1)

    def forward(self, x_in):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(tf.pad(x_in, p2d))
        disp = 0.3 * self.sigmoid(x)

        flow = self.conv3(tf.pad(x_in, p2d))
        mask = self.conv_mask(tf.pad(x_in, p2d))
        mask = self.sigmoid(mask)

        return x, disp, flow, mask

### shared predictor for all levels
class Resnet18_AllinOne(nn.Module):
    def __init__(self):
        super(Resnet18_AllinOne, self).__init__()

        # encoder
        self.conv1 = conv(32, 64, 7, 2)  # H/2  -   64D
        self.pool1 = maxpool(3)  # H/4  -   64D
        self.conv2 = resblock_basic(64, 64, 2, 2)  # H/8  -  64D
        self.conv3 = resblock_basic(64, 128, 2, 2)  # H/16 -  128D
        self.conv4 = resblock_basic(128, 256, 2, 2)  # H/32 - 256D
        # self.conv5 = resblock_basic(256, 512, 2, 2)  # H/64 - 512D

        # decoder
        self.upconv4 = upconv(256, 256, 3, 2)
        self.iconv4 = conv(128+256, 256, 3, 1)

        self.upconv3 = upconv(256, 128, 3, 2)
        self.iconv3 = conv(64+128, 128, 3, 1)
        self.disp3_layer = get_disp_3dflow(128)

        self.upconv2 = upconv(128, 64, 3, 2)
        self.iconv2 = conv(64+64 + 1 + 3+1, 64, 3, 1)
        self.disp2_layer = get_disp_3dflow(64)

        self.upconv1 = upconv(64, 32, 3, 2)
        self.iconv1 = conv(32+1+3+1, 32, 3, 1)
        self.disp1_layer = get_disp_3dflow(32)

        ### pose decoder
        # self.pose_convs[("squeeze")] = nn.Conv2d(512, 512, 1)
        # self.pose_convs[("pose", 0)] = nn.Conv2d(512, 256, 3, 1, 1)
        # self.pose_convs[("pose", 1)] = nn.Conv2d(256, 256, 3, 1, 1)
        # self.pose_convs[("pose", 2)] = nn.Conv2d(256, 6, 1)
        # self.relu = nn.ReLU()
        ### option 1: equivalent to above, the same as in monodepth2
        self.pose_convs_squeeze = conv_relu(256, 256, 1, 1)
        self.pose_convs_0 = conv_relu(256, 128, 3, 1)
        self.pose_convs_1 = conv_relu(128, 128, 3, 1)
        self.pose_convs_2 = nn.Conv2d(128, 6, 1)

        # ### option 2: use elu and batchnorm, to be consistent with encoder and decoder
        # self.pose_convs[("squeeze")] = conv(512, 512, 1, 1)
        # self.pose_convs[("pose", 0)] = conv(512, 256, 3, 1)
        # self.pose_convs[("pose", 1)] = conv(256, 256, 3, 1)
        # self.pose_convs[("pose", 2)] = nn.Conv2d(256, 6, 1)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_uniform_(m.weight)

    # @torchsnooper.snoop()
    def forward(self, x):
        # # encoder
        # x1 = self.conv1(x)
        # x_pool1 = self.pool1(x1)
        # x2 = self.conv2(x_pool1)
        # x3 = self.conv3(x2)
        # x4 = self.conv4(x3)
        # # x5 = self.conv5(x4)

        # # skips
        # skip1 = x1
        # skip2 = x_pool1
        # skip3 = x2
        # skip4 = x3
        # # skip5 = x4

        # encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        # skips
        skip1 = x1
        skip2 = x2
        skip3 = x3
        # skip5 = x4

        # decoder
        # # upconv6 = self.upconv6(x5)
        # # concat6 = torch.cat((upconv6, skip5), 1)
        # # iconv6 = self.iconv6(concat6)

        # # upconv5 = self.upconv5(iconv6)
        # upconv5 = self.upconv5(x4)
        # concat5 = torch.cat((upconv5, skip4), 1)
        # iconv5 = self.iconv5(concat5)

        # upconv4 = self.upconv4(iconv5)
        upconv4 = self.upconv4(x4)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3, disp3_raw, self.flow3, mask3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)
        self.uflow3 = nn.functional.interpolate(self.flow3, scale_factor=2, mode='bilinear', align_corners=True)
        umask3 = nn.functional.interpolate(mask3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3, self.uflow3, umask3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2, disp2_raw, self.flow2, mask2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)
        self.uflow2 = nn.functional.interpolate(self.flow2, scale_factor=2, mode='bilinear', align_corners=True)
        umask2 = nn.functional.interpolate(mask2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2, self.uflow2, umask2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1, disp1_raw, self.flow1, mask1 = self.disp1_layer(iconv1)

        flow_dicts = {}
        flow_dicts["1"] = self.flow1
        flow_dicts["2"] = self.flow2
        flow_dicts["3"] = self.flow3
        # flow_dicts["4"] = self.flow4
        
        disp_dicts = {}
        disp_dicts["1"] = self.disp1
        disp_dicts["2"] = self.disp2
        disp_dicts["3"] = self.disp3
        # disp_dicts["4"] = self.disp4

        mask_dicts = {}
        mask_dicts["1"] = mask1
        mask_dicts["2"] = mask2
        mask_dicts["3"] = mask3
        # mask_dicts["4"] = mask4

        ### pose decoder
        # out = self.relu(self.pose_convs["squeeze"](x5))
        # for i in range(3):
        #     out = self.convs[("pose", i)](out)
        #     if i != 2:
        #         out = self.relu(out)

        # out = self.pose_convs_squeeze(x5)
        out = self.pose_convs_squeeze(x4)
        out = self.pose_convs_0(out)
        out = self.pose_convs_1(out)
        out = self.pose_convs_2(out)

        out = out.mean(3).mean(2)
        pose = 0.01 * out.view(-1, 6)

        # return disp_dicts, flow_dicts, mask_dicts, pose, iconv1
        return iconv1, flow_dicts["1"], disp_dicts["1"], pose, mask_dicts["1"]


### shared predictor for all levels
class Resnet18_Pose(nn.Module):
    def __init__(self):
        super(Resnet18_Pose, self).__init__()

        # encoder
        self.conv1 = conv(6, 64, 7, 2)  # H/2  -   64D
        self.pool1 = maxpool(3)  # H/4  -   64D
        self.conv2 = resblock_basic(64, 64, 2, 2)  # H/8  -  64D
        self.conv3 = resblock_basic(64, 128, 2, 2)  # H/16 -  128D
        self.conv4 = resblock_basic(128, 256, 2, 2)  # H/32 - 256D
        self.conv5 = resblock_basic(256, 512, 2, 2)  # H/64 - 512D

        ### pose decoder
        # self.pose_convs[("squeeze")] = nn.Conv2d(512, 512, 1)
        # self.pose_convs[("pose", 0)] = nn.Conv2d(512, 256, 3, 1, 1)
        # self.pose_convs[("pose", 1)] = nn.Conv2d(256, 256, 3, 1, 1)
        # self.pose_convs[("pose", 2)] = nn.Conv2d(256, 6, 1)
        # self.relu = nn.ReLU()
        ### option 1: equivalent to above, the same as in monodepth2
        self.pose_convs_squeeze = conv_relu(512, 512, 1, 1)
        self.pose_convs_0 = conv_relu(512, 256, 3, 1)
        self.pose_convs_1 = conv_relu(256, 256, 3, 1)
        self.pose_convs_2 = nn.Conv2d(256, 6, 1)

        # ### option 2: use elu and batchnorm, to be consistent with encoder and decoder
        # self.pose_convs[("squeeze")] = conv(512, 512, 1, 1)
        # self.pose_convs[("pose", 0)] = conv(512, 256, 3, 1)
        # self.pose_convs[("pose", 1)] = conv(256, 256, 3, 1)
        # self.pose_convs[("pose", 2)] = nn.Conv2d(256, 6, 1)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_uniform_(m.weight)

    # @torchsnooper.snoop()
    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x_pool1 = self.pool1(x1)
        x2 = self.conv2(x_pool1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        ### pose decoder
        # out = self.relu(self.pose_convs["squeeze"](x5))
        # for i in range(3):
        #     out = self.convs[("pose", i)](out)
        #     if i != 2:
        #         out = self.relu(out)

        out = self.pose_convs_squeeze(x5)
        # out = self.pose_convs_squeeze(x4)
        out = self.pose_convs_0(out)
        out = self.pose_convs_1(out)
        out = self.pose_convs_2(out)

        out = out.mean(3).mean(2)
        pose = 0.01 * out.view(-1, 6)

        return pose


class Resnet18_MonoDepth_Single(nn.Module):
    def __init__(self):
        super(Resnet18_MonoDepth_Single, self).__init__()
        # encoder
        self.conv1 = conv(3, 64, 7, 2)  # H/2  -   64D
        self.pool1 = maxpool(3)  # H/4  -   64D
        self.conv2 = resblock_basic(64, 64, 2, 2)  # H/8  -  64D
        self.conv3 = resblock_basic(64, 128, 2, 2)  # H/16 -  128D
        self.conv4 = resblock_basic(128, 256, 2, 2)  # H/32 - 256D
        self.conv5 = resblock_basic(256, 512, 2, 2)  # H/64 - 512D

        # decoder
        self.upconv6 = upconv(512, 512, 3, 2)
        self.iconv6 = conv(256+512, 512, 3, 1)

        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(128+256, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(64+128, 128, 3, 1)
        self.disp4_layer = get_disp_1ch(128)

        self.upconv3 = upconv(128, 64, 3, 2)
        self.iconv3 = conv(64+64 + 1, 64, 3, 1)
        self.disp3_layer = get_disp_1ch(64)

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(64+32 + 1, 32, 3, 1)
        self.disp2_layer = get_disp_1ch(32)

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16+1, 16, 3, 1)
        self.disp1_layer = get_disp_1ch(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x_pool1 = self.pool1(x1)
        x2 = self.conv2(x_pool1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # skips
        skip1 = x1
        skip2 = x_pool1
        skip3 = x2
        skip4 = x3
        skip5 = x4

        # decoder
        upconv6 = self.upconv6(x5)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=2, mode='bilinear', align_corners=True)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)
        return self.disp1, self.disp2, self.disp3, self.disp4


class Resnet18_MonoDepth_Single_CamConv(nn.Module):
    def __init__(self):
        super(Resnet18_MonoDepth_Single_CamConv, self).__init__()
        # encoder
        self.conv1 = conv(3, 64, 7, 2)  # H/2  -   64D
        self.pool1 = maxpool(3)  # H/4  -   64D
        self.conv2 = resblock_basic(64, 64, 2, 2)  # H/8  -  64D
        self.conv3 = resblock_basic(64, 128, 2, 2)  # H/16 -  128D
        self.conv4 = resblock_basic(128, 256, 2, 2)  # H/32 - 256D
        self.conv5 = resblock_basic(256, 512, 2, 2)  # H/64 - 512D

        # decoder
        self.upconv6 = upconv(512 + 6, 512, 3, 2)
        self.iconv6 = conv(256+512 + 6, 512, 3, 1)

        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(128+256 + 6, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(64+128 + 6, 128, 3, 1)
        self.disp4_layer = get_disp_1ch(128)

        self.upconv3 = upconv(128, 64, 3, 2)
        self.iconv3 = conv(64+64 + 1 + 6, 64, 3, 1)
        self.disp3_layer = get_disp_1ch(64)

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(64+32 + 1 + 6, 32, 3, 1)
        self.disp2_layer = get_disp_1ch(32)

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16+1, 16, 3, 1)
        self.disp1_layer = get_disp_1ch(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

        self.camconv = CamConvModule()

    def forward(self, x, intrinsic):

        # encoder
        x1 = self.conv1(x)
        x_pool1 = self.pool1(x1)
        x2 = self.conv2(x_pool1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # skips
        skip1 = self.camconv(x1, x, intrinsic)
        skip2 = self.camconv(x_pool1)
        skip3 = self.camconv(x2)
        skip4 = self.camconv(x3)
        skip5 = self.camconv(x4)

        # decoder
        upconv6 = self.upconv6(self.camconv(x5))
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=2, mode='bilinear', align_corners=True)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)

        return self.disp1, self.disp2, self.disp3, self.disp4