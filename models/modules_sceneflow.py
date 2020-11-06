from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging 

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_ms, pts2pixel
from utils.sceneflow_util import pixel2pts_ms_and_depth, pixel2pts_ms_from_depth, pixel2pts_ms_and_uv1grid, disp2depth_kitti, depth2disp_kitti

from utils.interpolation import interpolate2d_as_enlarge_only

def get_grid(x):
    grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([grid_H, grid_V], 1)
    grids_cuda = grid.float().requires_grad_(False).cuda()
    return grids_cuda


class WarpingLayer_Flow(nn.Module):
    def __init__(self):
        super(WarpingLayer_Flow, self).__init__()

        torch_vs = torch.__version__
        digits = torch_vs.split(".")
        torch_vs_n = float(digits[0]) + float(digits[1]) * 0.1
        self.grid_sample_specify_align_flag = torch_vs_n >= 1.3  

    def forward(self, x, flow):
        flo_list = []
        flo_w = flow[:, 0] * 2 / max(x.size(3) - 1, 1)
        flo_h = flow[:, 1] * 2 / max(x.size(2) - 1, 1)
        flo_list.append(flo_w)
        flo_list.append(flo_h)
        flow_for_grid = torch.stack(flo_list).transpose(0, 1)
        grid = torch.add(get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)       
        if self.grid_sample_specify_align_flag:
            x_warp = tf.grid_sample(x, grid, align_corners=True)
        else:
            x_warp = tf.grid_sample(x, grid)

        mask = torch.ones(x.size(), requires_grad=False).cuda()
        if self.grid_sample_specify_align_flag:
            mask = tf.grid_sample(mask, grid, align_corners=True)
        else:
            mask = tf.grid_sample(mask, grid)

        # ### Original
        # mask = (mask >= 1.0).float()

        ### from PWC-Net
        # mask[mask<0.9999] = 0
        # mask[mask>0] = 1
        mask = (mask >= 0.9999).float() # changed Nov 1

        return x_warp * mask

class WarpingLayer_FlowNormalized(nn.Module):
    def __init__(self):
        super(WarpingLayer_FlowNormalized, self).__init__()

        torch_vs = torch.__version__
        digits = torch_vs.split(".")
        torch_vs_n = float(digits[0]) + float(digits[1]) * 0.1
        self.grid_sample_specify_align_flag = torch_vs_n >= 1.3  

    def forward(self, x, flow):

        _, _, h_x, w_x = x.size()
        assert flow.shape[2] == h_x, "{} {}".format(x.shape, flow.shape)
        assert flow.shape[3] == w_x, "{} {}".format(x.shape, flow.shape)
        flow = flow * w_x
        
        flo_list = []
        flo_w = flow[:, 0] * 2 / max(x.size(3) - 1, 1)
        flo_h = flow[:, 1] * 2 / max(x.size(2) - 1, 1)
        flo_list.append(flo_w)
        flo_list.append(flo_h)
        flow_for_grid = torch.stack(flo_list).transpose(0, 1)
        grid = torch.add(get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)       
        if self.grid_sample_specify_align_flag:
            x_warp = tf.grid_sample(x, grid, align_corners=True)
        else:
            x_warp = tf.grid_sample(x, grid)

        mask = torch.ones(x.size(), requires_grad=False).cuda()
        if self.grid_sample_specify_align_flag:
            mask = tf.grid_sample(mask, grid, align_corners=True)
        else:
            mask = tf.grid_sample(mask, grid)

        # ### Original
        # mask = (mask >= 1.0).float()

        ### from PWC-Net
        # mask[mask<0.9999] = 0
        # mask[mask>0] = 1
        mask = (mask >= 0.9999).float() # changed Nov 1

        return x_warp * mask


class WarpingLayer_SF(nn.Module):
    def __init__(self, reg_depth):
        super(WarpingLayer_SF, self).__init__()
 
        torch_vs = torch.__version__
        digits = torch_vs.split(".")
        torch_vs_n = float(digits[0]) + float(digits[1]) * 0.1
        self.grid_sample_specify_align_flag = torch_vs_n >= 1.3  

        self.reg_depth = reg_depth

    def forward(self, x, sceneflow, disp, k1, input_size):

        _, _, h_x, w_x = x.size()
        local_scale = torch.zeros_like(input_size)
        local_scale[:, 0] = h_x
        local_scale[:, 1] = w_x

        ### adapt to depth input
        if self.reg_depth:
            depth = disp
            pts1, k1_scale = pixel2pts_ms_from_depth(k1, depth, local_scale / input_size)
            _, _, coord1 = pts2pixel_ms(k1_scale, pts1, sceneflow, [h_x, w_x])

        else:
            disp = interpolate2d_as(disp, x) * w_x
            pts1, k1_scale = pixel2pts_ms(k1, disp, local_scale / input_size)
            _, _, coord1 = pts2pixel_ms(k1_scale, pts1, sceneflow, [h_x, w_x])

        grid = coord1.transpose(1, 2).transpose(2, 3)
        if self.grid_sample_specify_align_flag:
            x_warp = tf.grid_sample(x, grid, align_corners=True)
        else:
            x_warp = tf.grid_sample(x, grid)

        mask = torch.ones_like(x, requires_grad=False)
        if self.grid_sample_specify_align_flag:
            mask = tf.grid_sample(mask, grid, align_corners=True)
        else:
            mask = tf.grid_sample(mask, grid)

        # ### Original
        # mask = (mask >= 1.0).float()

        ### from PWC-Net
        # mask[mask<0.9999] = 0
        # mask[mask>0] = 1
        mask = (mask >= 0.9999).float() # changed Nov 1

        return x_warp * mask


def initialize_msra(modules):
    logging.info("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.LeakyReLU):
            pass

        elif isinstance(layer, nn.Sequential):
            pass


def upsample_outputs_as(input_list, ref_list):
    output_list = []
    for ii in range(0, len(input_list)):
        output_list.append(interpolate2d_as(input_list[ii], ref_list[ii]))

    return output_list

### add an enlarge_only mode
def upsample_outputs_as_enlarge_only(input_list, ref_list):
    output_list = []
    for ii in range(0, len(input_list)):
        output_list.append(interpolate2d_as_enlarge_only(input_list[ii], ref_list[ii]))

    return output_list

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
        return self.conv1(x)


class FeatureExtractor(nn.Module):
    def __init__(self, num_chs):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()

        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.Sequential(
                conv(ch_in, ch_out, stride=2),
                conv(ch_out, ch_out)
            )
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class MonoSceneFlowDecoder(nn.Module):
    def __init__(self, ch_in):
        super(MonoSceneFlowDecoder, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128),
            conv(128, 128),
            conv(128, 96),
            conv(96, 64),
            conv(64, 32)
        )
        self.conv_sf = conv(32, 3, isReLU=False)
        self.conv_d1 = conv(32, 1, isReLU=False)

    def forward(self, x):
        x_out = self.convs(x)
        sf = self.conv_sf(x_out)
        disp1 = self.conv_d1(x_out)

        return x_out, sf, disp1, None, None

class MonoSceneFlowUVDDecoder(nn.Module):
    def __init__(self, ch_in):
        super(MonoSceneFlowUVDDecoder, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128),
            conv(128, 128),
            conv(128, 96),
            conv(96, 64),
            conv(64, 32)
        )
        self.conv_sf = conv(32, 3, isReLU=False)
        self.conv_d1 = conv(32, 1, isReLU=False)
        self.sf_nl = torch.nn.Sigmoid()

    def forward(self, x):
        x_out = self.convs(x)
        sf = self.conv_sf(x_out)
        disp1 = self.conv_d1(x_out)

        sf = self.sf_nl(sf) * 0.3
        sf = 2 * sf - 0.3

        return x_out, sf, disp1, None, None

class MonoSceneFlowMaskDecoder(nn.Module):
    def __init__(self, ch_in):
        super(MonoSceneFlowMaskDecoder, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128),
            conv(128, 128),
            conv(128, 96),
            conv(96, 64),
            conv(64, 32)
        )
        self.conv_sf = conv(32, 3, isReLU=False)
        self.conv_d1 = conv(32, 1, isReLU=False)
        self.conv_mask = conv(32, 1, isReLU=False)

    def forward(self, x):
        x_out = self.convs(x)
        sf = self.conv_sf(x_out)
        disp1 = self.conv_d1(x_out)

        mask = self.conv_mask(x_out)
        mask = torch.sigmoid(mask)

        return x_out, sf, disp1, None, mask

class PoseHead(nn.Module):
    def __init__(self, separable):
        super(PoseHead, self).__init__()
        if separable:
            self.conv_pose = nn.Sequential(
                SpatialPyramidPooling([1, 2, 4, 8]),  # B*C(32)*N(43)
                nn.Linear(85, 16),                      # B*N(32)*C(16)
                # nn.Linear(43, 16),                      # B*N(32)*C(16)
                nn.ReLU(), 
                Transpose(-1, -2),                      # B*C(16)*N(32)
                nn.Linear(32, 16),                      # B*C(16)*N(16)
                nn.ReLU(), 
                Flatten(),                              # B*N(256)
                nn.Linear(256, 6)                       # B*N(6)
            )
        else:
            self.conv_pose = nn.Sequential(
                SpatialPyramidPooling([1, 2, 4, 8]),
                Flatten(),
                nn.Linear(85*32, 16), 
                nn.ReLU(), 
                nn.Linear(16, 6), 
            )
    def forward(self, x):
        pose = self.conv_pose(x)
        return pose
        

class MonoSceneFlowPoseDecoder(nn.Module):
    def __init__(self, ch_in, separable, conv_pose=None):
        super(MonoSceneFlowPoseDecoder, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128),
            conv(128, 128),
            conv(128, 96),
            conv(96, 64),
            conv(64, 32)
        )
        self.conv_sf = conv(32, 3, isReLU=False)
        self.conv_d1 = conv(32, 1, isReLU=False)

        self.conv_mask = conv(32, 1, isReLU=False)

        ### separable
        # self.conv_pose = nn.Sequential(
        #     SpatialPyramidPooling([1, 2, 4, 8]),  # B*C(32)*N(43)
        #     Transpose(-1, -2),                      # B*N(43)*C(32)
        #     nn.Linear(32, 16),                      # B*N(43)*C(16)
        #     nn.BatchNorm1d(43), 
        #     nn.ReLU(), 
        #     Transpose(-1, -2),                      # B*C(16)*N(43)
        #     nn.Linear(43, 16),                      # B*C(16)*N(16)
        #     nn.BatchNorm1d(16), 
        #     nn.ReLU(), 
        #     Flatten(),                              # B*N(256)
        #     nn.Linear(256, 6)                       # B*N(6)
        # )
        if conv_pose is None:
            if separable:
                self.conv_pose = nn.Sequential(
                    SpatialPyramidPooling([1, 2, 4, 8]),  # B*C(32)*N(43)
                    nn.Linear(85, 16),                      # B*N(32)*C(16)
                    # nn.Linear(43, 16),                      # B*N(32)*C(16)
                    nn.ReLU(), 
                    Transpose(-1, -2),                      # B*C(16)*N(32)
                    nn.Linear(32, 16),                      # B*C(16)*N(16)
                    nn.ReLU(), 
                    Flatten(),                              # B*N(256)
                    nn.Linear(256, 6)                       # B*N(6)
                )
            else:
                self.conv_pose = nn.Sequential(
                    SpatialPyramidPooling([1, 2, 4, 8]),
                    Flatten(),
                    nn.Linear(85*32, 16), 
                    nn.ReLU(), 
                    nn.Linear(16, 6), 
                )
        else:
            self.conv_pose = conv_pose

    def forward(self, x):
        x_out = self.convs(x)
        sf = self.conv_sf(x_out)
        disp1 = self.conv_d1(x_out)

        pose = self.conv_pose(x_out)
        # pose = pose * 0.1
        # pose[:, :3] = 0.1 * pose[:, :3]

        mask = self.conv_mask(x_out)
        mask = torch.sigmoid(mask)
            
        return x_out, sf, disp1, pose, mask

class MonoSceneFlowPoseHourGlass(nn.Module):
    # def __init__(self, ch_in, hourglass):
    def __init__(self, ch_in):
        super(MonoSceneFlowPoseHourGlass, self).__init__()
        self.conv = conv(ch_in, 32, 1, 1, 1)
        # self.hourglass = hourglass

    def forward(self, x):
        x = self.conv(x)
        return x
        # # x = interpolate2d_as(x, x_full)
        # disp_dicts, flow_dicts, mask_dicts, pose, iconv1 = self.hourglass(x)
        # return iconv1, flow_dicts["1"], disp_dicts["1"], pose, mask_dicts["1"]


class RigidFlowFromPose(nn.Module):
    def __init__(self, reg_depth):
        super(RigidFlowFromPose, self).__init__()

        self.reg_depth = reg_depth

    def forward(self, pose, disp, k_aug, aug_size):

        ### scale disp to pixel unit
        _, _, h_dp, w_dp = disp.size()

        ## scale alignment between input and current resolution (since it is coarse-to-fine)
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp   

        if self.reg_depth:
            depth = disp
            pts, k_scale = pixel2pts_ms_from_depth(k_aug, depth, local_scale / aug_size)
        else:
            disp = disp * w_dp
            pts, k_scale, depth = pixel2pts_ms_and_depth(k_aug, disp, local_scale / aug_size)

        # pose = torch.zeros_like(pose)
        # pose[:, 5] = 1
        ### apply transform
        T = transformation_from_parameters(pose[:, :3].unsqueeze(1), pose[:, 3:].unsqueeze(1))  # pose is B*6, T is B*4*4
        R = T[:, :3, :3]    # B*3*3
        t = T[:, :3, 3:]    # B*3*1
        pts_flat = pts.flatten(start_dim=2) # B*3*N
        pts_transformed = torch.matmul(R, pts_flat) + t
        pts_transformed_grid = pts_transformed.reshape_as(pts)
        rigid_flow = pts_transformed_grid - pts

        return rigid_flow, depth, k_scale, R, t


class ContextNetwork(nn.Module):
    def __init__(self, ch_in, reg_depth):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1)
        )
        self.conv_sf = conv(32, 3, isReLU=False)
        self.reg_depth = reg_depth
        if self.reg_depth:
            self.conv_d1 = nn.Sequential(
                conv(32, 1, isReLU=False), 
                torch.nn.Softplus()
            )
        else:
            self.conv_d1 = nn.Sequential(
                conv(32, 1, isReLU=False), 
                torch.nn.Sigmoid()
            )

    def forward(self, x):

        x_out = self.convs(x)
        sf = self.conv_sf(x_out)
        if self.reg_depth:
            d = self.conv_d1(x_out)
            d = d.clamp(min=1e-3, max=80)
        else:
            d = self.conv_d1(x_out) * 0.3

        return sf, d

class Flatten(nn.Module):
    ### in pytorch 1.2 there is not Flatten layer. In pytorch1.6 there is 
    ### https://discuss.pytorch.org/t/flatten-layer-of-pytorch-build-by-sequential-container/5983
    def forward(self, x):
        return x.view(x.size(0), -1)

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class SpatialPyramidPooling(nn.Module):
    """Generate fixed length representation regardless of image dimensions
    Based on the paper "Spatial Pyramid Pooling in Deep Convolutional Networks
    for Visual Recognition" (https://arxiv.org/pdf/1406.4729.pdf)
    :param [int] num_pools: Number of pools to split each input feature map into.
        Each element must be a perfect square in order to equally divide the
        pools across the feature map. Default corresponds to the original
        paper's implementation
    :param str mode: Specifies the type of pooling, either max or avg
    See: 
    https://github.com/addisonklinke/pytorch-architectures/blob/master/torcharch/modules/conv.py
    https://discuss.pytorch.org/t/elegant-implementation-of-spatial-pyramid-pooling-layer/831/2
    """

    def __init__(self, num_pools=[1, 2, 4], mode='max'):
        super(SpatialPyramidPooling, self).__init__()
        self.name = 'SpatialPyramidPooling'
        if mode == 'max':
            pool_func = nn.AdaptiveMaxPool2d
        elif mode == 'avg':
            pool_func = nn.AdaptiveAvgPool2d
        else:
            raise NotImplementedError(f"Unknown pooling mode '{mode}', expected 'max' or 'avg'")
        self.pools = nn.ModuleList([])
        for p in num_pools:
            # self.pools.append( pool_func( (max(1,int(p/2)), int(p)) ) )
            self.pools.append( pool_func( (int(p), int(p)) ) )

    def forward(self, feature_maps):
        """Pool feature maps at different bin levels and concatenate
        :param torch.tensor feature_maps: Arbitrarily shaped spatial and
            channel dimensions extracted from any generic convolutional
            architecture. Shape ``(N, C, H, W)``
        :return torch.tensor pooled: Concatenation of all pools with shape
            ``(N, C, sum(num_pools))``
        """
        assert feature_maps.dim() == 4, 'Expected 4D input of (N, C, H, W)'
        batch_size = feature_maps.size(0)
        channels = feature_maps.size(1)
        pooled = []
        for p in self.pools:
            pooled.append(p(feature_maps).view(batch_size, channels, -1))
        return torch.cat(pooled, dim=2)


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    from Monodepth2
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M

def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def uvd2xyz(flow_uvd, disp, k1, input_size):
    """flow_uvd: 3 channel grid of du, dv, ddisp"""
    ### uv1_grid, xy1_grid, xyz_0

    _, _, h_x, w_x = flow_uvd.size()
    local_scale = torch.zeros_like(input_size)
    local_scale[:, 0] = h_x
    local_scale[:, 1] = w_x

    disp = interpolate2d_as(disp, flow_uvd) * w_x
    pts1, k1_scale, uv1_grid = pixel2pts_ms_and_uv1grid(k1, disp, local_scale / input_size)

    ### uv1_grid_flowed
    flow_uvd_pixel = flow_uvd * w_x
    flow_uv_pixel = flow_uvd_pixel[:, :2]
    uv_grid_flowed = uv1_grid[:, :2] + flow_uv_pixel
    uv1_grid_flowed = torch.cat([uv_grid_flowed, uv1_grid[:, [2]]], dim=1)  # B*3*H*W
    
    ### xy1_grid_flowed
    disp_change = flow_uvd_pixel[:, [2]]
    disp_changed = disp + disp_change
    disp_changed = torch.clamp(disp_changed, min=1e-7)
    depth_changed = disp2depth_kitti(disp_changed, k1_scale[:, 0, 0])

    ### xyz_1
    depth_mat = depth_changed.flatten(2)
    pixel_mat = uv1_grid_flowed.flatten(2)
    pts_mat = torch.matmul(torch.inverse(k1_scale.cpu()).cuda(), pixel_mat) * depth_mat
    pts2 = pts_mat.reshape_as(uv1_grid_flowed)

    ### xyz_flow
    xyz_flow = pts2 - pts1

    return xyz_flow

def xyz2uvd(flow_xyz, disp, k1, input_size):

    _, _, h_x, w_x = flow_xyz.size()
    local_scale = torch.zeros_like(input_size)
    local_scale[:, 0] = h_x
    local_scale[:, 1] = w_x

    disp = interpolate2d_as(disp, flow_xyz) * w_x
    pts1, k1_scale, uv1_grid = pixel2pts_ms_and_uv1grid(k1, disp, local_scale / input_size)
    pts2 = pts1 + flow_xyz

    uvz2 = torch.matmul(k1_scale, pts2.flatten(2)).reshape_as(pts2)
    uv2 = uvz2[:, :2] / torch.clamp(uvz2[:, [2]], min=1e-3)
    ### or 
    # uv2 = pts2pixel(pts2, k1_scale)

    uv_flow = uv2 - uv1_grid[:, :2]

    depth2 = torch.clamp(pts2[:, [2]], min=1e-3)
    disp2 = depth2disp_kitti(depth2, k1_scale[:, 0, 0])

    d_flow = disp2 - disp

    uvd_flow_pixel = torch.cat([uv_flow, d_flow], dim=1)
    uvd_flow_normalized = uvd_flow_pixel / w_x

    return uvd_flow_normalized
