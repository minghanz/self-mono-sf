from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf

from models.forwardwarp_package.forward_warp import forward_warp
from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_ms, reconstructImg, reconstructPts, projectSceneFlow2Flow
from utils.monodepth_eval import compute_errors, compute_d1_all
from models.modules_sceneflow import WarpingLayer_Flow

from utils.sceneflow_util import pixel2pts_ms_and_depth ### needed by C3D loss
from utils.sceneflow_util import reconstructImgMask ### needed to calculate depth l1 loss between gt and flowed depth

import c3d
import torchsnooper
import logging
###############################################
## Basic Module 
###############################################

def _elementwise_epe(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=2, dim=1, keepdim=True)

def _elementwise_l1(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=1, dim=1, keepdim=True)

def _elementwise_robust_epe_char(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.pow(torch.norm(residual, p=2, dim=1, keepdim=True) + 0.01, 0.4)

def _SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1)(x)
    mu_y = nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    SSIM_img = torch.clamp((1 - SSIM) / 2, 0, 1)

    return tf.pad(SSIM_img, pad=(1, 1, 1, 1), mode='constant', value=0)

def _apply_disparity(img, disp):
    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1

    torch_vs = torch.__version__
    digits = torch_vs.split(".")
    torch_vs_n = float(digits[0]) + float(digits[1]) * 0.1
    grid_sample_specify_align_flag = torch_vs_n >= 1.3 
    if grid_sample_specify_align_flag:
        output = tf.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros', align_corners=True)
    else:
        output = tf.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')

    return output

def _generate_image_left(img, disp):
    return _apply_disparity(img, -disp)

def _adaptive_disocc_detection(flow):

    # init mask
    b, _, h, w, = flow.size()
    mask = torch.ones(b, 1, h, w, dtype=flow.dtype, device=flow.device).float().requires_grad_(False)    
    flow = flow.transpose(1, 2).transpose(2, 3)

    disocc = torch.clamp(forward_warp()(mask, flow), 0, 1) 
    disocc_map = (disocc > 0.5)

    if disocc_map.float().sum() < (b * h * w / 2):
        disocc_map = torch.ones(b, 1, h, w, dtype=torch.bool, device=flow.device).requires_grad_(False)
        
    return disocc_map

# @torchsnooper.snoop()
def _adaptive_disocc_detection_disp(disp):

    # # init
    b, _, h, w, = disp.size()
    mask = torch.ones(b, 1, h, w, dtype=disp.dtype, device=disp.device).float().requires_grad_(False)
    flow = torch.zeros(b, 2, h, w, dtype=disp.dtype, device=disp.device).float().requires_grad_(False)
    flow[:, 0:1, :, : ] = disp * w
    flow = flow.transpose(1, 2).transpose(2, 3)

    disocc = torch.clamp(forward_warp()(mask, flow), 0, 1) 
    disocc_map = (disocc > 0.5)

    if disocc_map.float().sum() < (b * h * w / 2):
        disocc_map = torch.ones(b, 1, h, w, dtype=torch.bool, device=disp.device).requires_grad_(False)
        
    return disocc_map

def _gradient_x(img):
    img = tf.pad(img, (0, 1, 0, 0), mode="replicate")
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
    return gx

def _gradient_y(img):
    img = tf.pad(img, (0, 0, 0, 1), mode="replicate")
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
    return gy

def _gradient_x_2nd(img):
    img_l = tf.pad(img, (1, 0, 0, 0), mode="replicate")[:, :, :, :-1]
    img_r = tf.pad(img, (0, 1, 0, 0), mode="replicate")[:, :, :, 1:]
    gx = img_l + img_r - 2 * img
    return gx

def _gradient_y_2nd(img):
    img_t = tf.pad(img, (0, 0, 1, 0), mode="replicate")[:, :, :-1, :]
    img_b = tf.pad(img, (0, 0, 0, 1), mode="replicate")[:, :, 1:, :]
    gy = img_t + img_b - 2 * img
    return gy

def _smoothness_motion_2nd(sf, img, beta=1):
    sf_grad_x = _gradient_x_2nd(sf)
    sf_grad_y = _gradient_y_2nd(sf)

    img_grad_x = _gradient_x(img) 
    img_grad_y = _gradient_y(img) 
    weights_x = torch.exp(-torch.mean(torch.abs(img_grad_x), 1, keepdim=True) * beta)
    weights_y = torch.exp(-torch.mean(torch.abs(img_grad_y), 1, keepdim=True) * beta)

    smoothness_x = sf_grad_x * weights_x
    smoothness_y = sf_grad_y * weights_y

    return (smoothness_x.abs() + smoothness_y.abs())

### for motion mask smoothness
def _smoothness_motion_1st(mask, img, beta=1):
    sf_grad_x = _gradient_x(mask)
    sf_grad_y = _gradient_y(mask)
    sf_grad_x = torch.mean(torch.abs(sf_grad_x), 1, keepdim=True)
    sf_grad_y = torch.mean(torch.abs(sf_grad_y), 1, keepdim=True)

    img_grad_x = _gradient_x(img) 
    img_grad_y = _gradient_y(img) 
    weights_x = torch.exp(-torch.mean(torch.abs(img_grad_x), 1, keepdim=True) * beta)
    weights_y = torch.exp(-torch.mean(torch.abs(img_grad_y), 1, keepdim=True) * beta)

    smoothness_x = sf_grad_x * weights_x
    smoothness_y = sf_grad_y * weights_y

    return (smoothness_x.abs() + smoothness_y.abs())

def _disp2depth_kitti_K(disp, k_value): 

    mask = (disp > 0).float()
    depth = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / (disp + (1.0 - mask))

    return depth

def _depth2disp_kitti_K(depth, k_value):

    disp = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / depth

    return disp



###############################################
## Loss function
###############################################

class Loss_SceneFlow_SelfSup_C3D(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup_C3D, self).__init__()
                
        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_smooth_w = 0.1
        self._sf_3d_pts = 0.2
        self._sf_3d_sm = 200

        self.c3d_pose_weight = args.c3d_pose_weight
        if self.c3d_pose_weight > 0:
            self.c3d_pose_loss = c3d.C3dLossKnnBtwnGT()

        if not args.disable_c3d:
            if args.c3d_knn_mode:
                ### c3d loss
                self.c3d_loss = c3d.C3DLossKnn()
            else:
                torch_vs = torch.__version__
                digits = torch_vs.split(".")
                torch_vs_n = float(digits[0]) + float(digits[1]) * 0.1
                assert torch_vs_n < 1.3  # avoid mistakenly working on pytorch3d env
                self.c3d_loss = c3d.C3DLoss()
                f_input = args.c3d_config_file #"/home/minghanz/self-mono-sf/scripts/c3d_config.txt"
                self.c3d_loss.parse_opts(f_input=f_input)
        
        # self._sf_c3d = 1e-4 if self.c3d_loss.opts.log_loss else 1e-6
        self._sf_c3d = args.c3d_weight
        
        # self._sf_dep = 1e-2
        # self._sf_dep = 1e-3
        self._sf_dep = args.dep_weight
        self.dep_l1_loss = c3d.DepthL1Loss(inbalance_to_closer=args.inbalance_to_closer)

        self.disable_c3d = args.disable_c3d
        self.dep_warp_loss = args.dep_warp_loss

        self.rigid = args.rigid
        self.rigid_mask_weight = 0.1
        self.rigid_norm_weight = 0.1
        self.rigid_mask_sm_weight = 0.1
        
        self.rigid_flow_loss = args.rigid_flow_loss

    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

        ## Photometric loss
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
        loss_img = (img_diff[left_occ]).mean()
        img_diff[~left_occ].detach_()

        ## Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        return loss_img + self._disp_smooth_w * loss_smooth, left_occ


    # @torchsnooper.snoop()
    def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii, target_dict):

        _, _, h_dp, w_dp = sf_f.size()
        disp_l1 = disp_l1 * w_dp
        disp_l2 = disp_l2 * w_dp

        ## scale
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp         

        pts1, k1_scale, depth1 = pixel2pts_ms_and_depth(k_l1_aug, disp_l1, local_scale / aug_size)
        pts2, k2_scale, depth2 = pixel2pts_ms_and_depth(k_l2_aug, disp_l2, local_scale / aug_size)

        _, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
        _, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

        pts2_warp = reconstructPts(coord1, pts2)
        pts1_warp = reconstructPts(coord2, pts1) 

        flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
        flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
        occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
        occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1
        ### occ_map_b is a mask on frame 2 by forward flowing pixels in frame 1. The non-masked region indicate which points in frame 2 does not have correspondence in frame 1 
        ### (when forward forwarding points from frame 1, some pixels in frame 2 does not have landing pixels from frame 1, meaning that they are occluded in frame 1. )
        ### This mask is to be used when comparing frame 2 and pseudo frame 2 sampled from frame 1 using backward flow. 
        ### The grid sampling operation automatically skip the disoccluded region in frame 1 (not present in frame 2 but present in frame 1).
        ### After grid sampling, the occluded region will show a duplicated of nearby region, the out-of-view region will show black or border pixels depending on the padding_mode arg in grid_sample. 
        ### Now the occ_map_b takes care of the occluded region in frame 1 (present in frame 2 but not present in frame 1)
        ### In ideal case, occ_map_b will also exclude the out-of-view pixels (present in frame 2 but out of view in frame 1). Before the network converges, it may not be the case. 
        ### The mask in reconstructImg takes care of out-of-view pixels, which is essentially the same grid_sample on a white mask. 

        ## Image reconstruction loss
        img_l2_warp = reconstructImg(coord1, img_l2_aug)
        img_l1_warp = reconstructImg(coord2, img_l1_aug)

        img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_im1 = img_diff1[occ_map_f].mean()
        loss_im2 = img_diff2[occ_map_b].mean()
        img_diff1[~occ_map_f].detach_()
        img_diff2[~occ_map_b].detach_()
        loss_im = loss_im1 + loss_im2
        
        ## Point reconstruction Loss
        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
        pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
        loss_pts1 = pts_diff1[occ_map_f].mean()
        loss_pts2 = pts_diff2[occ_map_b].mean()
        pts_diff1[~occ_map_f].detach_()
        pts_diff2[~occ_map_b].detach_()
        loss_pts = loss_pts1 + loss_pts2

        ############## C3D loss
        # torch.autograd.set_detect_anomaly(False)
        # if ii >= 1:
        if False:
            loss_c3d = torch.tensor(0)
            loss_dep = torch.tensor(0)
        else:
            depth_img_dict_1 = {}
            depth_img_dict_1["rgb"] = img_l1_aug
            depth_img_dict_1["pred"] = depth1
            depth_img_dict_1["pred_mask"] = target_dict["c3d_mask_l1_aug_scale_"+str(ii)]
            depth_img_dict_1["gt"] = target_dict["c3d_depth_l1_aug_scale_"+str(ii)]
            depth_img_dict_1["gt_mask"] = target_dict["c3d_mask_gt_l1_aug_scale_"+str(ii)]
            depth_img_dict_2 = {}
            depth_img_dict_2["rgb"] = img_l2_aug
            depth_img_dict_2["pred"] = depth2
            depth_img_dict_2["pred_mask"] = target_dict["c3d_mask_l2_aug_scale_"+str(ii)]
            depth_img_dict_2["gt"] = target_dict["c3d_depth_l2_aug_scale_"+str(ii)]
            depth_img_dict_2["gt_mask"] = target_dict["c3d_mask_gt_l2_aug_scale_"+str(ii)]
            flow_dict_1to2 = {}
            flow_dict_1to2["pred"] = sf_f

            # print(depth1.shape)
            # print(depth1.max())
            # print(depth1.min())

            # print(depth2.shape)
            # print(depth2.max())
            # print(depth2.min())

            # print(sf_f.shape)
            # print(sf_f.max())
            # print(sf_f.min())

            # print(sf_b.shape)
            # print(sf_b.max())
            # print(sf_b.min())
            
            if self.disable_c3d:
                loss_c3d = torch.tensor(0)
            else:
                flow_dict_1to2["mask"] = occ_map_f
                flow_dict_2to1 = {}
                flow_dict_2to1["pred"] = sf_b
                flow_dict_2to1["mask"] = occ_map_b
                cam_info = target_dict["c3d_cam_info_l_scale_"+str(ii)]
                loss_c3d = self.c3d_loss(depth_img_dict_1=depth_img_dict_1, depth_img_dict_2=depth_img_dict_2, flow_dict_1to2=flow_dict_1to2, flow_dict_2to1=flow_dict_2to1, cam_info=cam_info)
                loss_c3d = - loss_c3d

            ### l1 depth loss
            loss_dep_1 = self.dep_l1_loss(depth_img_dict_1["pred"], depth_img_dict_1["gt"], depth_img_dict_1["gt_mask"])
            loss_dep_2 = self.dep_l1_loss(depth_img_dict_2["pred"], depth_img_dict_2["gt"], depth_img_dict_2["gt_mask"])
            loss_dep = loss_dep_1 + loss_dep_2
            
            if self.dep_warp_loss:
                ### l1 depth on reconstructed depth
                ## Image reconstruction loss
                dep_l2_warp, dep_l2_mask = reconstructImgMask(coord1, depth_img_dict_2["pred"].detach())
                dep_l1_warp, dep_l1_mask = reconstructImgMask(coord2, depth_img_dict_1["pred"].detach())
                assert sf_f.ndim == 4 and sf_f.shape[1] == 3
                assert sf_b.ndim == 4 and sf_b.shape[1] == 3
                dep_l1_warp_from_2 = dep_l2_warp - sf_f[:,[2]]
                dep_l2_warp_from_1 = dep_l1_warp - sf_b[:,[2]]
                
                mask_l1_warp_from_2 = dep_l2_mask & depth_img_dict_1["gt_mask"] & occ_map_f
                mask_l2_warp_from_1 = dep_l1_mask & depth_img_dict_2["gt_mask"] & occ_map_b
                loss_dep_warp_1 = self.dep_l1_loss(dep_l1_warp_from_2, depth_img_dict_1["gt"], mask_l1_warp_from_2)
                loss_dep_warp_2 = self.dep_l1_loss(dep_l2_warp_from_1, depth_img_dict_2["gt"], mask_l2_warp_from_1)
                loss_dep = loss_dep + loss_dep_warp_1 + loss_dep_warp_2


        ## 3D motion smoothness loss
        loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

        ## Loss Summnation
        sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s
        
        ### include c3d loss
        # loss_c3d_weight = self._sf_c3d * 8**ii              ### 8**ii is because the c3d loss scale is not consistent with differently scaled images
        loss_c3d_weight = self._sf_c3d              ### 8**ii is because the c3d loss scale is not consistent with differently scaled images
        sceneflow_loss_c3d = sceneflow_loss + loss_c3d_weight * loss_c3d + self._sf_dep * loss_dep

        # print("ii: {}, sum: {:.4f}, loss_im: {:.4f}, loss_pts: {:.4f}, self._sf_3d_pts: {:.4f}, loss_pts_wted: {:.4f}, loss_3d_s: {:.4f}, self._sf_3d_sm: {:.4f}, loss_3d_s_wted: {:.4f}, loss_c3d: {:.4f}, loss_c3d_wt: {:.4f}, loss_c3d_wted: {:.4f}".format(
        #         ii, sceneflow_loss, loss_im, loss_pts, self._sf_3d_pts, self._sf_3d_pts * loss_pts, loss_3d_s, self._sf_3d_sm, self._sf_3d_sm * loss_3d_s, loss_c3d, loss_c3d_weight, loss_c3d_weight * loss_c3d))

        return sceneflow_loss, loss_im, loss_pts, loss_3d_s, loss_c3d, loss_dep, sceneflow_loss_c3d

    def detaching_grad_of_outputs(self, output_dict):
        
        for ii in range(0, len(output_dict['flow_f'])):
            output_dict['flow_f'][ii].detach_()
            output_dict['flow_b'][ii].detach_()
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()

        return None

    def rigid_loss(self, res_norm_f, res_norm_b, res_mask_f, res_mask_b, img_l1, img_l2):
        rigid_mask_f = 1 - res_mask_f
        rigid_mask_b = 1 - res_mask_b

        
        # assert rigid_mask_f.max() < 1 and rigid_mask_f.min() > 0, "{} {}".format(rigid_mask_f.max(), rigid_mask_f.min())
        # assert rigid_mask_b.max() < 1 and rigid_mask_b.min() > 0, "{} {}".format(rigid_mask_b.max(), rigid_mask_b.min())

        # rigid_mask_f_bin = rigid_mask_f > 0.5
        # rigid_mask_b_bin = rigid_mask_b > 0.5
        
        # res_norm_mean_f = res_norm_f[rigid_mask_f_bin].mean()
        # res_norm_mean_b = res_norm_b[rigid_mask_b_bin].mean()

        res_norm_mean_f = (res_norm_f * rigid_mask_f).mean()
        res_norm_mean_b = (res_norm_b * rigid_mask_b).mean()
        
        # rigid_mask_f = rigid_mask_f.clamp(min=1e-7, max=1)
        # rigid_mask_b = rigid_mask_b.clamp(min=1e-7, max=1)
        
        # res_mask_f_cross_entropy = - torch.log(rigid_mask_f).mean()
        # res_mask_b_cross_entropy = - torch.log(rigid_mask_b).mean()

        res_mask_f_cross_entropy = res_norm_f.mean().detach() * res_mask_f.mean()
        res_mask_b_cross_entropy = res_norm_b.mean().detach() * res_mask_b.mean()
        
        res_mask_f_smoothness = _smoothness_motion_1st(res_mask_f, img_l1, 10).mean()
        res_mask_b_smoothness = _smoothness_motion_1st(res_mask_b, img_l2, 10).mean()
        
        loss_rigid_norm = (res_norm_mean_f + res_norm_mean_b) * self.rigid_norm_weight
        loss_rigid_mask = (res_mask_f_cross_entropy + res_mask_b_cross_entropy) * self.rigid_mask_weight
        loss_rigid_mask_smoothness = (res_mask_f_smoothness + res_mask_b_smoothness) * self.rigid_mask_sm_weight
        loss_rigid_mask = loss_rigid_mask + loss_rigid_mask_smoothness

        return loss_rigid_mask, loss_rigid_norm

    def calc_c3d_pose_loss(self, target_dict, R12, t12, R21, t21, img_l1_aug, img_l2_aug, ii):
        depth_img_dict_1 = {}
        depth_img_dict_1["rgb"] = img_l1_aug
        depth_img_dict_1["gt"] = target_dict["c3d_depth_l1_aug_scale_"+str(ii)]
        depth_img_dict_1["gt_mask"] = target_dict["c3d_mask_gt_l1_aug_scale_"+str(ii)]
        depth_img_dict_2 = {}
        depth_img_dict_2["rgb"] = img_l2_aug
        depth_img_dict_2["gt"] = target_dict["c3d_depth_l2_aug_scale_"+str(ii)]
        depth_img_dict_2["gt_mask"] = target_dict["c3d_mask_gt_l2_aug_scale_"+str(ii)]
        cam_info = target_dict["c3d_cam_info_l_scale_"+str(ii)]

        c3d_pose_loss = - self.c3d_pose_loss(depth_img_dict_1, depth_img_dict_2, cam_info, R12, t12, R21, t21) * self.c3d_pose_weight
        return c3d_pose_loss

    # @torchsnooper.snoop()
    def forward_core(self, flow_f, flow_b, disp_l1_all, disp_l2_all, disp_r1_dict, disp_r2_dict, target_dict, output_dict, dup=False):
        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_sf_2d = 0
        loss_sf_3d = 0
        loss_sf_sm = 0

        ### c3d loss
        loss_sf_c3d = 0
        loss_sf_dep = 0
        loss_sf_sum_c3d = 0

        ### rigid loss
        loss_rigid_mask_sum = 0
        loss_rigid_norm_sum = 0

        ### c3d pose loss
        loss_c3d_pose_sum = torch.tensor(0, device=flow_f[0].device)

        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(flow_f, flow_b, disp_l1_all, disp_l2_all, disp_r1_dict, disp_r2_dict)):

            assert(sf_f.size()[2:4] == sf_b.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
            
            ## For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

            ## Disp Loss
            loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


            ## Sceneflow Loss           
            loss_sceneflow, loss_im, loss_pts, loss_3d_s, loss_c3d, loss_dep, loss_sceneflow_c3d = self.sceneflow_loss(sf_f, sf_b, 
                                                                            disp_l1, disp_l2,
                                                                            disp_occ_l1, disp_occ_l2,
                                                                            k_l1_aug, k_l2_aug,
                                                                            img_l1_aug, img_l2_aug, 
                                                                            aug_size, ii, target_dict)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
            loss_sf_2d = loss_sf_2d + loss_im * self._weights[ii]            
            loss_sf_3d = loss_sf_3d + loss_pts * self._weights[ii]            
            loss_sf_sm = loss_sf_sm + loss_3d_s * self._weights[ii]            

            ### c3d loss
            loss_sf_c3d = loss_sf_c3d + loss_c3d * self._weights[ii]            
            loss_sf_dep = loss_sf_dep + loss_dep * self._weights[ii]            
            loss_sf_sum_c3d = loss_sf_sum_c3d + loss_sceneflow_c3d * self._weights[ii]

            ### rigid loss
            if not dup:
                if self.rigid:
                    norm_f = output_dict["flow_f_res_norm"][ii]
                    norm_b = output_dict["flow_b_res_norm"][ii]
                    mask_f = output_dict["flow_f_res_mask"][ii]
                    mask_b = output_dict["flow_b_res_mask"][ii]
                    loss_rigid_mask, loss_rigid_norm = self.rigid_loss(norm_f, norm_b, mask_f, mask_b, img_l1_aug, img_l2_aug)
                    loss_rigid_mask_sum = loss_rigid_mask_sum + loss_rigid_mask * self._weights[ii]
                    loss_rigid_norm_sum = loss_rigid_norm_sum + loss_rigid_norm * self._weights[ii]

                    ### c3d pose loss
                    if self.c3d_pose_weight > 0:
                        R12 = output_dict["R12s"][ii]
                        t12 = output_dict["t12s"][ii]
                        R21 = output_dict["R21s"][ii]
                        t21 = output_dict["t21s"][ii]
                        c3d_pose_loss = self.calc_c3d_pose_loss(target_dict, R12, t12, R21, t21, img_l1_aug, img_l2_aug, ii)
                        loss_c3d_pose_sum = loss_c3d_pose_sum + c3d_pose_loss * self._weights[ii]

        return loss_dp_sum, loss_sf_sum, loss_sf_2d, loss_sf_3d, loss_sf_sm, loss_sf_c3d, loss_sf_dep, loss_sf_sum_c3d, loss_rigid_mask_sum, loss_rigid_norm_sum, loss_c3d_pose_sum

    # @torchsnooper.snoop()
    def forward(self, output_dict, target_dict):

        loss_dict = {}

        batch_size = target_dict['input_l1'].size(0)

        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        loss_dp_sum, loss_sf_sum, loss_sf_2d, loss_sf_3d, loss_sf_sm, loss_sf_c3d, loss_sf_dep, loss_sf_sum_c3d, loss_rigid_mask_sum, loss_rigid_norm_sum, loss_c3d_pose_sum = \
            self.forward_core(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict, target_dict, output_dict)
        
        if self.rigid and self.rigid_flow_loss:
            loss_dp_sum_2, loss_sf_sum_2, loss_sf_2d_2, loss_sf_3d_2, loss_sf_sm_2, loss_sf_c3d_2, loss_sf_dep_2, loss_sf_sum_c3d_2, loss_rigid_mask_sum_2, loss_rigid_norm_sum_2, loss_c3d_pose_sum_2 = \
                self.forward_core(output_dict['flow_f_rigid_acc'], output_dict['flow_b_rigid_acc'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict, target_dict, output_dict, dup=True)

            loss_dp_sum = (loss_dp_sum + loss_dp_sum_2) * 0.5
            loss_sf_sum = (loss_sf_sum + loss_sf_sum_2) * 0.5

            loss_sf_2d = (loss_sf_2d + loss_sf_2d_2) * 0.5
            loss_sf_3d = (loss_sf_3d + loss_sf_3d_2) * 0.5

            loss_sf_sm = (loss_sf_sm + loss_sf_sm_2) * 0.5
            loss_sf_c3d = (loss_sf_c3d + loss_sf_c3d_2) * 0.5

            loss_sf_dep = (loss_sf_dep + loss_sf_dep_2) * 0.5
            loss_sf_sum_c3d = (loss_sf_sum_c3d + loss_sf_sum_c3d_2) * 0.5

            # loss_rigid_mask_sum = (loss_rigid_mask_sum + loss_rigid_mask_sum_2) * 0.5
            # loss_rigid_norm_sum = (loss_rigid_norm_sum + loss_rigid_norm_sum_2) * 0.5

        # finding weight
        f_loss = loss_sf_sum.detach()
        d_loss = loss_dp_sum.detach()
        max_val = torch.max(f_loss, d_loss)
        f_weight = max_val / f_loss
        d_weight = max_val / d_loss

        total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum * d_weight
        loss_dict["sf"] = loss_sf_sum * f_weight
        loss_dict["s_2"] = loss_sf_2d * f_weight
        loss_dict["s_3"] = loss_sf_3d * self._sf_3d_pts * f_weight
        loss_dict["s_3s"] = loss_sf_sm * self._sf_3d_sm * f_weight
        loss_dict["total_loss"] = total_loss

        ### c3d loss
        total_loss_w_c3d = loss_sf_sum_c3d * f_weight + loss_dp_sum * d_weight

        ### rigid residual loss
        loss_dict["fw"] = f_weight
        loss_dict["dw"] = d_weight
        if self.rigid:
            loss_dict["s_rigm"] = loss_rigid_mask_sum
            loss_dict["s_rign"] = loss_rigid_norm_sum
            loss_dict["s_pose"] = loss_c3d_pose_sum
            
            total_loss_w_c3d = total_loss_w_c3d + loss_dict["s_rigm"] + loss_dict["s_rign"] + loss_dict["s_pose"]

        loss_dict["s_c3d"] = loss_sf_c3d * self._sf_c3d * f_weight
        loss_dict["s_dep"] = loss_sf_dep * self._sf_dep * f_weight
        loss_dict["t_loss_w_c3d"] = total_loss_w_c3d

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict

class Loss_SceneFlow_SelfSup(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup, self).__init__()
                
        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_smooth_w = 0.1
        self._sf_3d_pts = 0.2
        self._sf_3d_sm = 200

    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

        ## Photometric loss
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
        loss_img = (img_diff[left_occ]).mean()
        img_diff[~left_occ].detach_()

        ## Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        return loss_img + self._disp_smooth_w * loss_smooth, left_occ


    # @torchsnooper.snoop()
    def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

        _, _, h_dp, w_dp = sf_f.size()
        disp_l1 = disp_l1 * w_dp
        disp_l2 = disp_l2 * w_dp

        ## scale
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp         

        pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
        pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

        _, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
        _, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

        pts2_warp = reconstructPts(coord1, pts2)
        pts1_warp = reconstructPts(coord2, pts1) 

        flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
        flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
        occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
        occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

        ## Image reconstruction loss
        img_l2_warp = reconstructImg(coord1, img_l2_aug)
        img_l1_warp = reconstructImg(coord2, img_l1_aug)

        img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_im1 = img_diff1[occ_map_f].mean()
        loss_im2 = img_diff2[occ_map_b].mean()
        img_diff1[~occ_map_f].detach_()
        img_diff2[~occ_map_b].detach_()
        loss_im = loss_im1 + loss_im2
        
        ## Point reconstruction Loss
        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
        pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
        loss_pts1 = pts_diff1[occ_map_f].mean()
        loss_pts2 = pts_diff2[occ_map_b].mean()
        pts_diff1[~occ_map_f].detach_()
        pts_diff2[~occ_map_b].detach_()
        loss_pts = loss_pts1 + loss_pts2

        ## 3D motion smoothness loss
        loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

        ## Loss Summnation
        sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s
        
        return sceneflow_loss, loss_im, loss_pts, loss_3d_s

    def detaching_grad_of_outputs(self, output_dict):
        
        for ii in range(0, len(output_dict['flow_f'])):
            output_dict['flow_f'][ii].detach_()
            output_dict['flow_b'][ii].detach_()
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()

        return None

    # @torchsnooper.snoop()
    def forward(self, output_dict, target_dict):

        loss_dict = {}

        batch_size = target_dict['input_l1'].size(0)
        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_sf_2d = 0
        loss_sf_3d = 0
        loss_sf_sm = 0
        
        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

            assert(sf_f.size()[2:4] == sf_b.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
            
            ## For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

            ## Disp Loss
            loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


            ## Sceneflow Loss           
            loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f, sf_b, 
                                                                            disp_l1, disp_l2,
                                                                            disp_occ_l1, disp_occ_l2,
                                                                            k_l1_aug, k_l2_aug,
                                                                            img_l1_aug, img_l2_aug, 
                                                                            aug_size, ii)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
            loss_sf_2d = loss_sf_2d + loss_im            
            loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s

        # finding weight
        f_loss = loss_sf_sum.detach()
        d_loss = loss_dp_sum.detach()
        max_val = torch.max(f_loss, d_loss)
        f_weight = max_val / f_loss
        d_weight = max_val / d_loss

        total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["sf"] = loss_sf_sum
        loss_dict["s_2"] = loss_sf_2d
        loss_dict["s_3"] = loss_sf_3d
        loss_dict["s_3s"] = loss_sf_sm
        loss_dict["total_loss"] = total_loss

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict


class Loss_SceneFlow_SemiSupFinetune(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SemiSupFinetune, self).__init__()        

        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._unsup_loss = Loss_SceneFlow_SelfSup(args)


    def forward(self, output_dict, target_dict):

        loss_dict = {}

        unsup_loss_dict = self._unsup_loss(output_dict, target_dict)
        unsup_loss = unsup_loss_dict['total_loss']

        ## Ground Truth
        gt_disp1 = target_dict['target_disp']
        gt_disp1_mask = (target_dict['target_disp_mask']==1).float()   
        gt_disp2 = target_dict['target_disp2_occ']
        gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()   
        gt_flow = target_dict['target_flow']
        gt_flow_mask = (target_dict['target_flow_mask']==1).float()

        b, _, h_dp, w_dp = gt_disp1.size()     

        disp_loss = 0
        flow_loss = 0

        for ii, sf_f in enumerate(output_dict['flow_f_pp']):

            ## disp1
            disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][ii], gt_disp1, mode="bilinear") * w_dp
            valid_abs_rel = torch.abs(gt_disp1 - disp_l1) * gt_disp1_mask
            valid_abs_rel[gt_disp1_mask == 0].detach_()
            disp_l1_loss = valid_abs_rel[gt_disp1_mask != 0].mean()

            ## Flow Loss
            sf_f_up = interpolate2d_as(sf_f, gt_flow, mode="bilinear")
            out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], sf_f_up, disp_l1)
            valid_epe = _elementwise_robust_epe_char(out_flow, gt_flow) * gt_flow_mask
                
            valid_epe[gt_flow_mask == 0].detach_()
            flow_l1_loss = valid_epe[gt_flow_mask != 0].mean()

            ## disp1_next
            out_depth_l1 = _disp2depth_kitti_K(disp_l1, target_dict['input_k_l1'][:, 0, 0])
            out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
            out_depth_l1_next = out_depth_l1 + sf_f_up[:, 2:3, :, :]
            disp_l1_next = _depth2disp_kitti_K(out_depth_l1_next, target_dict['input_k_l1'][:, 0, 0])

            valid_abs_rel = torch.abs(gt_disp2 - disp_l1_next) * gt_disp2_mask
            valid_abs_rel[gt_disp2_mask == 0].detach_()
            disp_l2_loss = valid_abs_rel[gt_disp2_mask != 0].mean()
             
            disp_loss = disp_loss + (disp_l1_loss + disp_l2_loss) * self._weights[ii]
            flow_loss = flow_loss + flow_l1_loss * self._weights[ii]

        # finding weight
        u_loss = unsup_loss.detach()
        d_loss = disp_loss.detach()
        f_loss = flow_loss.detach()

        max_val = torch.max(torch.max(f_loss, d_loss), u_loss)

        u_weight = max_val / u_loss
        d_weight = max_val / d_loss 
        f_weight = max_val / f_loss 

        total_loss = unsup_loss * u_weight + disp_loss * d_weight + flow_loss * f_weight
        loss_dict["unsup_loss"] = unsup_loss
        loss_dict["dp_loss"] = disp_loss
        loss_dict["fl_loss"] = flow_loss
        loss_dict["total_loss"] = total_loss

        return loss_dict



###############################################
## Eval
###############################################

def eval_module_disp_depth(gt_disp, gt_disp_mask, output_disp, gt_depth, output_depth):
    
    loss_dict = {}
    batch_size = gt_disp.size(0)
    gt_disp_mask_f = gt_disp_mask.float()

    ## KITTI disparity metric
    d_valid_epe = _elementwise_epe(output_disp, gt_disp) * gt_disp_mask_f
    d_outlier_epe = (d_valid_epe > 3).float() * ((d_valid_epe / gt_disp) > 0.05).float() * gt_disp_mask_f
    loss_dict["otl"] = (d_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
    loss_dict["otl_img"] = d_outlier_epe

    ## MonoDepth metric
    abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = compute_errors(gt_depth[gt_disp_mask], output_depth[gt_disp_mask])        
    loss_dict["abs_rel"] = abs_rel
    loss_dict["sq_rel"] = sq_rel
    loss_dict["rms"] = rms
    loss_dict["log_rms"] = log_rms
    loss_dict["a1"] = a1
    loss_dict["a2"] = a2
    loss_dict["a3"] = a3

    return loss_dict


class Eval_MonoDepth_Eigen(nn.Module):
    def __init__(self):
        super(Eval_MonoDepth_Eigen, self).__init__()

    def forward(self, output_dict, target_dict):
        
        loss_dict = {}

        ## Depth Eval
        gt_depth = target_dict['target_depth']

        out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_depth, mode="bilinear") * gt_depth.size(3)
        out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, target_dict['input_k_l1'][:, 0, 0])
        out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
        gt_depth_mask = (gt_depth > 1e-3) * (gt_depth < 80)        

        ## Compute metrics
        abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = compute_errors(gt_depth[gt_depth_mask], out_depth_l1[gt_depth_mask])

        output_dict["out_disp_l_pp"] = out_disp_l1
        output_dict["out_depth_l_pp"] = out_depth_l1
        loss_dict["ab_r"] = abs_rel
        loss_dict["sq_r"] = sq_rel
        loss_dict["rms"] = rms
        loss_dict["log_rms"] = log_rms
        loss_dict["a1"] = a1
        loss_dict["a2"] = a2
        loss_dict["a3"] = a3

        return loss_dict


class Eval_SceneFlow_KITTI_Test(nn.Module):
    def __init__(self):
        super(Eval_SceneFlow_KITTI_Test, self).__init__()

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        ##################################################
        ## Depth 1
        ##################################################
        input_l1 = target_dict['input_l1']
        intrinsics = target_dict['input_k_l1']

        out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], input_l1, mode="bilinear") * input_l1.size(3)
        out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
        out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
        output_dict["out_disp_l_pp"] = out_disp_l1

        ##################################################
        ## Optical Flow Eval
        ##################################################
        out_sceneflow = interpolate2d_as(output_dict['flow_f_pp'][0], input_l1, mode="bilinear")
        out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])        
        output_dict["out_flow_pp"] = out_flow

        ##################################################
        ## Depth 2
        ##################################################
        out_depth_l1_next = out_depth_l1 + out_sceneflow[:, 2:3, :, :]
        out_disp_l1_next = _depth2disp_kitti_K(out_depth_l1_next, intrinsics[:, 0, 0])
        output_dict["out_disp_l_pp_next"] = out_disp_l1_next        

        loss_dict['sf'] = (out_disp_l1_next * 0).sum()

        return loss_dict


class Eval_SceneFlow_KITTI_Train(nn.Module):
    def __init__(self, args):
        super(Eval_SceneFlow_KITTI_Train, self).__init__()


    def forward(self, output_dict, target_dict):

        loss_dict = {}

        gt_flow = target_dict['target_flow']
        gt_flow_mask = (target_dict['target_flow_mask']==1).float()

        gt_disp = target_dict['target_disp']
        gt_disp_mask = (target_dict['target_disp_mask']==1).float()

        gt_disp2_occ = target_dict['target_disp2_occ']
        gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()

        gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask

        intrinsics = target_dict['input_k_l1']                

        ##################################################
        ## Depth 1
        ##################################################

        batch_size, _, _, width = gt_disp.size()

        out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * width
        out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
        out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
        gt_depth_l1 = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

        dict_disp0_occ = eval_module_disp_depth(gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)
        
        output_dict["out_disp_l_pp"] = out_disp_l1
        output_dict["out_depth_l_pp"] = out_depth_l1

        d0_outlier_image = dict_disp0_occ['otl_img']
        loss_dict["d_abs"] = dict_disp0_occ['abs_rel']
        loss_dict["d_sq"] = dict_disp0_occ['sq_rel']
        loss_dict["d1"] = dict_disp0_occ['otl']

        ##################################################
        ## Optical Flow Eval
        ##################################################
        
        out_sceneflow = interpolate2d_as(output_dict['flow_f_pp'][0], gt_flow, mode="bilinear")
        out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])

        ## Flow Eval
        valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask
        loss_dict["f_epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
        output_dict["out_flow_pp"] = out_flow

        flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
        flow_outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
        loss_dict["f1"] = (flow_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68


        ##################################################
        ## Depth 2
        ##################################################

        out_depth_l1_next = out_depth_l1 + out_sceneflow[:, 2:3, :, :]
        out_disp_l1_next = _depth2disp_kitti_K(out_depth_l1_next, intrinsics[:, 0, 0])
        gt_depth_l1_next = _disp2depth_kitti_K(gt_disp2_occ, intrinsics[:, 0, 0])

        dict_disp1_occ = eval_module_disp_depth(gt_disp2_occ, gt_disp2_mask.bool(), out_disp_l1_next, gt_depth_l1_next, out_depth_l1_next)
        
        output_dict["out_disp_l_pp_next"] = out_disp_l1_next
        output_dict["out_depth_l_pp_next"] = out_depth_l1_next

        d1_outlier_image = dict_disp1_occ['otl_img']
        loss_dict["d2"] = dict_disp1_occ['otl']


        ##################################################
        ## Scene Flow Eval
        ##################################################

        outlier_sf = (flow_outlier_epe.bool() + d0_outlier_image.bool() + d1_outlier_image.bool()).float() * gt_sf_mask
        loss_dict["sf"] = (outlier_sf.view(batch_size, -1).sum(1)).mean() / 91873.4

        return loss_dict



###############################################
## Ablation - Loss_SceneFlow_SelfSup
###############################################

class Loss_SceneFlow_SelfSup_NoOcc(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup_NoOcc, self).__init__()
                
        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_smooth_w = 0.1
        self._sf_3d_pts = 0.2
        self._sf_3d_sm = 200


    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        # left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

        ## Photometric loss: 
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
        loss_img = img_diff.mean()
        # loss_img = (img_diff[left_occ]).mean()
        # img_diff[~left_occ].detach_()

        ## Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        return loss_img + self._disp_smooth_w * loss_smooth#, left_occ


    def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

        ## Depth2Pts
        _, _, h_dp, w_dp = sf_f.size()
        disp_l1 = disp_l1 * w_dp
        disp_l2 = disp_l2 * w_dp

        ## scale
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp         

        pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
        pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

        _, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
        _, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

        pts2_warp = reconstructPts(coord1, pts2)
        pts1_warp = reconstructPts(coord2, pts1) 

        flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
        flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
        # occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
        # occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

        ## Image reconstruction loss
        # img_l2_warp = self.warping_layer_aug(img_l2, flow_f, aug_scale, coords)
        # img_l1_warp = self.warping_layer_aug(img_l1, flow_b, aug_scale, coords)
        img_l2_warp = reconstructImg(coord1, img_l2_aug)
        img_l1_warp = reconstructImg(coord2, img_l1_aug)

        img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_im1 = img_diff1.mean()
        loss_im2 = img_diff2.mean()
        # loss_im1 = img_diff1[occ_map_f].mean()
        # loss_im2 = img_diff2[occ_map_b].mean()
        # img_diff1[~occ_map_f].detach_()
        # img_diff2[~occ_map_b].detach_()
        loss_im = loss_im1 + loss_im2
        
        ## Point Reconstruction Loss
        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
        pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
        loss_pts1 = pts_diff1.mean()
        loss_pts2 = pts_diff2.mean()
        # loss_pts1 = pts_diff1[occ_map_f].mean()
        # loss_pts2 = pts_diff2[occ_map_b].mean()
        # pts_diff1[~occ_map_f].detach_()
        # pts_diff2[~occ_map_b].detach_()
        loss_pts = loss_pts1 + loss_pts2

        ## 3D motion smoothness loss
        loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

        ## Loss Summnation
        sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s
        
        return sceneflow_loss, loss_im, loss_pts, loss_3d_s

    def detaching_grad_of_outputs(self, output_dict):
        
        for ii in range(0, len(output_dict['flow_f'])):
            output_dict['flow_f'][ii].detach_()
            output_dict['flow_b'][ii].detach_()
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()

        return None

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        ## SceneFlow Loss
        batch_size = target_dict['input_l1'].size(0)
        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_sf_2d = 0
        loss_sf_3d = 0
        loss_sf_sm = 0
        
        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

            assert(sf_f.size()[2:4] == sf_b.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
            
            ## For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

            ## Depth Loss
            loss_disp_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


            ## Sceneflow Loss           
            loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f, sf_b, 
                                                                            disp_l1, disp_l2,
                                                                            k_l1_aug, k_l2_aug,
                                                                            img_l1_aug, img_l2_aug, 
                                                                            aug_size, ii)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
            loss_sf_2d = loss_sf_2d + loss_im            
            loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s

        # finding weight
        f_loss = loss_sf_sum.detach()
        d_loss = loss_dp_sum.detach()
        max_val = torch.max(f_loss, d_loss)
        f_weight = max_val / f_loss
        d_weight = max_val / d_loss

        total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["sf"] = loss_sf_sum
        loss_dict["s_2"] = loss_sf_2d
        loss_dict["s_3"] = loss_sf_3d
        loss_dict["s_3s"] = loss_sf_sm
        loss_dict["total_loss"] = total_loss

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict

class Loss_SceneFlow_SelfSup_NoPts(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup_NoPts, self).__init__()
                
        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_smooth_w = 0.1
        self._sf_3d_pts = 0.2
        self._sf_3d_sm = 200


    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

        ## Photometric loss: 
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
        loss_img = (img_diff[left_occ]).mean()
        img_diff[~left_occ].detach_()

        ## Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        return loss_img + self._disp_smooth_w * loss_smooth, left_occ


    def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

        ## Depth2Pts
        _, _, h_dp, w_dp = sf_f.size()
        disp_l1 = disp_l1 * w_dp
        disp_l2 = disp_l2 * w_dp

        ## scale
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp         

        pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
        pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

        _, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
        _, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

        # pts2_warp = reconstructPts(coord1, pts2)
        # pts1_warp = reconstructPts(coord2, pts1) 

        flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
        flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
        occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
        occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

        ## Image reconstruction loss
        # img_l2_warp = self.warping_layer_aug(img_l2, flow_f, aug_scale, coords)
        # img_l1_warp = self.warping_layer_aug(img_l1, flow_b, aug_scale, coords)
        img_l2_warp = reconstructImg(coord1, img_l2_aug)
        img_l1_warp = reconstructImg(coord2, img_l1_aug)

        img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_im1 = img_diff1[occ_map_f].mean()
        loss_im2 = img_diff2[occ_map_b].mean()
        img_diff1[~occ_map_f].detach_()
        img_diff2[~occ_map_b].detach_()
        loss_im = loss_im1 + loss_im2
        
        # ## Point Reconstruction Loss
        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
        # pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        # pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
        # loss_pts1 = pts_diff1[occ_map_f].mean()
        # loss_pts2 = pts_diff2[occ_map_b].mean()
        # pts_diff1[~occ_map_f].detach_()
        # pts_diff2[~occ_map_b].detach_()
        # loss_pts = loss_pts1 + loss_pts2

        ## 3D motion smoothness loss
        loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

        ## Loss Summnation
        sceneflow_loss = loss_im + self._sf_3d_sm * loss_3d_s# + self._sf_3d_pts * loss_pts
        
        return sceneflow_loss, loss_im, loss_3d_s#, loss_pts

    def detaching_grad_of_outputs(self, output_dict):
        
        for ii in range(0, len(output_dict['flow_f'])):
            output_dict['flow_f'][ii].detach_()
            output_dict['flow_b'][ii].detach_()
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()

        return None

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        ## SceneFlow Loss
        batch_size = target_dict['input_l1'].size(0)
        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_sf_2d = 0
        # loss_sf_3d = 0
        loss_sf_sm = 0
        
        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

            assert(sf_f.size()[2:4] == sf_b.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
            
            ## For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

            ## Depth Loss
            loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


            ## Sceneflow Loss           
            loss_sceneflow, loss_im, loss_3d_s = self.sceneflow_loss(sf_f, sf_b, 
                                                                            disp_l1, disp_l2,
                                                                            disp_occ_l1, disp_occ_l2,
                                                                            k_l1_aug, k_l2_aug,
                                                                            img_l1_aug, img_l2_aug, 
                                                                            aug_size, ii)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
            loss_sf_2d = loss_sf_2d + loss_im            
            # loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s

        # finding weight
        f_loss = loss_sf_sum.detach()
        d_loss = loss_dp_sum.detach()
        max_val = torch.max(f_loss, d_loss)
        f_weight = max_val / f_loss
        d_weight = max_val / d_loss

        total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["sf"] = loss_sf_sum
        loss_dict["s_2"] = loss_sf_2d
        # loss_dict["s_3"] = loss_sf_3d
        loss_dict["s_3s"] = loss_sf_sm
        loss_dict["total_loss"] = total_loss

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict

class Loss_SceneFlow_SelfSup_NoPtsNoOcc(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup_NoPtsNoOcc, self).__init__()
                
        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_smooth_w = 0.1
        self._sf_3d_pts = 0.2
        self._sf_3d_sm = 200


    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        # left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

        ## Photometric loss: 
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
        loss_img = img_diff.mean()
        # loss_img = (img_diff[left_occ]).mean()
        # img_diff[~left_occ].detach_()

        ## Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        return loss_img + self._disp_smooth_w * loss_smooth#, left_occ


    def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

        ## Depth2Pts
        _, _, h_dp, w_dp = sf_f.size()
        disp_l1 = disp_l1 * w_dp
        disp_l2 = disp_l2 * w_dp

        ## scale
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp         

        pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
        pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

        _, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
        _, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

        # pts2_warp = reconstructPts(coord1, pts2)
        # pts1_warp = reconstructPts(coord2, pts1) 

        flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
        flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
        # occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
        # occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

        ## Image reconstruction loss
        # img_l2_warp = self.warping_layer_aug(img_l2, flow_f, aug_scale, coords)
        # img_l1_warp = self.warping_layer_aug(img_l1, flow_b, aug_scale, coords)
        img_l2_warp = reconstructImg(coord1, img_l2_aug)
        img_l1_warp = reconstructImg(coord2, img_l1_aug)

        img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_im1 = img_diff1.mean()
        loss_im2 = img_diff2.mean()
        # loss_im1 = img_diff1[occ_map_f].mean()
        # loss_im2 = img_diff2[occ_map_b].mean()
        # img_diff1[~occ_map_f].detach_()
        # img_diff2[~occ_map_b].detach_()
        loss_im = loss_im1 + loss_im2
        
        ## Point Reconstruction Loss
        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
        # pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        # pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
        # loss_pts1 = pts_diff1.mean()
        # loss_pts2 = pts_diff2.mean()
        # loss_pts1 = pts_diff1[occ_map_f].mean()
        # loss_pts2 = pts_diff2[occ_map_b].mean()
        # pts_diff1[~occ_map_f].detach_()
        # pts_diff2[~occ_map_b].detach_()
        # loss_pts = loss_pts1 + loss_pts2

        ## 3D motion smoothness loss
        loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

        ## Loss Summnation
        sceneflow_loss = loss_im + self._sf_3d_sm * loss_3d_s # + self._sf_3d_pts * loss_pts
        
        return sceneflow_loss, loss_im, loss_3d_s # , loss_pts

    def detaching_grad_of_outputs(self, output_dict):
        
        for ii in range(0, len(output_dict['flow_f'])):
            output_dict['flow_f'][ii].detach_()
            output_dict['flow_b'][ii].detach_()
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()

        return None

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        ## SceneFlow Loss
        batch_size = target_dict['input_l1'].size(0)
        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_sf_2d = 0
        # loss_sf_3d = 0
        loss_sf_sm = 0
        
        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

            assert(sf_f.size()[2:4] == sf_b.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
            
            ## For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

            ## Depth Loss
            loss_disp_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


            ## Sceneflow Loss           
            loss_sceneflow, loss_im, loss_3d_s = self.sceneflow_loss(sf_f, sf_b, 
                                                                            disp_l1, disp_l2,
                                                                            k_l1_aug, k_l2_aug,
                                                                            img_l1_aug, img_l2_aug, 
                                                                            aug_size, ii)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
            loss_sf_2d = loss_sf_2d + loss_im            
            # loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s

        # finding weight
        f_loss = loss_sf_sum.detach()
        d_loss = loss_dp_sum.detach()
        max_val = torch.max(f_loss, d_loss)
        f_weight = max_val / f_loss
        d_weight = max_val / d_loss

        total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["sf"] = loss_sf_sum
        loss_dict["s_2"] = loss_sf_2d
        # loss_dict["s_3"] = loss_sf_3d
        loss_dict["s_3s"] = loss_sf_sm
        loss_dict["total_loss"] = total_loss

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict


###############################################
## Ablation - Separate Decoder
###############################################

class Loss_Flow_Only(nn.Module):
    def __init__(self):
        super(Loss_Flow_Only, self).__init__()

        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._warping_layer = WarpingLayer_Flow()

    def forward(self, output_dict, target_dict):

        ## Loss
        total_loss = 0
        loss_sf_2d = 0
        loss_sf_sm = 0

        for ii, (sf_f, sf_b) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'])):

            ## Depth2Pts            
            img_l1 = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2 = interpolate2d_as(target_dict["input_l2_aug"], sf_b)

            img_l2_warp = self._warping_layer(img_l2, sf_f)
            img_l1_warp = self._warping_layer(img_l1, sf_b)
            occ_map_f = _adaptive_disocc_detection(sf_b).detach()
            occ_map_b = _adaptive_disocc_detection(sf_f).detach()

            img_diff1 = (_elementwise_l1(img_l1, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
            img_diff2 = (_elementwise_l1(img_l2, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
            loss_im1 = img_diff1[occ_map_f].mean()
            loss_im2 = img_diff2[occ_map_b].mean()
            img_diff1[~occ_map_f].detach_()
            img_diff2[~occ_map_b].detach_()
            loss_im = loss_im1 + loss_im2

            loss_smooth = _smoothness_motion_2nd(sf_f / 20.0, img_l1, beta=10.0).mean() + _smoothness_motion_2nd(sf_b / 20.0, img_l2, beta=10.0).mean()
            
            total_loss = total_loss + (loss_im + 10.0 * loss_smooth) * self._weights[ii]
            
            loss_sf_2d = loss_sf_2d + loss_im 
            loss_sf_sm = loss_sf_sm + loss_smooth

        loss_dict = {}
        loss_dict["ofd2"] = loss_sf_2d
        loss_dict["ofs2"] = loss_sf_sm
        loss_dict["total_loss"] = total_loss

        return loss_dict

class Eval_Flow_Only(nn.Module):
    def __init__(self):
        super(Eval_Flow_Only, self).__init__()
    

    def upsample_flow_as(self, flow, output_as):
        size_inputs = flow.size()[2:4]
        size_targets = output_as.size()[2:4]
        resized_flow = tf.interpolate(flow, size=size_targets, mode="bilinear", align_corners=True)
        # correct scaling of flow
        u, v = resized_flow.chunk(2, dim=1)
        u *= float(size_targets[1] / size_inputs[1])
        v *= float(size_targets[0] / size_inputs[0])
        return torch.cat([u, v], dim=1)


    def forward(self, output_dict, target_dict):

        loss_dict = {}

        im_l1 = target_dict['input_l1']
        batch_size, _, _, _ = im_l1.size()

        gt_flow = target_dict['target_flow']
        gt_flow_mask = target_dict['target_flow_mask']

        ## Flow EPE
        out_flow = self.upsample_flow_as(output_dict['flow_f'][0], gt_flow)
        valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask.float()
        loss_dict["epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
        
        flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
        outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
        loss_dict["f1"] = (outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68

        output_dict["out_flow_pp"] = out_flow

        return loss_dict


class Loss_Disp_Only(nn.Module):
    def __init__(self, args):
        super(Loss_Disp_Only, self).__init__()
                
        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_smooth_w = 0.1


    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

        ## Image loss: 
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
        loss_img = (img_diff[left_occ]).mean()
        img_diff[~left_occ].detach_()

        ## Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        return loss_img + self._disp_smooth_w * loss_smooth, left_occ

    def detaching_grad_of_outputs(self, output_dict):
        
        for ii in range(0, len(output_dict['disp_l1'])):
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()

        return None

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        ## SceneFlow Loss
        batch_size = target_dict['input_l1'].size(0)
        loss_dp_sum = 0

        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        for ii, (disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

            assert(disp_l1.size()[2:4] == disp_l2.size()[2:4])
            
            ## For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], disp_l1)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], disp_l2)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], disp_l1)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], disp_l2)

            ## Depth Loss
            loss_disp_l1, _ = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2, _ = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]

        total_loss = loss_dp_sum

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["total_loss"] = total_loss

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict

class Eval_Disp_Only(nn.Module):
    def __init__(self):
        super(Eval_Disp_Only, self).__init__()

        self.verify_lidar_depth = False

    def forward(self, output_dict, target_dict):
        

        loss_dict = {}

        ## Depth Eval
        gt_disp = target_dict['target_disp']
        gt_disp_mask = (target_dict['target_disp_mask']==1)
        intrinsics = target_dict['input_k_l1']

        out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * gt_disp.size(3)
        out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
        out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
        gt_depth_pp = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

        output_dict_displ = eval_module_disp_depth(gt_disp, gt_disp_mask, out_disp_l1, gt_depth_pp, out_depth_l1)

        output_dict["out_disp_l_pp"] = out_disp_l1
        output_dict["out_depth_l_pp"] = out_depth_l1

        loss_dict["d1"] = output_dict_displ['otl']

        loss_dict["ab"] = output_dict_displ['abs_rel']
        loss_dict["sq"] = output_dict_displ['sq_rel']
        loss_dict["rms"] = output_dict_displ['rms']
        loss_dict["lrms"] = output_dict_displ['log_rms']
        loss_dict["a1"] = output_dict_displ['a1']
        loss_dict["a2"] = output_dict_displ['a2']
        loss_dict["a3"] = output_dict_displ['a3']

        if self.verify_lidar_depth:
            ### check depth consistency
            gt_depth = target_dict["target_depth_l1"]
            gt_depth_mask = (target_dict["target_depth_mask_l1"]==1)
            gt_depth_disp_mask = gt_depth_mask & gt_disp_mask
            
            ## MonoDepth metric
            abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = compute_errors(gt_depth[gt_depth_disp_mask], out_depth_l1[gt_depth_disp_mask])        
            loss_dict["ab_"] = abs_rel
            loss_dict["sq_"] = sq_rel
            loss_dict["rms_"] = rms
            loss_dict["lrms_"] = log_rms
            loss_dict["a1_"] = a1
            loss_dict["a2_"] = a2
            loss_dict["a3_"] = a3

            gt_depth_masked = gt_depth.clone()
            gt_depth_masked[~gt_depth_disp_mask] = 0
            gt_depth_pp_masked = gt_depth_pp.clone()
            gt_depth_pp_masked[~gt_depth_disp_mask] = 0
            img = target_dict["input_l1"]

            for ib in range(gt_depth_masked.shape[0]):
                c3d.utils_general.dep_img_bw(gt_depth_mask[ib], path="/home/minghanz/repos/self-mono-sf/debug_dep", name="dep_mask_{}.jpg".format(target_dict["index"][ib]))
                c3d.utils_general.dep_img_bw(gt_disp_mask[ib], path="/home/minghanz/repos/self-mono-sf/debug_dep", name="disp_mask_{}.jpg".format(target_dict["index"][ib]))
                c3d.utils_general.dep_img_bw(gt_depth_disp_mask[ib], path="/home/minghanz/repos/self-mono-sf/debug_dep", name="dd_mask_{}.jpg".format(target_dict["index"][ib]))

                c3d.utils_general.dep_img_bw(gt_depth[ib], path="/home/minghanz/repos/self-mono-sf/debug_dep", name="depth_gt_{}.jpg".format(target_dict["index"][ib]))
                c3d.utils_general.dep_img_bw(gt_disp[ib], path="/home/minghanz/repos/self-mono-sf/debug_dep", name="disp_gt_{}.jpg".format(target_dict["index"][ib]))

                dep_diff = c3d.utils_general.vis_depth_err(gt_depth_pp_masked[ib], gt_depth_masked[ib])
                img_np = c3d.utils_general.uint8_np_from_img_tensor(img[ib])
                c3d.utils_general.overlay_dep_on_rgb_np(dep_diff, img_np, path="/home/minghanz/repos/self-mono-sf/debug_dep", name="depdiff_{}.jpg".format(target_dict["index"][ib]), overlay=True)

            err = gt_depth_pp_masked[gt_depth_disp_mask] - gt_depth_masked[gt_depth_disp_mask]
            logging.info("err max {} min {}, mean_abs {}".format(err.max(), err.min(), torch.abs(err).mean() ) )


        return loss_dict


###############################################
## MonoDepth Experiment
###############################################

class Basis_MonoDepthLoss(nn.Module):
    def __init__(self):
        super(Basis_MonoDepthLoss, self).__init__()
        self.ssim_w = 0.85
        self.disp_gradient_w = 0.1
        self.lr_w = 1.0
        self.n = 4

        torch_vs = torch.__version__
        digits = torch_vs.split(".")
        torch_vs_n = float(digits[0]) + float(digits[1]) * 0.1
        self.grid_sample_specify_align_flag = torch_vs_n >= 1.3  

    def scale_pyramid(self, img_input, depths):
        scaled_imgs = []
        for _, depth in enumerate(depths):
            scaled_imgs.append(interpolate2d_as(img_input, depth))
        return scaled_imgs

    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = tf.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = tf.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def apply_disparity(self, img, disp):
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        if self.grid_sample_specify_align_flag:
            output = tf.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros', align_corners=True)
        else:
            output = tf.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')

        return output

    def generate_image_left(self, img, disp):
        return self.apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        return self.apply_disparity(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        SSIM_img = torch.clamp((1 - SSIM) / 2, 0, 1)

        return tf.pad(SSIM_img, pad=(1,1,1,1), mode='constant', value=0)

    def disp_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(self.n)]

        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i]) for i in range(self.n)]

    def forward(self, disp_l, disp_r, img_l, img_r):

        self.n = len(disp_l)

        ## Image pyramid
        img_l_pyramid = self.scale_pyramid(img_l, disp_l)
        img_r_pyramid = self.scale_pyramid(img_r, disp_r)

        ## Disocc map
        right_occ = [_adaptive_disocc_detection_disp(-disp_l[i]) for i in range(self.n)]
        left_occ  = [_adaptive_disocc_detection_disp(disp_r[i]) for i in range(self.n)]


        ## Image reconstruction loss
        left_est = [self.generate_image_left(img_r_pyramid[i], disp_l[i]) for i in range(self.n)]
        right_est = [self.generate_image_right(img_l_pyramid[i], disp_r[i]) for i in range(self.n)]

        # L1
        l1_left = [torch.mean((torch.abs(left_est[i] - img_l_pyramid[i])).mean(dim=1, keepdim=True)[left_occ[i]]) for i in range(self.n)]
        l1_right = [torch.mean((torch.abs(right_est[i] - img_r_pyramid[i])).mean(dim=1, keepdim=True)[right_occ[i]]) for i in range(self.n)]

        # SSIM
        ssim_left = [torch.mean((self.SSIM(left_est[i], img_l_pyramid[i])).mean(dim=1, keepdim=True)[left_occ[i]]) for i in range(self.n)]
        ssim_right = [torch.mean((self.SSIM(right_est[i], img_r_pyramid[i])).mean(dim=1, keepdim=True)[right_occ[i]]) for i in range(self.n)]

        image_loss_left = [self.ssim_w * ssim_left[i] + (1 - self.ssim_w) * l1_left[i] for i in range(self.n)]
        image_loss_right = [self.ssim_w * ssim_right[i] + (1 - self.ssim_w) * l1_right[i] for i in range(self.n)]
        image_loss = sum(image_loss_left + image_loss_right)


        ## L-R Consistency loss
        right_left_disp = [self.generate_image_left(disp_r[i], disp_l[i]) for i in range(self.n)]
        left_right_disp = [self.generate_image_right(disp_l[i], disp_r[i]) for i in range(self.n)]

        lr_left_loss = [torch.mean((torch.abs(right_left_disp[i] - disp_l[i]))[left_occ[i]]) for i in range(self.n)]
        lr_right_loss = [torch.mean((torch.abs(left_right_disp[i] - disp_r[i]))[right_occ[i]]) for i in range(self.n)]
        lr_loss = sum(lr_left_loss + lr_right_loss)


        ## Disparities smoothness
        disp_left_smoothness = self.disp_smoothness(disp_l, img_l_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_r, img_r_pyramid)

        disp_left_loss = [torch.mean(torch.abs(disp_left_smoothness[i])) / 2 ** i for i in range(self.n)]
        disp_right_loss = [torch.mean(torch.abs(disp_right_smoothness[i])) / 2 ** i for i in range(self.n)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)


        ## Loss sum
        loss = image_loss + self.disp_gradient_w * disp_gradient_loss + self.lr_w * lr_loss

        return loss

class Loss_MonoDepth(nn.Module):
    def __init__(self):

        super(Loss_MonoDepth, self).__init__()
        self._depth_loss = Basis_MonoDepthLoss()

    def forward(self, output_dict, target_dict):

        loss_dict = {}
        depth_loss = self._depth_loss(output_dict['disp_l1'], output_dict['disp_r1'], target_dict['input_l1'], target_dict['input_r1'])
        loss_dict['total_loss'] = depth_loss

        return loss_dict

class Eval_MonoDepth(nn.Module):
    def __init__(self):
        super(Eval_MonoDepth, self).__init__()

    def forward(self, output_dict, target_dict):
        
        loss_dict = {}

        ## Depth Eval
        gt_disp = target_dict['target_disp']
        gt_disp_mask = (target_dict['target_disp_mask']==1)
        intrinsics = target_dict['input_k_l1_orig']

        out_disp_l_pp = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * gt_disp.size(3)
        out_depth_l_pp = _disp2depth_kitti_K(out_disp_l_pp, intrinsics[:, 0, 0])
        out_depth_l_pp = torch.clamp(out_depth_l_pp, 1e-3, 80)
        gt_depth_pp = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

        output_dict_displ = eval_module_disp_depth(gt_disp, gt_disp_mask, out_disp_l_pp, gt_depth_pp, out_depth_l_pp)

        output_dict["out_disp_l_pp"] = out_disp_l_pp
        output_dict["out_depth_l_pp"] = out_depth_l_pp
        loss_dict["ab_r"] = output_dict_displ['abs_rel']
        loss_dict["sq_r"] = output_dict_displ['sq_rel']

        return loss_dict

###############################################

