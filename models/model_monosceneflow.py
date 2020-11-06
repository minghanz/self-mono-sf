from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging

from .correlation_package.correlation import Correlation

from .modules_sceneflow import get_grid, WarpingLayer_SF
from .modules_sceneflow import initialize_msra, upsample_outputs_as
from .modules_sceneflow import upconv
from .modules_sceneflow import FeatureExtractor, MonoSceneFlowDecoder, ContextNetwork

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import flow_horizontal_flip, intrinsic_scale, get_pixelgrid, post_processing

from .modules_sceneflow import RigidFlowFromPose, MonoSceneFlowPoseDecoder, MonoSceneFlowPoseHourGlass, PoseHead, MonoSceneFlowMaskDecoder

from utils.sceneflow_util import pixel2pts_ms, pts2pixel_ms
from utils.sceneflow_util import pixel2pts_ms_and_depth, pixel2pts_ms_from_depth, depth2dispn_kitti_ms

from .modules_monodepth import Resnet18_AllinOne, Resnet18_Pose

from .modules_sceneflow import upsample_outputs_as_enlarge_only
from models.forwardwarp_package.forward_warp import ForwardWarpDWeight
from utils.sceneflow_util import projectSceneFlow2Flow
import torchsnooper
from .modules_sceneflow import WarpingLayer_FlowNormalized, MonoSceneFlowUVDDecoder, uvd2xyz, xyz2uvd

class MonoSceneFlow(nn.Module):
    def __init__(self, args):
        super(MonoSceneFlow, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        
        self.flow_estimators = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        self.hourglass_decoder = args.hourglass_decoder
        self.highres_loss = args.highres_loss

        self.shared_posehead = args.shared_posehead
        self.pose_separable = not args.pose_not_separable

        self.use_posenet = args.use_posenet

        self.pred_xyz_add_uvd = args.pred_xyz_add_uvd
        self.strict_flow_acc_uvd = args.strict_flow_acc_uvd
        self.strict_flow_acc = args.strict_flow_acc
        self.strict_flow_acc_ramp = args.strict_flow_acc_ramp
        self.forward_dweighted = ForwardWarpDWeight(ref_scale=5)

        self.rigid = args.rigid
        self.rigid_pred = args.rigid_pred
        assert self.rigid_pred in ["full", "res"]
        self.rigid_warp = args.rigid_warp
        assert self.rigid_warp in ["full", "res", "rigid"]
        ### pred = "full": rigid flow provides a guidance for rigid regions. 
        ### pred = "res": rigid flow composes flow for rigid regions. 
        ### warp = "full": pose at different levels are not the same. Final flow = total full flow
        ### warp = "res": expect all pose predictions to be the same. Final flow = final rigid flow + total res flow
        ### warp = "rigid": pose at different levels are not the same. Final flow = total rigid flow + final res flow
        self.rigid_pass = args.rigid_pass
        assert self.rigid_pass in ["full", "warp", "pred"]
        self.disable_residual_pred = args.disable_residual_pred

        self.reg_depth = args.reg_depth
        self.warping_layer_sf = WarpingLayer_SF(reg_depth=self.reg_depth)
        if self.strict_flow_acc_uvd:
            self.warping_layer_flow = WarpingLayer_FlowNormalized()

        if self.rigid:
            if self.use_posenet:
                self.pose_net = Resnet18_Pose()
            else:
                if self.shared_posehead:
                    shared_poseheadnet = PoseHead(self.pose_separable)
                else:
                    shared_poseheadnet = None

        if self.hourglass_decoder:
            self.hourglass = Resnet18_AllinOne()
            last_level_chnl = 32
        else:
            last_level_chnl = 32

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + ch 
            else:
                num_ch_in = self.dim_corr + ch + last_level_chnl + 3 + 1
                self.upconv_layers.append(upconv(last_level_chnl, last_level_chnl, 3, 2))

            if self.hourglass_decoder:
                layer_sf = MonoSceneFlowPoseHourGlass(num_ch_in)#, self.hourglass)
            elif self.use_posenet:
                layer_sf = MonoSceneFlowMaskDecoder(num_ch_in)  
            elif self.rigid:
                layer_sf = MonoSceneFlowPoseDecoder(num_ch_in, self.pose_separable, shared_poseheadnet)     
            elif self.strict_flow_acc_uvd:
                layer_sf = MonoSceneFlowUVDDecoder(num_ch_in)
            else:
                layer_sf = MonoSceneFlowDecoder(num_ch_in)            
            self.flow_estimators.append(layer_sf)            

        if self.rigid:
            print("Rigid mode on")
            self.rigid_flow_net = RigidFlowFromPose(reg_depth=self.reg_depth)

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks = ContextNetwork(last_level_chnl + 3 + 1, reg_depth=self.reg_depth)
        self.sigmoid = torch.nn.Sigmoid()
        self.softplus = torch.nn.Softplus()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []

        if self.rigid:
            sceneflows_f_res_n = []
            sceneflows_b_res_n = []
            sceneflows_f_mask = []
            sceneflows_b_mask = []
            sceneflows_f_res_acc = []
            sceneflows_b_res_acc = []
            sceneflows_f_rigid_acc = []
            sceneflows_b_rigid_acc = []
            sceneflows_f_res_mask_acc = []
            sceneflows_b_res_mask_acc = []
            sceneflows_f_rigid_mask_acc = []
            sceneflows_b_rigid_mask_acc = []

            R12s = []
            t12s = []
            R21s = []
            t21s = []
            pose21s = []
            pose12s = []

        if self.strict_flow_acc:
            if self.strict_flow_acc_ramp == 0:
                weight_warped = 1
                weight_original = 0
            else:
                weight_warped = min((input_dict["epoch"]-1) / self.strict_flow_acc_ramp, 1)     # weight_warped = 1 after self.strict_flow_acc_ramp epochs. The first epoch always has weight_warped = 0
                assert weight_warped >= 0, "{} {}".format(input_dict["epoch"], weight_warped)
                weight_original = 1 - weight_warped

        if self.use_posenet:
            x1_raw_scaled = interpolate2d_as(x1_raw, x1_pyramid[self.output_level], mode="bilinear")
            x2_raw_scaled = interpolate2d_as(x2_raw, x1_pyramid[self.output_level], mode="bilinear")

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            if self.hourglass_decoder:
                if l == 0:
                    pass
                else:
                    ### decide how to warp the feature map
                    if self.rigid_warp == "full":
                        flow_f_warp = flow_f_full.clone()
                        flow_b_warp = flow_b_full.clone()
                    elif self.rigid_warp == "res":
                        flow_f_warp = flow_f_res.clone()
                        flow_b_warp = flow_b_res.clone()
                    elif self.rigid_warp == "rigid":
                        flow_f_warp = flow_f_rigid.clone()
                        flow_b_warp = flow_b_rigid.clone()
                    else:
                        raise ValueError("self.rigid_warp not recognized, {}".format(self.rigid_warp))

                    flow_f_warp_ = interpolate2d_as(flow_f_warp, x1, mode="bilinear")
                    flow_b_warp_ = interpolate2d_as(flow_b_warp, x1, mode="bilinear")
            else:
                ### interpolation
                if l == 0:
                    pass
                else:
                    d_l1_ = interpolate2d_as(d_l1_, x1, mode="bilinear")
                    d_l2_ = interpolate2d_as(d_l2_, x1, mode="bilinear")
                    x1_out = self.upconv_layers[l-1](x1_out)
                    x2_out = self.upconv_layers[l-1](x2_out)
                    if self.rigid:
                        flow_f_res = interpolate2d_as(flow_f_res, x1, mode="bilinear")
                        flow_b_res = interpolate2d_as(flow_b_res, x1, mode="bilinear")
                        flow_f_rigid = interpolate2d_as(flow_f_rigid, x1, mode="bilinear")
                        flow_b_rigid = interpolate2d_as(flow_b_rigid, x1, mode="bilinear")

                        flow_f_res_mask = interpolate2d_as(flow_f_res_mask, x1, mode="bilinear")
                        flow_b_res_mask = interpolate2d_as(flow_b_res_mask, x1, mode="bilinear")
                        flow_f_rigid_mask = interpolate2d_as(flow_f_rigid_mask, x1, mode="bilinear")
                        flow_b_rigid_mask = interpolate2d_as(flow_b_rigid_mask, x1, mode="bilinear")
                        if self.rigid_warp == "res" and self.strict_flow_acc:
                            flow_f_res_or = interpolate2d_as(flow_f_res_or, x1, mode="bilinear")
                            flow_b_res_or = interpolate2d_as(flow_b_res_or, x1, mode="bilinear")
                    else:
                        flow_f_full = interpolate2d_as(flow_f_full, x1, mode="bilinear")
                        flow_b_full = interpolate2d_as(flow_b_full, x1, mode="bilinear")
                    flow_f_level_pred = interpolate2d_as(flow_f_level_pred, x1, mode="bilinear")
                    flow_b_level_pred = interpolate2d_as(flow_b_level_pred, x1, mode="bilinear")

                    ### decide how to warp the feature map
                    if self.rigid_warp == "full":
                        flow_f_warp = flow_f_full
                        flow_b_warp = flow_b_full
                    elif self.rigid_warp == "res":
                        if self.strict_flow_acc:
                            flow_f_warp = flow_f_res_or
                            flow_b_warp = flow_b_res_or
                        else:
                            flow_f_warp = flow_f_res
                            flow_b_warp = flow_b_res
                    elif self.rigid_warp == "rigid":
                        flow_f_warp = flow_f_rigid
                        flow_b_warp = flow_b_rigid
                    else:
                        raise ValueError("self.rigid_warp not recognized, {}".format(self.rigid_warp))

                    flow_f_warp_ = flow_f_warp
                    flow_b_warp_ = flow_b_warp

            if l == 0:
                pass
            else:
                ### decide what to pass to next level input
                if self.rigid_pass == "warp":
                    flow_f_pass = flow_f_warp
                    flow_b_pass = flow_b_warp
                elif self.rigid_pass == "pred":
                    flow_f_pass = flow_f_level_pred
                    flow_b_pass = flow_b_level_pred
                else:
                    raise ValueError("self.rigid_pass not recognized, {}".format(self.rigid_pass))

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                if self.strict_flow_acc_uvd:
                    x2_warp = self.warping_layer_flow(x2, flow_f_warp_[:,:2])  
                    x1_warp = self.warping_layer_flow(x1, flow_b_warp_[:,:2])
                else:
                    x2_warp = self.warping_layer_sf(x2, flow_f_warp_, d_l1_, k1, input_dict['aug_size'])  # becuase K can be changing when doing augmentation
                    x1_warp = self.warping_layer_sf(x1, flow_b_warp_, d_l2_, k2, input_dict['aug_size'])    ### ! error before Oct. 30 18:23

            # correlation
            out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
            out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            if self.hourglass_decoder:
                out_corr_relu_f = interpolate2d_as(out_corr_relu_f, x1_pyramid[self.output_level])
                out_corr_relu_b = interpolate2d_as(out_corr_relu_b, x1_pyramid[self.output_level])
                x1 = interpolate2d_as(x1, x1_pyramid[self.output_level])
                x2 = interpolate2d_as(x2, x1_pyramid[self.output_level])
                    

            # monosf estimator
            ### prediction
            if self.hourglass_decoder:
                if l == 0:
                    feature_interm = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1], dim=1))
                    x1_out, flow_f_level_pred, d_l1, pose21, flow_f_level_res_mask = self.hourglass(feature_interm)
                    feature_interm = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2], dim=1))
                    x2_out, flow_b_level_pred, d_l2, pose12, flow_b_level_res_mask = self.hourglass(feature_interm)
                else:
                    feature_interm = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1, x1_out, flow_f_pass, d_l1_], dim=1))
                    x1_out, flow_f_level_pred, d_l1, pose21, flow_f_level_res_mask = self.hourglass(feature_interm)
                    feature_interm = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2, x2_out, flow_b_pass, d_l2_], dim=1))
                    x2_out, flow_b_level_pred, d_l2, pose12, flow_b_level_res_mask = self.hourglass(feature_interm)
            else:
                if l == 0:
                    x1_out, flow_f_level_pred, d_l1, pose21, flow_f_level_res_mask = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1], dim=1))
                    x2_out, flow_b_level_pred, d_l2, pose12, flow_b_level_res_mask = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2], dim=1))
                else:
                    x1_out, flow_f_level_pred, d_l1, pose21, flow_f_level_res_mask = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1, x1_out, flow_f_pass, d_l1_], dim=1))
                    x2_out, flow_b_level_pred, d_l2, pose12, flow_b_level_res_mask = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2, x2_out, flow_b_pass, d_l2_], dim=1))

            if self.use_posenet:
                if l == 0:
                    pose21 = self.pose_net(torch.cat([x1_raw_scaled, x2_raw_scaled], dim=1))
                    pose12 = self.pose_net(torch.cat([x2_raw_scaled, x1_raw_scaled], dim=1))

                    pose_21_0 = pose21
                    pose_12_0 = pose12
                    
                else:
                    # flow_f_warp_raw_scale = interpolate2d_as(flow_f_warp_, x1_pyramid[self.output_level], mode="bilinear")
                    # flow_b_warp_raw_scale = interpolate2d_as(flow_b_warp_, x1_pyramid[self.output_level], mode="bilinear")
                    # d_l1_raw_scale = interpolate2d_as(d_l1_, x1_pyramid[self.output_level], mode="bilinear")
                    # d_l2_raw_scale = interpolate2d_as(d_l2_, x1_pyramid[self.output_level], mode="bilinear")
                    
                    # x2_raw_warp = self.warping_layer_sf(x2_raw_scaled, flow_f_warp_raw_scale, d_l1_raw_scale, k1, input_dict['aug_size'])  # becuase K can be changing when doing augmentation
                    # x1_raw_warp = self.warping_layer_sf(x1_raw_scaled, flow_b_warp_raw_scale, d_l2_raw_scale, k2, input_dict['aug_size'])  # becuase K can be changing when doing augmentation
                    
                    # pose21 = self.pose_net(torch.cat([x1_raw_scaled, x2_raw_warp], dim=1))
                    # pose12 = self.pose_net(torch.cat([x2_raw_scaled, x1_raw_warp], dim=1))

                    pose21 = pose_21_0
                    pose12 = pose_12_0
                    


            # logging.info("\n d_l1.shape: {} d_l2.shape: {}\n".format(d_l1.shape, d_l2.shape))
            if self.reg_depth:
                d_l1_ = self.softplus(d_l1).clamp(min=1e-3, max=80)
                d_l2_ = self.softplus(d_l2).clamp(min=1e-3, max=80)

                depth_l1 = d_l1_
                depth_l2 = d_l2_

                disp_l1, _ = depth2dispn_kitti_ms(depth_l1, k1, input_dict['aug_size'])
                disp_l2, _ = depth2dispn_kitti_ms(depth_l2, k2, input_dict['aug_size'])
                
            else:
                d_l1_ = self.sigmoid(d_l1) * 0.3
                d_l2_ = self.sigmoid(d_l2) * 0.3

                disp_l1 = d_l1_
                disp_l2 = d_l2_
                

            ### process the prediction
            if self.rigid:
                flow_f_level_rigid, depth_l1, k1_scaled, R21, t21 = self.rigid_flow_net(pose21, d_l1_, k1, input_dict['aug_size'])
                flow_b_level_rigid, depth_l2, k2_scaled, R12, t12 = self.rigid_flow_net(pose12, d_l2_, k2, input_dict['aug_size'])

                if self.rigid_pred == "full":
                    flow_f_level_full = flow_f_level_pred
                    flow_b_level_full = flow_b_level_pred
                    flow_f_level_res = flow_f_level_full - flow_f_level_rigid
                    flow_b_level_res = flow_b_level_full - flow_b_level_rigid
                    
                elif self.rigid_pred == "res":
                    flow_f_level_res = flow_f_level_pred
                    flow_b_level_res = flow_b_level_pred

                    flow_zeros_dummy = torch.zeros_like(flow_f_level_res)
                    flow_f_level_res = torch.where(flow_f_level_res_mask>0.5, flow_f_level_res, flow_zeros_dummy)
                    flow_b_level_res = torch.where(flow_b_level_res_mask>0.5, flow_b_level_res, flow_zeros_dummy)
                    
                    flow_f_level_full = flow_f_level_res + flow_f_level_rigid
                    flow_b_level_full = flow_b_level_res + flow_b_level_rigid
                else:
                    raise ValueError("self.rigid_pred not recognized, {}".format(self.rigid_pred))

                flow_f_level_res_n = flow_f_level_res.norm(dim=1, keepdim=True)
                flow_b_level_res_n = flow_b_level_res.norm(dim=1, keepdim=True)

            else:
                flow_f_level_full = flow_f_level_pred
                flow_b_level_full = flow_b_level_pred

            ### accumulate residual prediction
            if self.rigid:
                if l == 0 or self.disable_residual_pred:
                    if self.rigid_warp == "res":
                        if self.strict_flow_acc:
                            disp_l1_px = disp_l1 * disp_l1.shape[3]
                            disp_l2_px = disp_l2 * disp_l2.shape[3]
                            flow_f_level_rigid_optical = projectSceneFlow2Flow(k1_scaled, flow_f_level_rigid, disp_l1_px)
                            flow_b_level_rigid_optical = projectSceneFlow2Flow(k2_scaled, flow_b_level_rigid, disp_l2_px)
                            
                            flow_f_res_or = self.forward_dweighted(flow_f_level_res, flow_f_level_rigid_optical, depth_l1)
                            flow_b_res_or = self.forward_dweighted(flow_b_level_res, flow_b_level_rigid_optical, depth_l2)

                            flow_f_res_or = flow_f_res_or * weight_warped + flow_f_level_res * weight_original
                            flow_b_res_or = flow_b_res_or * weight_warped + flow_b_level_res * weight_original
                            
                    flow_f_res = flow_f_level_res
                    flow_b_res = flow_b_level_res
                    flow_f_rigid = flow_f_level_rigid
                    flow_b_rigid = flow_b_level_rigid

                    mask_zeros = torch.zeros_like(flow_f_level_res_mask)
                    flow_f_res_mask = torch.where(flow_f_level_res_mask > 0.5, flow_f_level_res_mask, mask_zeros).clamp(min=0, max=1)
                    flow_b_res_mask = torch.where(flow_b_level_res_mask > 0.5, flow_b_level_res_mask, mask_zeros).clamp(min=0, max=1)
                    flow_f_rigid_mask = torch.where(flow_f_level_res_mask < 0.5, 1 - flow_f_level_res_mask, mask_zeros).clamp(min=0, max=1)
                    flow_b_rigid_mask = torch.where(flow_b_level_res_mask < 0.5, 1 - flow_b_level_res_mask, mask_zeros).clamp(min=0, max=1)

                else:
                    mask_zeros = torch.zeros_like(flow_f_level_res_mask)
                    flow_f_res_mask_ = torch.where(flow_f_level_res_mask > 0.5, flow_f_level_res_mask, mask_zeros)
                    flow_b_res_mask_ = torch.where(flow_b_level_res_mask > 0.5, flow_b_level_res_mask, mask_zeros)
                    flow_f_rigid_mask_ = torch.where(flow_f_level_res_mask < 0.5, 1 - flow_f_level_res_mask, mask_zeros)
                    flow_b_rigid_mask_ = torch.where(flow_b_level_res_mask < 0.5, 1 - flow_b_level_res_mask, mask_zeros)

                    flow_f_res_mask = torch.max(flow_f_res_mask, flow_f_res_mask_).clamp(min=0, max=1)
                    flow_b_res_mask = torch.max(flow_b_res_mask, flow_b_res_mask_).clamp(min=0, max=1)
                    
                    flow_f_rigid_mask = torch.max(flow_f_rigid_mask, flow_f_rigid_mask_).clamp(min=0, max=1)
                    flow_b_rigid_mask = torch.max(flow_b_rigid_mask, flow_b_rigid_mask_).clamp(min=0, max=1)

                    if self.rigid_warp == "full":
                        flow_f_res = flow_f_res + flow_f_level_res
                        flow_b_res = flow_b_res + flow_b_level_res
                        flow_f_rigid = flow_f_rigid + flow_f_level_rigid
                        flow_b_rigid = flow_b_rigid + flow_b_level_rigid
                    elif self.rigid_warp == "rigid":
                        flow_f_res = flow_f_level_res
                        flow_b_res = flow_b_level_res
                        flow_f_rigid = flow_f_rigid + flow_f_level_rigid
                        flow_b_rigid = flow_b_rigid + flow_b_level_rigid
                    elif self.rigid_warp == "res":
                        if self.strict_flow_acc:
                            disp_l1_px = disp_l1 * disp_l1.shape[3]
                            disp_l2_px = disp_l2 * disp_l2.shape[3]
                            flow_f_level_rigid_optical = projectSceneFlow2Flow(k1_scaled, flow_f_level_rigid, disp_l1_px)
                            flow_b_level_rigid_optical = projectSceneFlow2Flow(k2_scaled, flow_b_level_rigid, disp_l2_px)

                            flow_f_res_or_level = self.forward_dweighted(flow_f_level_res, flow_f_level_rigid_optical, depth_l1)
                            flow_b_res_or_level = self.forward_dweighted(flow_b_level_res, flow_b_level_rigid_optical, depth_l2)
                            flow_f_res_or_append = self.warping_layer_sf(flow_f_res_or, flow_f_res_or_level, d_l1_, k1, input_dict['aug_size'] )
                            flow_b_res_or_append = self.warping_layer_sf(flow_b_res_or, flow_b_res_or_level, d_l2_, k2, input_dict['aug_size'] )
                            flow_f_res_append = self.warping_layer_sf(flow_f_res_or, flow_f_level_full, d_l1_, k1, input_dict['aug_size'] )
                            flow_b_res_append = self.warping_layer_sf(flow_b_res_or, flow_b_level_full, d_l2_, k2, input_dict['aug_size'] )
                            flow_f_res_or_ = flow_f_res_or_level + flow_f_res_or_append
                            flow_b_res_or_ = flow_b_res_or_level + flow_b_res_or_append
                            flow_f_res_ = flow_f_level_res + flow_f_res_append
                            flow_b_res_ = flow_b_level_res + flow_b_res_append

                            flow_f_res_ori = flow_f_level_res + flow_f_res
                            flow_b_res_ori = flow_b_level_res + flow_b_res

                            flow_f_res = flow_f_res_ * weight_warped + flow_f_res_ori * weight_original
                            flow_b_res = flow_b_res_ * weight_warped + flow_b_res_ori * weight_original
                            
                            flow_f_res_or = flow_f_res_or_ * weight_warped + flow_f_res_ori * weight_original
                            flow_b_res_or = flow_b_res_or_ * weight_warped + flow_b_res_ori * weight_original
                            
                        else:
                            flow_f_res = flow_f_res + flow_f_level_res
                            flow_b_res = flow_b_res + flow_b_level_res
                        flow_f_rigid = flow_f_level_rigid
                        flow_b_rigid = flow_b_level_rigid 
                    else:
                        raise ValueError("self.rigid_warp not recognized, {}".format(self.rigid_warp))
                flow_f_full = flow_f_rigid + flow_f_res
                flow_b_full = flow_b_rigid + flow_b_res
            else:
                if l == 0 or self.disable_residual_pred:
                    flow_f_full = flow_f_level_full
                    flow_b_full = flow_b_level_full
                else:
                    if self.pred_xyz_add_uvd:
                        flow_f_full_uvd = xyz2uvd(flow_f_full, disp_l1, k1, input_dict['aug_size'])
                        flow_f_level_full_uvd = xyz2uvd(flow_f_level_full, disp_l1, k1, input_dict['aug_size'])
                        flow_b_full_uvd = xyz2uvd(flow_b_full, disp_l2, k2, input_dict['aug_size'])
                        flow_b_level_full_uvd = xyz2uvd(flow_b_level_full, disp_l2, k2, input_dict['aug_size'])
                        if self.strict_flow_acc:
                            flow_f_full_uvd_ = self.warping_layer_sf(flow_f_full_uvd, flow_f_level_full, d_l1_, k1, input_dict['aug_size'])
                            flow_b_full_uvd_ = self.warping_layer_sf(flow_b_full_uvd, flow_b_level_full, d_l2_, k2, input_dict['aug_size'])

                            flow_f_full_uvd = flow_f_full_uvd_ * weight_warped + flow_f_full_uvd * weight_original
                            flow_b_full_uvd = flow_b_full_uvd_ * weight_warped + flow_b_full_uvd * weight_original

                        flow_f_full_uvd = flow_f_full_uvd + flow_f_level_full_uvd
                        flow_b_full_uvd = flow_b_full_uvd + flow_b_level_full_uvd
                        
                        flow_f_full = uvd2xyz(flow_f_full_uvd, disp_l1, k1, input_dict['aug_size'])
                        flow_b_full = uvd2xyz(flow_b_full_uvd, disp_l2, k2, input_dict['aug_size'])

                    if self.strict_flow_acc:
                        if self.strict_flow_acc_uvd:
                            flow_f_full_ = self.warping_layer_flow(flow_f_full, flow_f_level_full[:, :2])
                            flow_b_full_ = self.warping_layer_flow(flow_b_full, flow_b_level_full[:, :2]) 
                        else:
                            flow_f_full_ = self.warping_layer_sf(flow_f_full, flow_f_level_full, d_l1_, k1, input_dict['aug_size'])
                            flow_b_full_ = self.warping_layer_sf(flow_b_full, flow_b_level_full, d_l2_, k2, input_dict['aug_size'])
                        
                        flow_f_full = flow_f_full_ * weight_warped + flow_f_full * weight_original
                        flow_b_full = flow_b_full_ * weight_warped + flow_b_full * weight_original

                    flow_f_full = flow_f_full + flow_f_level_full
                    flow_b_full = flow_b_full + flow_b_level_full

                if self.strict_flow_acc_uvd:
                    flow_f_full_xyz = uvd2xyz(flow_f_full, disp_l1, k1, input_dict['aug_size'])
                    flow_b_full_xyz = uvd2xyz(flow_b_full, disp_l2, k2, input_dict['aug_size'])
                    
                        
                    

            # upsampling or post-processing
            if self.rigid:
                sceneflows_f_res_n.append(flow_f_level_res_n)
                sceneflows_b_res_n.append(flow_b_level_res_n)
                sceneflows_f_mask.append(flow_f_level_res_mask)
                sceneflows_b_mask.append(flow_b_level_res_mask)
                sceneflows_f_res_acc.append(flow_f_res)
                sceneflows_b_res_acc.append(flow_b_res)
                sceneflows_f_rigid_acc.append(flow_f_rigid)
                sceneflows_b_rigid_acc.append(flow_b_rigid)
                sceneflows_f_res_mask_acc.append(flow_f_res_mask)
                sceneflows_b_res_mask_acc.append(flow_b_res_mask)
                sceneflows_f_rigid_mask_acc.append(flow_f_rigid_mask)
                sceneflows_b_rigid_mask_acc.append(flow_b_rigid_mask)

                R12s.append(R12)
                t12s.append(t12)
                R21s.append(R21)
                t21s.append(t21)
                pose21s.append(pose21)
                pose12s.append(pose12)
                
                
            if l != self.output_level:
                if self.strict_flow_acc_uvd:
                    sceneflows_f.append(flow_f_full_xyz)
                    sceneflows_b.append(flow_b_full_xyz)  
                else:
                    sceneflows_f.append(flow_f_full)
                    sceneflows_b.append(flow_b_full)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
            else:
                flow_res_f, d_l1 = self.context_networks(torch.cat([x1_out, flow_f_full, d_l1], dim=1))
                flow_res_b, d_l2 = self.context_networks(torch.cat([x2_out, flow_b_full, d_l2], dim=1))
                if self.reg_depth:
                    depth_l1 = d_l1
                    depth_l2 = d_l2
                    disp_l1, _ = depth2dispn_kitti_ms(depth_l1, k1, input_dict['aug_size'])
                    disp_l2, _ = depth2dispn_kitti_ms(depth_l2, k2, input_dict['aug_size'])
                else:
                    disp_l1 = d_l1
                    disp_l2 = d_l2

                if self.strict_flow_acc_uvd:
                    flow_res_f = self.sigmoid(flow_res_f) * 0.3
                    flow_res_b = self.sigmoid(flow_res_b) * 0.3
                    flow_res_f = 2 * flow_res_f - 0.3
                    flow_res_b = 2 * flow_res_b - 0.3
                    flow_f_full = flow_f_full + flow_res_f
                    flow_b_full = flow_b_full + flow_res_b
                    flow_f_full_xyz = uvd2xyz(flow_f_full, disp_l1, k1, input_dict['aug_size'])
                    flow_b_full_xyz = uvd2xyz(flow_b_full, disp_l2, k2, input_dict['aug_size'])
                    sceneflows_f.append(flow_f_full_xyz)
                    sceneflows_b.append(flow_b_full_xyz)
                else:
                    flow_f_full = flow_f_full + flow_res_f
                    flow_b_full = flow_b_full + flow_res_b
                    sceneflows_f.append(flow_f_full)
                    sceneflows_b.append(flow_b_full)

                disps_1.append(disp_l1)
                disps_2.append(disp_l2)                
                break

        if self.hourglass_decoder and self.highres_loss:
            x1_rev = x1_pyramid[::-1]

            output_dict['flow_f'] = upsample_outputs_as_enlarge_only(sceneflows_f[::-1], x1_rev)
            output_dict['flow_b'] = upsample_outputs_as_enlarge_only(sceneflows_b[::-1], x1_rev)
            output_dict['disp_l1'] = upsample_outputs_as_enlarge_only(disps_1[::-1], x1_rev)
            output_dict['disp_l2'] = upsample_outputs_as_enlarge_only(disps_2[::-1], x1_rev)

            if self.rigid:
                output_dict['flow_f_res_norm'] = upsample_outputs_as_enlarge_only(sceneflows_f_res_n[::-1], x1_rev)
                output_dict['flow_b_res_norm'] = upsample_outputs_as_enlarge_only(sceneflows_b_res_n[::-1], x1_rev)
                output_dict['flow_f_res_mask'] = upsample_outputs_as_enlarge_only(sceneflows_f_mask[::-1], x1_rev)
                output_dict['flow_b_res_mask'] = upsample_outputs_as_enlarge_only(sceneflows_b_mask[::-1], x1_rev)

                output_dict['flow_f_res_acc'] = upsample_outputs_as_enlarge_only(sceneflows_f_res_acc[::-1], x1_rev)
                output_dict['flow_b_res_acc'] = upsample_outputs_as_enlarge_only(sceneflows_b_res_acc[::-1], x1_rev)
                output_dict['flow_f_rigid_acc'] = upsample_outputs_as_enlarge_only(sceneflows_f_rigid_acc[::-1], x1_rev)
                output_dict['flow_b_rigid_acc'] = upsample_outputs_as_enlarge_only(sceneflows_b_rigid_acc[::-1], x1_rev)

                output_dict['flow_f_res_mask_acc'] = upsample_outputs_as_enlarge_only(sceneflows_f_res_mask_acc[::-1], x1_rev)
                output_dict['flow_b_res_mask_acc'] = upsample_outputs_as_enlarge_only(sceneflows_b_res_mask_acc[::-1], x1_rev)
                output_dict['flow_f_rigid_mask_acc'] = upsample_outputs_as_enlarge_only(sceneflows_f_rigid_mask_acc[::-1], x1_rev)
                output_dict['flow_b_rigid_mask_acc'] = upsample_outputs_as_enlarge_only(sceneflows_b_rigid_mask_acc[::-1], x1_rev)

        else:
            x1_rev = x1_pyramid[::-1]

            output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
            output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
            output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
            output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)

            if self.rigid:
                output_dict['flow_f_res_norm'] = upsample_outputs_as(sceneflows_f_res_n[::-1], x1_rev)
                output_dict['flow_b_res_norm'] = upsample_outputs_as(sceneflows_b_res_n[::-1], x1_rev)
                output_dict['flow_f_res_mask'] = upsample_outputs_as(sceneflows_f_mask[::-1], x1_rev)
                output_dict['flow_b_res_mask'] = upsample_outputs_as(sceneflows_b_mask[::-1], x1_rev)

                output_dict['flow_f_res_acc'] = upsample_outputs_as(sceneflows_f_res_acc[::-1], x1_rev)
                output_dict['flow_b_res_acc'] = upsample_outputs_as(sceneflows_b_res_acc[::-1], x1_rev)
                output_dict['flow_f_rigid_acc'] = upsample_outputs_as(sceneflows_f_rigid_acc[::-1], x1_rev)
                output_dict['flow_b_rigid_acc'] = upsample_outputs_as(sceneflows_b_rigid_acc[::-1], x1_rev)

                output_dict['flow_f_res_mask_acc'] = upsample_outputs_as(sceneflows_f_res_mask_acc[::-1], x1_rev)
                output_dict['flow_b_res_mask_acc'] = upsample_outputs_as(sceneflows_b_res_mask_acc[::-1], x1_rev)
                output_dict['flow_f_rigid_mask_acc'] = upsample_outputs_as(sceneflows_f_rigid_mask_acc[::-1], x1_rev)
                output_dict['flow_b_rigid_mask_acc'] = upsample_outputs_as(sceneflows_b_rigid_mask_acc[::-1], x1_rev)

        if self.rigid:
            output_dict['R21s'] = R21s[::-1]
            output_dict['t21s'] = t21s[::-1]
            output_dict['R12s'] = R12s[::-1]
            output_dict['t12s'] = t12s[::-1]
            output_dict['pose21s'] = pose21s[::-1]
            output_dict['pose12s'] = pose12s[::-1]
            
            
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])

            output_dict['output_dict_r'] = output_dict_r

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp

        return output_dict
