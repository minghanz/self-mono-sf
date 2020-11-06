import torch
import os
import numpy as np

import torch.utils.data as data

from torchvision import transforms as vision_transforms
from common import read_image_as_byte, read_calib_into_dict, read_png_flow, read_png_disp, numpy2torch
from common import kitti_crop_image_list, kitti_adjust_intrinsic, intrinsic_scale, get_date_from_width
from common import list_flatten
from common import read_raw_calib_file
import skimage.io as io
import skimage
import scipy
from pypardiso import spsolve
import imageio

import sys
sys.path.append("/home/minghanz/repos/self-mono-sf/")
from models.modules_sceneflow import WarpingLayer_Flow
from models.forwardwarp_package.forward_warp import ForwardWarpDWeight
from models.modules_sceneflow import get_grid, WarpingLayer_SF

def read_calib_ext(path):
    file_data = read_raw_calib_file(path)
    R = file_data['R'].reshape((3, 3))
    t = file_data['T'].reshape((3, 1))
    T = np.concatenate([R, t], axis=1)
    T = np.concatenate([T, np.array([[0,0,0,1.0]])], axis=0)
    return R, t, T

#
# fill_depth_colorization.m
# Preprocesses the kinect depth image using a gray scale version of the
# RGB image as a weighting for the smoothing. This code is a slight
# adaptation of Anat Levin's colorization code:
#
# See: www.cs.huji.ac.il/~yweiss/Colorization/
#
# Args:
#  imgRgb - HxWx3 matrix, the rgb image for the current frame. This must
#      be between 0 and 1.
#  imgDepth - HxW matrix, the depth image for the current frame in
#       absolute (meters) space.
#  alpha - a penalty value between 0 and 1 for the current depth values.

def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
	imgIsNoise = imgDepthInput == 0
	maxImgAbsDepth = np.max(imgDepthInput)
	imgDepth = imgDepthInput / maxImgAbsDepth
	imgDepth[imgDepth > 1] = 1
	(H, W) = imgDepth.shape
	numPix = H * W
	indsM = np.arange(numPix).reshape((W, H)).transpose()
	knownValMask = (imgIsNoise == False).astype(int)
	grayImg = skimage.color.rgb2gray(imgRgb)
	winRad = 1
	len_ = 0
	absImgNdx = 0
	len_window = (2 * winRad + 1) ** 2
	len_zeros = numPix * len_window

	cols = np.zeros(len_zeros) - 1
	rows = np.zeros(len_zeros) - 1
	vals = np.zeros(len_zeros) - 1
	gvals = np.zeros(len_window) - 1

	for j in range(W):
		for i in range(H):
			nWin = 0
			for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
				for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
					if ii == i and jj == j:
						continue

					rows[len_] = absImgNdx
					cols[len_] = indsM[ii, jj]
					gvals[nWin] = grayImg[ii, jj]

					len_ = len_ + 1
					nWin = nWin + 1

			curVal = grayImg[i, j]
			gvals[nWin] = curVal
			c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin+ 1])) ** 2)

			csig = c_var * 0.6
			mgv = np.min((gvals[:nWin] - curVal) ** 2)
			if csig < -mgv / np.log(0.01):
				csig = -mgv / np.log(0.01)

			if csig < 2e-06:
				csig = 2e-06

			gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
			gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
			vals[len_ - nWin:len_] = -gvals[:nWin]

	  		# Now the self-reference (along the diagonal).
			rows[len_] = absImgNdx
			cols[len_] = absImgNdx
			vals[len_] = 1  # sum(gvals(1:nWin))

			len_ = len_ + 1
			absImgNdx = absImgNdx + 1

	vals = vals[:len_]
	cols = cols[:len_]
	rows = rows[:len_]
	A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	rows = np.arange(0, numPix)
	cols = np.arange(0, numPix)
	vals = (knownValMask * alpha).transpose().reshape(numPix)
	G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	A = A + G
	b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

	#print ('Solving system..')

	new_vals = spsolve(A, b)
	new_vals = np.reshape(new_vals, (H, W), 'F')

	#print ('Done.')

	denoisedDepthImg = new_vals * maxImgAbsDepth
    
	output = denoisedDepthImg.reshape((H, W)).astype('float32')

	output = np.multiply(output, (1-knownValMask)) + imgDepthInput
    
	return output

if __name__ == "__main__":
    ### load rgb, depth (disparity), pose, flow
    root_sceneflow = "/scratch/hpeng_root/hpeng1/minghanz/scene_flow/kitti_scene_flow_single"
    root_raw = "/scratch/hpeng_root/hpeng1/minghanz/kitti_data_all/2011_09_29/2011_09_29_drive_0004_sync"

    idx_sceneflow = 157
    idx_raw = 36

    rgb_0_path = "/scratch/hpeng_root/hpeng1/minghanz/scene_flow/kitti_scene_flow_single/training/image_2/000157_10.png"
    rgb_1_path = "/scratch/hpeng_root/hpeng1/minghanz/scene_flow/kitti_scene_flow_single/training/image_2/000157_11.png"

    disp_0_path = "/scratch/hpeng_root/hpeng1/minghanz/scene_flow/kitti_scene_flow_single/training/disp_occ_0/000157_10.png"
    disp_1_path = "/scratch/hpeng_root/hpeng1/minghanz/scene_flow/kitti_scene_flow_single/training/disp_occ_1/000157_10.png"

    flow_0_path = "/scratch/hpeng_root/hpeng1/minghanz/scene_flow/kitti_scene_flow_single/training/flow_occ/000157_10.png"

    calib_velo_cam_path = "/scratch/hpeng_root/hpeng1/minghanz/scene_flow/kitti_scene_flow_single/training/calib_velo_to_cam/000157.txt"
    calib_imu_velo_path = "/scratch/hpeng_root/hpeng1/minghanz/scene_flow/kitti_scene_flow_single/training/calib_imu_to_velo/000157.txt"

    depth_0_path = "/scratch/hpeng_root/hpeng1/minghanz/kitti_data_all/2011_09_29/2011_09_29_drive_0004_sync/depth_completed/DeepLidar/image_02/0000000036.png"
    depth_1_path = "/scratch/hpeng_root/hpeng1/minghanz/kitti_data_all/2011_09_29/2011_09_29_drive_0004_sync/depth_completed/DeepLidar/image_02/0000000037.png"

    pose_path = "/scratch/hpeng_root/hpeng1/minghanz/kitti_data_all/2011_09_29/2011_09_29_drive_0004_sync/poses/cam_02.txt"

    rgb_0 = read_image_as_byte(rgb_0_path)      # read into <class 'imageio.core.util.Array'>(which is a subclass of np.ndarray) of int max 255 (isinstance(rgb_0, np.ndarray)==True)
    rgb_1 = read_image_as_byte(rgb_1_path)

    flow_0, flow_mask_0 = read_png_flow(flow_0_path)

    disp_0, disp_mask_0 = read_png_disp(disp_0_path)
    disp_1, disp_mask_1 = read_png_disp(disp_1_path)
    
    depth_0, depth_mask_0 = read_png_disp(depth_0_path)
    depth_1, depth_mask_1 = read_png_disp(depth_1_path)

    disp_0 = disp_0.astype(np.float64)
    disp_1 = disp_1.astype(np.float64)
    depth_0 = depth_0.astype(np.float64)
    depth_1 = depth_1.astype(np.float64)

    ## loading camera intrinsic matrix
    path_dir = os.path.dirname(os.path.realpath(__file__))
    intrinsic_dict_l, intrinsic_dict_r = read_calib_into_dict(path_dir)

    k_l1 = intrinsic_dict_l["2011_09_29"]
    k_r1 = intrinsic_dict_r["2011_09_29"]
    
    ## load cam-lidar and lidar-imu extrinsic matrix
    R_cam_velo, t_cam_velo, T_cam_velo = read_calib_ext(calib_velo_cam_path)
    R_velo_imu, t_velo_imu, T_velo_imu = read_calib_ext(calib_imu_velo_path)

    T_cam_imu = T_cam_velo.dot(T_velo_imu)
    R_cam_imu = T_cam_imu[:3, :3]
    t_cam_imu = T_cam_imu[:3, 3:]

    # input size
    h_orig, w_orig, _ = rgb_0.shape
    input_im_size = np.array([h_orig, w_orig])

    ## load pose
    poses = np.loadtxt(pose_path)
    pose_0 = poses[idx_raw].reshape((3,4))
    pose_1 = poses[idx_raw+1].reshape((3,4))

    pose_0 = np.concatenate([pose_0, np.array([[0,0,0,1.0]])], axis=0)
    pose_1 = np.concatenate([pose_1, np.array([[0,0,0,1.0]])], axis=0)
    
    pose_01 = np.linalg.inv(pose_0).dot(pose_1)
    pose_10 = np.linalg.inv(pose_1).dot(pose_0)
    print(pose_01)

    ### fill the disparity and flow, which is originally semi-dense. 
    disp_filled_0 = fill_depth_colorization(imgRgb=rgb_0, imgDepthInput=disp_0[...,0])
    disp_filled_1 = fill_depth_colorization(imgRgb=rgb_0, imgDepthInput=disp_1[...,0])
    depth_filled_0 = k_l1[0,0] * 0.54 / (disp_filled_0 + 1e-8)
    depth_filled_1 = k_l1[0,0] * 0.54 / (disp_filled_1 + 1e-8)

    depth_filled_0 = np.clip(depth_filled_0, a_min=1e-3, a_max=80).reshape((h_orig, w_orig, 1))
    depth_filled_1 = np.clip(depth_filled_1, a_min=1e-3, a_max=80).reshape((h_orig, w_orig, 1))
    
    # disp_filled_0_out = (disp_filled_0 * 256).astype(np.uint16))
    # print(type(disp_filled_0_out))
    # print(disp_filled_0_out.shape)
    # print(disp_filled_0_out.dtype)

    flow_filled_ch0 = fill_depth_colorization(imgRgb=rgb_0, imgDepthInput=flow_0[...,0])
    flow_filled_ch1 = fill_depth_colorization(imgRgb=rgb_0, imgDepthInput=flow_0[...,1])

    flow_filled_0 = np.stack([flow_filled_ch0, flow_filled_ch1], axis=2)
    
    ### both ok
    # imageio.imwrite("disp_0.png", disp_filled_0_out)
    # io.imsave("disp_0.png", disp_filled_0_out)
    # print("image saved")

    ### uv1_grid
    inv_k_l1 = np.linalg.inv(k_l1)
    u_vec = np.arange(w_orig)
    v_vec = np.arange(h_orig)
    u_grid, v_grid = np.meshgrid(u_vec, v_vec)
    uv1_grid = np.stack([u_grid, v_grid, np.ones_like(u_grid)], 0)  # 3*H*W
    xy1_grid = inv_k_l1.dot(uv1_grid.reshape((3, -1))).reshape((3, h_orig, w_orig))
    uv1_grid = uv1_grid.transpose(1, 2, 0)
    xy1_grid = xy1_grid.transpose(1, 2, 0)
    

    input_dict = {}

    input_dict["rgb_0"] = rgb_0
    input_dict["rgb_1"] = rgb_1

    input_dict["disp_0"] = disp_0
    input_dict["disp_1"] = disp_1

    input_dict["depth_0"] = depth_0
    input_dict["depth_1"] = depth_1

    input_dict["disp_mask_0"] = disp_mask_0
    input_dict["disp_mask_1"] = disp_mask_1

    input_dict["depth_mask_0"] = depth_mask_0
    input_dict["depth_mask_1"] = depth_mask_1

    input_dict["pose_01"] = pose_01
    input_dict["pose_10"] = pose_10
    
    input_dict["flow_0"] = flow_0
    input_dict["flow_mask_0"] = flow_mask_0
    
    input_dict["flow_filled_0"] = flow_filled_0

    input_dict["depth_filled_0"] = depth_filled_0
    input_dict["depth_filled_1"] = depth_filled_1

    input_dict['k_l'] = k_l1
    input_dict['inv_k_l'] = inv_k_l1

    input_dict["uv1_grid"] = uv1_grid
    input_dict["xy1_grid"] = xy1_grid

    input_dict["input_im_size"] = input_im_size

    for key in input_dict:
        input_dict[key] = torch.from_numpy(input_dict[key]).float()
        if input_dict[key].ndim == 3:
            input_dict[key] = input_dict[key].permute(2, 0, 1)

    ### scene flow
    # xyz_0 = input_dict["xy1_grid"] * input_dict["depth_0"]
    ## or
    xyz_0 = input_dict["xy1_grid"] * input_dict["depth_filled_0"]
    
    uv1_grid = input_dict["uv1_grid"]
    flow_2D_full = input_dict["flow_filled_0"]
    uv1_grid_flowed = uv1_grid.clone()
    uv1_grid_flowed[:2, ...] = uv1_grid_flowed[:2, ...] + flow_2D_full
    print(uv1_grid.shape)
    print(uv1_grid_flowed.shape)
    print(flow_2D_full.shape)
    print(input_dict['inv_k_l'].shape)
    
    xy1_grid_flowed = torch.matmul(input_dict['inv_k_l'], uv1_grid_flowed.flatten(1)).reshape_as(uv1_grid_flowed)

    # depth_1 = input_dict["depth_1"]
    # warping_2D_layer = WarpingLayer_Flow()
    # depth_1_warped = warping_2D_layer(depth_1, flow_2D_full)
    ## or 
    depth_1_warped = input_dict["depth_filled_1"]
    xyz_1 = xy1_grid_flowed * depth_1_warped

    flow_3D_full = xyz_1 - xyz_0

    ### rigid flow
    pose10 = input_dict["pose_10"]
    xyz_0_homo = torch.cat([xyz_0, torch.ones_like(xyz_0[[0]])], dim=0) # 4*H*W
    xyz_0_trans  = torch.matmul(pose10, xyz_0_homo.flatten(1)).reshape_as(xyz_0_homo)
    flow_3D_rigid = xyz_0_trans[:3] - xyz_0

    uvz_trans = torch.matmul(input_dict['k_l'], xyz_0_trans[:3].flatten(1)).reshape_as(flow_3D_rigid)
    uv_trans = uvz_trans[:2] / uvz_trans[[2]]
    flow_2D_rigid = uv_trans - uv1_grid[:2]

    ### residual flow
    flow_3D_res = flow_3D_full - flow_3D_rigid

    ### dynamic flow
    forward_layer = ForwardWarpDWeight(5)
    flow_3D_dyn = forward_layer(flow_3D_res.unsqueeze(0), flow_2D_rigid.unsqueeze(0), input_dict["depth_filled_0"].unsqueeze(0) )

    depth_0_ori = input_dict["depth_filled_0"].clone()
    forwarded_depth = forward_layer(depth_0_ori.unsqueeze(0), flow_2D_rigid.unsqueeze(0), input_dict["depth_filled_0"].unsqueeze(0))   ###!!! Is this what's missing in current implementation?

    ### warping with the dynamic flow
    warping_3D_layer = WarpingLayer_SF(reg_depth=True)
    rgb_1_static = warping_3D_layer(input_dict["rgb_1"].unsqueeze(0), flow_3D_dyn, forwarded_depth, input_dict['k_l'], input_dict["input_im_size"])

    io.imsave("rgb_1_static.jpg", rgb_1_static[0].permute(1,2,0).numpy())
    print("rgb_1_static.jpg saved")