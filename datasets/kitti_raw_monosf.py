from __future__ import absolute_import, division, print_function

import os.path
import torch
import torch.utils.data as data
import numpy as np

from torchvision import transforms as vision_transforms
from .common import read_image_as_byte, read_calib_into_dict
from .common import kitti_crop_image_list, kitti_adjust_intrinsic

import c3d
from .collate import default_collate_with_camops_caminfo, decollate_with_camops_caminfo

class KITTI_Raw(data.Dataset):
    def __init__(self,
                 args,
                 images_root=None,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[370, 1224],
                 num_examples=-1,
                 index_file=None, 
                 augmentation_instance=None):

        self._args = args
        self._seq_len = 1
        self._flip_augmentations = flip_augmentations
        self._preprocessing_crop = preprocessing_crop
        self._crop_size = crop_size

        ### c3d: move augmentation into dataset to improve the efficiency when c3d loader is used.
        self._augmentation = augmentation_instance

        path_dir = os.path.dirname(os.path.realpath(__file__))
        path_index_file = os.path.join(path_dir, index_file)

        if not os.path.exists(path_index_file):
            raise ValueError("Index File '%s' not found!", path_index_file)
        index_file = open(path_index_file, 'r')

        ## loading image -----------------------------------
        if not os.path.isdir(images_root):
            raise ValueError("Image directory '%s' not found!")

        filename_list = [line.rstrip().split(' ') for line in index_file.readlines()]
        self._image_list = []
        view1 = 'image_02/data'
        view2 = 'image_03/data'
        ext = '.jpg'
        
        for item in filename_list:
            date = item[0][:10]
            scene = item[0]
            idx_src = item[1]
            idx_tgt = '%.10d' % (int(idx_src) + 1)
            name_l1 = os.path.join(images_root, date, scene, view1, idx_src) + ext
            name_l2 = os.path.join(images_root, date, scene, view1, idx_tgt) + ext
            name_r1 = os.path.join(images_root, date, scene, view2, idx_src) + ext
            name_r2 = os.path.join(images_root, date, scene, view2, idx_tgt) + ext

            if os.path.isfile(name_l1) and os.path.isfile(name_l2) and os.path.isfile(name_r1) and os.path.isfile(name_r2):
                self._image_list.append([name_l1, name_l2, name_r1, name_r2])

        if num_examples > 0:
            self._image_list = self._image_list[:num_examples]

        self._size = len(self._image_list)

        ## loading calibration matrix
        self.intrinsic_dict_l = {}
        self.intrinsic_dict_r = {}        
        self.intrinsic_dict_l, self.intrinsic_dict_r = read_calib_into_dict(path_dir)

        self._to_tensor = vision_transforms.Compose([
            vision_transforms.ToPILImage(),
            vision_transforms.transforms.ToTensor()
        ])

        # ######################################## c3d loader
        # ### c3d.datareader for cam_info and lidar depth processing
        # self.datareader = c3d.utils_general.DataReaderKITTI(data_root=images_root)
        # ###################################################

        ############ c3d v2
        ### in dataset, track the augmentation and record the file name

        # self.timer = c3d.utils_general.Timing()

    def __getitem__(self, index):
        # self.timer.log_temp("__getitem__ before aug")
        index = index % self._size

        # ######################################## c3d loader
        # lidar_in_cam_frame_list = []
        # K_list = []
        # for img_name in self._image_list[index]:
        #     data_dict = self.datareader.read_datadict_from_img_path(img_name, ftype_list=['lidar', 'calib'])
        #     extr_cam_li = data_dict['calib'].P_cam_li   # 4*4
        #     lidar_pts = data_dict['lidar']              # n*4
        #     lidar_in_cam_frame = np.dot(extr_cam_li, lidar_pts.T).T # N*4
        #     lidar_in_cam_frame = lidar_in_cam_frame[lidar_in_cam_frame[:,2] > 0, :]
        #     lidar_in_cam_frame = torch.from_numpy(lidar_in_cam_frame).float()
        #     lidar_in_cam_frame_list.append(lidar_in_cam_frame)

        #     K_unit = data_dict['calib'].K_unit
        #     K = bev.utils_general.scale_K(K_unit, new_width=data_dict['calib'].width, new_height=data_dict['calib'].height, torch_mode=False, align_corner=self.datareader.align_corner)
        #     K = torch.from_numpy(K).float()
        #     K_list.append(K)
        # k_l1 = K_list[0]
        # k_r1 = K_list[2]
        # ###################################################

        ############ c3d v2
        cam_ops = []
        
        # read images and flow
        # im_l1, im_l2, im_r1, im_r2
        img_list_np = [read_image_as_byte(img) for img in self._image_list[index]]

        # example filename
        im_l1_filename = self._image_list[index][0]
        basename = os.path.basename(im_l1_filename)[:6]
        dirname = os.path.dirname(im_l1_filename)[-51:]
        datename = dirname[:10]
        k_l1 = torch.from_numpy(self.intrinsic_dict_l[datename]).float()
        k_r1 = torch.from_numpy(self.intrinsic_dict_r[datename]).float()
        ### changed by Minghan: load from datareader
        
        # input size
        h_orig, w_orig, _ = img_list_np[0].shape
        input_im_size = torch.from_numpy(np.array([h_orig, w_orig])).float()

        # cropping 
        if self._preprocessing_crop:

            # get starting positions
            crop_height = self._crop_size[0]
            crop_width = self._crop_size[1]
            x = np.random.uniform(0, w_orig - crop_width + 1)
            y = np.random.uniform(0, h_orig - crop_height + 1)
            crop_info = [int(x), int(y), int(x + crop_width), int(y + crop_height)]

            # cropping images and adjust intrinsic accordingly
            img_list_np = kitti_crop_image_list(img_list_np, crop_info)
            k_l1, k_r1 = kitti_adjust_intrinsic(k_l1, k_r1, crop_info)

            ######################## c3d v2
            cam_ops.append(c3d.utils.cam.CamCrop(x_start=crop_info[0], y_start=crop_info[1], x_size=crop_width, y_size=crop_height))
        
        # to tensors
        img_list_tensor = [self._to_tensor(img) for img in img_list_np]
        
        im_l1 = img_list_tensor[0]
        im_l2 = img_list_tensor[1]
        im_r1 = img_list_tensor[2]
        im_r2 = img_list_tensor[3]
       
        common_dict = {
            "index": index,
            "basename": basename,
            "datename": datename,
            "input_size": input_im_size
        }

        ######################## c3d v2
        fname_dict = {
            "im_l1": self._image_list[index][0], 
            "im_l2": self._image_list[index][1], 
            "im_r1": self._image_list[index][2], 
            "im_r2": self._image_list[index][3], 
        }

        # random flip
        if self._flip_augmentations is True and torch.rand(1) > 0.5:
            _, _, ww = im_l1.size()
            im_l1_flip = torch.flip(im_l1, dims=[2])
            im_l2_flip = torch.flip(im_l2, dims=[2])
            im_r1_flip = torch.flip(im_r1, dims=[2])
            im_r2_flip = torch.flip(im_r2, dims=[2])

            # k_l1[0, 2] = ww - k_l1[0, 2]
            # k_r1[0, 2] = ww - k_r1[0, 2]
            ### changed by Minghan: 
            ### In python, left-most pixel index is 0, right-most pixel index is ww-1. They exchange. 
            k_l1[0, 2] = ww - k_l1[0, 2] - 1
            k_r1[0, 2] = ww - k_r1[0, 2] - 1

            ############### c3d v2
            cam_ops.append(c3d.utils.cam.CamFlip(hori=True, vert=False))

            # ######################################## c3d loader
            # ### cropping does not alter the lidar points (it changed the intrinsic matrix)
            # for lidar_in_cam_frame in lidar_in_cam_frame_list:
            #     lidar_in_cam_frame[:,0] = -lidar_in_cam_frame[:,0]

            # lidar_l1 = lidar_in_cam_frame_list[0]
            # lidar_l2 = lidar_in_cam_frame_list[1]
            # lidar_r1 = lidar_in_cam_frame_list[2]
            # lidar_r2 = lidar_in_cam_frame_list[3]

            # cam_info_l = c3d.utils.CamInfo_from_K_batched(im_l1.shape[2], im_l1.shape[1], k_l1.unsqueeze(0))
            # cam_info_r = c3d.utils.CamInfo_from_K_batched(im_r1.shape[2], im_r1.shape[1], k_r1.unsqueeze(0))

            # dep_l1 = cam_info_l.lidar_to_depth(lidar_l1)
            # dep_l2 = cam_info_l.lidar_to_depth(lidar_l2)
            # dep_r1 = cam_info_r.lidar_to_depth(lidar_r1)
            # dep_r2 = cam_info_r.lidar_to_depth(lidar_r2)

            # ###################################################
            example_dict = {
                "input_l1": im_r1_flip,
                "input_r1": im_l1_flip,
                "input_l2": im_r2_flip,
                "input_r2": im_l2_flip,                
                "input_k_l1": k_r1,
                "input_k_r1": k_l1,
                "input_k_l2": k_r1,
                "input_k_r2": k_l1,
                # "lidar_l1": lidar_r1, 
                # "lidar_l2": lidar_r2, 
                # "lidar_r1": lidar_l1, 
                # "lidar_r2": lidar_l2, 
                # "dep_l1": dep_r1, 
                # "dep_l2": dep_r2, 
                # "dep_r1": dep_l1, 
                # "dep_r2": dep_l2, 
                # "cam_info_l": cam_info_r, 
                # "cam_info_r": cam_info_l, 
            }
            example_dict.update(common_dict)

            ######################## c3d v2
            example_dict.update({
                "c3d_l1_fname": fname_dict["im_r1"],
                "c3d_l2_fname": fname_dict["im_r2"],
                "c3d_r1_fname": fname_dict["im_l1"],
                "c3d_r2_fname": fname_dict["im_l2"],
                "c3d_cam_ops": cam_ops,
            })
        else:


            # ######################################## c3d loader
            # lidar_l1 = lidar_in_cam_frame_list[0]
            # lidar_l2 = lidar_in_cam_frame_list[1]
            # lidar_r1 = lidar_in_cam_frame_list[2]
            # lidar_r2 = lidar_in_cam_frame_list[3]

            # cam_info_l = c3d.utils.CamInfo_from_K_batched(im_l1.shape[2], im_l1.shape[1], k_l1.unsqueeze(0))
            # cam_info_r = c3d.utils.CamInfo_from_K_batched(im_r1.shape[2], im_r1.shape[1], k_r1.unsqueeze(0))

            # dep_l1 = cam_info_l.lidar_to_depth(lidar_l1)    # B=1*H*W
            # dep_l2 = cam_info_l.lidar_to_depth(lidar_l2)
            # dep_r1 = cam_info_r.lidar_to_depth(lidar_r1)
            # dep_r2 = cam_info_r.lidar_to_depth(lidar_r2)

            # ###################################################
            example_dict = {
                "input_l1": im_l1,
                "input_r1": im_r1,
                "input_l2": im_l2,
                "input_r2": im_r2,
                "input_k_l1": k_l1,
                "input_k_r1": k_r1,
                "input_k_l2": k_l1,
                "input_k_r2": k_r1,
                # "lidar_l1": lidar_l1, 
                # "lidar_l2": lidar_l2, 
                # "lidar_r1": lidar_r1, 
                # "lidar_r2": lidar_r2, 
                # "dep_l1": dep_l1, 
                # "dep_l2": dep_l2, 
                # "dep_r1": dep_r1, 
                # "dep_r2": dep_r2, 
                # "cam_info_l": cam_info_l, 
                # "cam_info_r": cam_info_r, 
            }
            example_dict.update(common_dict)

            ######################## c3d v2
            example_dict.update({
                "c3d_l1_fname": fname_dict["im_l1"],
                "c3d_l2_fname": fname_dict["im_l2"],
                "c3d_r1_fname": fname_dict["im_r1"],
                "c3d_r2_fname": fname_dict["im_r2"],
                "c3d_cam_ops": cam_ops,
            })

        # self.timer.log_temp_end("__getitem__ before aug")

        # self.timer.log_temp("__getitem__ aug")

        ### do the augmentation here instead of after dataloader iteration, making use of the multiprocessing to improve efficiency
        ### but notice that we cannot process GPU tensor here because of the multiprocessing setup. 
        if self._augmentation is not None:
            example_dict = default_collate_with_camops_caminfo([example_dict])
            example_dict = self._augmentation.forward_c3d(example_dict)     ### if do all the augmentation here(c3d+original), use forward_all()
            example_dict = decollate_with_camops_caminfo(example_dict)
        # print("-------------example_dict[c3d_cam_ops]:", example_dict["c3d_cam_ops"])
        example_dict.pop("c3d_cam_ops")
        # self.timer.log_temp_end("__getitem__ aug")
        return example_dict

    def __len__(self):
        return self._size



class KITTI_Raw_KittiSplit_Train(KITTI_Raw):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[370, 1224],
                 num_examples=-1, 
                 augmentation_instance=None):
        super(KITTI_Raw_KittiSplit_Train, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/kitti_train.txt", 
            augmentation_instance=augmentation_instance)


class KITTI_Raw_KittiSplit_Valid(KITTI_Raw):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=False,
                 preprocessing_crop=False,
                 crop_size=[370, 1224],
                 num_examples=-1, 
                 augmentation_instance=None):
        super(KITTI_Raw_KittiSplit_Valid, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/kitti_valid.txt", 
            augmentation_instance=augmentation_instance)


class KITTI_Raw_KittiSplit_Full(KITTI_Raw):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[370, 1224],
                 num_examples=-1, 
                 augmentation_instance=None):
        super(KITTI_Raw_KittiSplit_Full, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/kitti_full.txt", 
            augmentation_instance=augmentation_instance)


class KITTI_Raw_EigenSplit_Train(KITTI_Raw):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[370, 1224],
                 num_examples=-1, 
                 augmentation_instance=None):
        super(KITTI_Raw_EigenSplit_Train, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/eigen_train.txt", 
            augmentation_instance=augmentation_instance)


class KITTI_Raw_EigenSplit_Valid(KITTI_Raw):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=False,
                 preprocessing_crop=False,
                 crop_size=[370, 1224],
                 num_examples=-1, 
                 augmentation_instance=None):
        super(KITTI_Raw_EigenSplit_Valid, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/eigen_valid.txt", 
            augmentation_instance=augmentation_instance)


class KITTI_Raw_EigenSplit_Full(KITTI_Raw):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[370, 1224],
                 num_examples=-1, 
                 augmentation_instance=None):
        super(KITTI_Raw_EigenSplit_Full, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/eigen_full.txt", 
            augmentation_instance=augmentation_instance)