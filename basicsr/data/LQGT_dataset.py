import os
import random
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data
import rawpy
import glob
from pdb import set_trace as st
import pdb
try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass


class LQGTDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        if self.opt["phase"] == "train":
            self.train_fns = glob.glob(self.opt["dataroot_lq"] + '0*.ARW')
            self.train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in self.train_fns]
        elif self.opt["phase"] == "val":
            self.train_fns = glob.glob(self.opt["dataroot_lq"] + '2*_00*.ARW')
            self.train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in self.train_fns]
        elif self.opt["phase"] == "test1":
            self.train_fns = glob.glob(self.opt["dataroot_lq"] + '1*_00*.ARW')
            self.train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in self.train_fns]
        
    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(
            self.opt["dataroot_gt"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.LR_env = lmdb.open(
            self.opt["dataroot_lq"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def pack_raw(self,raw):
    # pack Bayer image to 4 channels
        im = raw.raw_image_visible.astype(np.float32)
        im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

        im = np.expand_dims(im, axis=2)
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]
      
        out = np.concatenate((im[0:H:2, 0:W:2, :],
                            im[0:H:2, 1:W:2, :],
                            im[1:H:2, 1:W:2, :],
                            im[1:H:2, 0:W:2, :]), axis=2) 
        # out0 = out[:, :, 0:1]
        # out1 = out[:, :, 1:2]
        # out2 = out[:, :, 2:3]
        # out3 = out[:, :, 3:4]
        # #out3为三通道，out为四通道
        # out3 = np.concatenate((out0, (out1 + out3) / 2., out2), axis=2)
        return out
    def pack_raw_test(self,raw):
    # pack Bayer image to 4 channels
        im = raw.raw_image_visible.astype(np.float32)
        im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

        im = np.expand_dims(im, axis=2)
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]
      
        out = np.concatenate((im[0:H:2, 0:W:2, :],
                            im[0:H:2, 1:W:2, :],
                            im[1:H:2, 1:W:2, :],
                            im[1:H:2, 0:W:2, :]), axis=2) 

        return out
    
    def __getitem__(self, index):
        
        scale = 2 #self.opt["scale"]
        GT_size = self.opt["GT_size"]
        LR_size = self.opt["LR_size"]

        # for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        #train_id = self.train_ids[index]
        in_files = self.train_fns[index]
        # in_files = glob.glob(self.opt["dataroot_LQ"] + '%05d_00*.ARW' % train_id)
        # in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        in_path = in_files
        in_fn = os.path.basename(in_path)
        
        train_id = self.train_ids[index] 
        gt_files = glob.glob(self.opt["dataroot_gt"] + '%05d_00*.ARW' % train_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        # st = time.time()
        # cnt += 1
        in_raw = rawpy.imread(in_path)
        gt_raw = rawpy.imread(gt_path)
        gt_copy = gt_raw
        #pdb.set_trace()
        if self.opt["phase"] == "train" or self.opt["phase"] == "val":
            in_raw = self.pack_raw(in_raw)* ratio
            gt_raw = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_raw = np.float32(gt_raw / 65535.0)
        elif self.opt["phase"] == "test1":
            in_raw = self.pack_raw_test(in_raw)* ratio
            gt_raw = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_raw = np.float32(gt_raw / 65535.0)
        # gt_raw = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
        # gt_raw = np.float32(gt_raw / 65535.0)
        img_GT = gt_raw
        img_LR = in_raw
        GT_path, LR_path = gt_path, in_path
        
        if self.opt["phase"] == "train" :
            H, W, C = img_LR.shape
            #assert LR_size == GT_size // scale, "GT size does not match LR size"

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[
                rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :
            ]

            # augmentation - flip, rotate
            img_LR, img_GT = util.augment(
                [img_LR, img_GT],
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
            )
        elif self.opt["phase"] == "val":
            img_GT = img_GT[0 : 256, 0 : 256, :]
            img_LR = img_LR[0 : 128, 0 : 128, :]
        elif LR_size is not None:
            H, W, C = img_LR.shape
            assert LR_size == GT_size // scale, "GT size does not match LR size"

            if LR_size < H and LR_size < W:
                # center crop
                rnd_h = H // 2 - LR_size//2
                rnd_w = W // 2 - LR_size//2
                img_LR = img_LR[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size, :]
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[
                    rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :
                ]
           
        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()
        img_LR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))
        ).float()

        if LR_path is None:
            LR_path = GT_path
   
        return {"lq": img_LR, "gt": img_GT, "lq_path": LR_path, "gt_path": GT_path}
        # return {"LQ": img_LR, "GT": img_GT, "LQ_path": LR_path, "GT_path": GT_path,"GT_copy":gt_copy}

    def __len__(self):
        return len(self.train_ids)
