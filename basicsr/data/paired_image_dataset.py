from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_folder3,
                                    paired_paths_from_folder6,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop3,paired_random_crop6, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding3, padding6, padding_DP, imfrombytesDP

import random
import numpy as np
import torch
import cv2
from ipdb import set_trace as st

class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']
            
    def  max_operation(self, img):
        img = img.float().numpy()
        x = np.maximum(img[:, :, :-1, :], img[:, :, 1:, :]) #相邻像素进行逐元素的最大值比较
        x = np.concatenate((x, np.expand_dims(img[:, :, -1, :], 2)), 2) #在指定维度height上进行连接，并将结果赋值给变量x。
        #(batch_size, channels, height)

        y = np.maximum(x[:, :, :, :-1], x[:, :, :, 1:])
        y = np.concatenate((y, np.expand_dims(x[:, :, :, -1], 3)), 3)

        y = torch.from_numpy(y)

        return y
    
    def edge_operation(self, img):
        img = img.float().numpy()

        x1 = img[:, :, :-1, :] - img[:, :, 1:, :]
        x1 = np.concatenate((x1, np.expand_dims(img[:, :, -1, :], 2)), 2)

        x2 = img[:, :, 1:, :] - img[:, :, :-1, :]
        x2 = np.concatenate((np.expand_dims(img[:, :, 0, :], 2), x2), 2)

        y1 = img[:, :, :, :-1] - img[:, :, :, 1:]
        y1 = np.concatenate((y1, np.expand_dims(img[:, :, :, -1], 3)), 3)

        y2 = img[:, :, :, 1:] - img[:, :, :, :-1]
        y2 = np.concatenate((np.expand_dims(img[:, :, :, 0], 3), y2), 3)

        img = (np.abs(x1) + np.abs(x2) + np.abs(y1) + np.abs(y2)) / 4.0

        y = torch.from_numpy(img)

        return y

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True) 
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            
        if self.opt['use_grayatten']:
            r,g,b = img_lq[0]+1, img_lq[1]+1, img_lq[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
    
            
            return {
                'lq': img_lq,
                'gt': img_gt,
                'gray': A_gray,
                'lq_path': lq_path,
                'gt_path': gt_path
            }
        elif self.opt['use_illguidance']:
            r,g,b = img_lq[0]+1, img_lq[1]+1, img_lq[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
            A_gray = torch.unsqueeze(A_gray, 0)
            # st()
            max_out = self.max_operation(A_gray)
            edge_out = self.edge_operation(A_gray)
            gadience = max_out + edge_out
            A_gray = torch.cat([gadience, A_gray], 1).squeeze(0)
            # st()
            return {
                'lq': img_lq,
                'gt': img_gt,
                'gray': A_gray,
                # 'gray': img_lq, #input_guidance 
                'lq_path': lq_path,
                'gt_path': gt_path
            }    
        else:    
            return {
                'lq': img_lq,
                'gt': img_gt,
                'lq_path': lq_path,
                'gt_path': gt_path
            }
            

    def __len__(self):
        return len(self.paths)
   
    
class Dataset_PairedImage3(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage3, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None 
        self.gt_folder, self.lq_folder, self.lh_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_lh']  
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder, self.lh_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt', 'lh']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder, self.lh_folder], ['lq', 'gt', 'lh'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder, self.lh_folder], ['lq', 'gt', 'lh'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder3(
                [self.lq_folder, self.gt_folder, self.lh_folder], ['lq', 'gt', 'lh'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        
        lh_path = self.paths[index]['lh_path']
        img_bytes = self.file_client.get(lh_path, 'lh')
        try:
            img_lh = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lh path {} not working".format(lh_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # st()
            # padding
            img_gt, img_lq, img_lh = padding3(img_gt, img_lq, img_lh, gt_size)

            # random crop
            img_gt, img_lq, img_lh = paired_random_crop3(img_gt, img_lq, img_lh, gt_size, scale, gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq, img_lh = random_augmentation(img_gt, img_lq, img_lh)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_lh = img2tensor([img_gt, img_lq, img_lh],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_lh, self.mean, self.std, inplace=True)
            
        if self.opt['use_grayatten']:
            r,g,b = img_lq[0]+1, img_lq[1]+1, img_lq[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
            
            return {
                'lq': img_lq,
                'gt': img_gt,
                'lh': img_lh,
                'gray': A_gray,
                'lq_path': lq_path,
                'gt_path': gt_path
            }
        else:    
            return {
                'lq': img_lq,
                'gt': img_gt,
                'lh': img_lh,
                'lq_path': lq_path,
                'gt_path': gt_path
            }

    def __len__(self):
        return len(self.paths)

class Dataset_PairedImage6(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage6, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder, self.lh_folder, self.gt_val_folder, self.lq_val_folder, self.lh_val_folder= opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_lh'],opt['dataroot_gt_val'], opt['dataroot_lq_val'], opt['dataroot_lh_val']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder, self.lh_folder,self.lq_val_folder,self.gt_val_folder, self.lh_val_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt', 'lh','lq_val','gt_val','lh_val']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder, self.lh_folder,self.lq_val_folder,self.gt_val_folder, self.lh_val_folder], ['lq', 'gt', 'lh','lq_val','gt_val','lh_val'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder, self.lh_folder,self.lq_val_folder,self.gt_val_folder, self.lh_val_folder], ['lq', 'gt', 'lh','lq_val','gt_val','lh_val'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder6(
                [self.lq_folder, self.gt_folder, self.lh_folder,self.lq_val_folder,self.gt_val_folder, self.lh_val_folder], ['lq', 'gt', 'lh','lq_val','gt_val','lh_val'],
                self.filename_tmpl)
      
        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        
        lh_path = self.paths[index]['lh_path']
        img_bytes = self.file_client.get(lh_path, 'lh')
        try:
            img_lh = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lh path {} not working".format(lh_path))
        
        gt_val_path = self.paths[index]['gt_val_path']
        img_bytes = self.file_client.get(gt_val_path, 'gt_val')
        try:
            img_gt_val = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt_val path {} not working".format(gt_val_path))
        
        lq_val_path = self.paths[index]['lq_val_path']
        img_bytes = self.file_client.get(lq_val_path, 'lq_val')
        try:
            img_lq_val = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq_val path {} not working".format(lq_val_path))
        
        lh_val_path = self.paths[index]['lh_val_path']
        img_bytes = self.file_client.get(lh_val_path, 'lh_val')
        try:
            img_lh_val = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lh_val path {} not working".format(lh_val_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            #st()
            # padding
            img_gt, img_lq, img_lh, img_gt_val, img_lq_val, img_lh_val = padding6(img_gt, img_lq, img_lh, img_gt_val, img_lq_val, img_lh_val, gt_size)

            # random crop
            img_gt, img_lq, img_lh, img_gt_val, img_lq_val, img_lh_val= paired_random_crop6(img_gt, img_lq, img_lh, img_gt_val, img_lq_val, img_lh_val, gt_size, scale, gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq, img_lh, img_gt_val, img_lq_val, img_lh_val = random_augmentation(img_gt, img_lq, img_lh, img_gt_val, img_lq_val, img_lh_val)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_lh, img_gt_val, img_lq_val, img_lh_val = img2tensor([img_gt, img_lq, img_lh, img_gt_val, img_lq_val, img_lh_val],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_lh, self.mean, self.std, inplace=True)
            normalize(img_lq_val, self.mean, self.std, inplace=True)
            normalize(img_gt_val, self.mean, self.std, inplace=True)
            normalize(img_lh_val, self.mean, self.std, inplace=True)
            
        if self.opt['use_grayatten']:
            r,g,b = img_lq[0]+1, img_lq[1]+1, img_lq[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
            
            return {
                'lq': img_lq,
                'gt': img_gt,
                'lh': img_lh,
                'lq_val': img_lq_val,
                'gt_val': img_gt_val,
                'lh_val': img_lh_val,
                'gray': A_gray,
                'lq_path': lq_path,
                'gt_path': gt_path,
                'lq_val_path': lq_val_path,
                'gt_val_path': gt_val_path
            }
        else:    
            return {
                'lq': img_lq,
                'gt': img_gt,
                'lh': img_lh,
                'lq_val': img_lq_val,
                'gt_val': img_gt_val,
                'lh_val': img_lh_val,
                'lq_path': lq_path,
                'gt_path': gt_path,
                'lq_val_path': lq_val_path,
                'gt_val_path': gt_val_path
            }

    def __len__(self):
        return len(self.paths)

class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type  = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None        

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.gt_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)


            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value])/255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:            
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test/255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lqL_folder, self.lqR_folder = opt['dataroot_gt'], opt['dataroot_lqL'], opt['dataroot_lqR']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_DP_paths_from_folder(
            [self.lqL_folder, self.lqR_folder, self.gt_folder], ['lqL', 'lqR', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lqL_path = self.paths[index]['lqL_path']
        img_bytes = self.file_client.get(lqL_path, 'lqL')
        try:
            img_lqL = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqL path {} not working".format(lqL_path))

        lqR_path = self.paths[index]['lqR_path']
        img_bytes = self.file_client.get(lqR_path, 'lqR')
        try:
            img_lqR = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqR path {} not working".format(lqR_path))


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)

            # random crop
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(img_lqL, img_lqR, img_gt, gt_size, scale, gt_path)
            
            # flip, rotation            
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt = img2tensor([img_lqL, img_lqR, img_gt],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], 0)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lqL_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
