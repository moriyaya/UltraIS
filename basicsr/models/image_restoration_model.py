import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from torch.autograd import Variable
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial
from ipdb import set_trace as st

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs]) 

class Mixing_Augment:  
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_
      
 
class CSDLLSRv4(BaseModel): 

    def __init__(self, opt):
        super(CSDLLSRv4, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))  
        self.net_g = self.model_to_device(self.net_g)  
        self.print_network(self.net_g)  

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)  
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))
     
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}') 
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)   
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)  
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            raise ValueError('pixel loss are None.')
            
        if train_opt.get('SATV_opt'):
            satv_type = train_opt['SATV_opt'].pop('type')
            cri_SATV_cls = getattr(loss_module, satv_type)
            self.cri_SATV = cri_SATV_cls(**train_opt['SATV_opt']).to(self.device)
            
        if train_opt.get('perceptual_opt'):
            perceptual_type = train_opt['perceptual_opt'
                                        ].pop('type')
            cri_perceptual_cls = getattr(loss_module, perceptual_type)
            self.cri_perceptual = cri_perceptual_cls(**train_opt['perceptual_opt']).to(self.device)
            
        if train_opt.get('IllSmoothL1_opt'):
            SmoothL1_type = train_opt['IllSmoothL1_opt'].pop('type')
            cri_SmoothL1_cls = getattr(loss_module, SmoothL1_type)
            self.cri_SmoothL1 = cri_SmoothL1_cls(**train_opt['IllSmoothL1_opt']).to(self.device)

        if train_opt.get('IllColor_opt'):
            IllColor_type = train_opt['IllColor_opt'].pop('type')
 
            cri_IllColor_cls = getattr(loss_module, IllColor_type)
            self.cri_IllColor = cri_IllColor_cls(**train_opt['IllColor_opt']).to(self.device)          
    
            
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
      
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):  
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            
        if 'gray' in data:
            self.atten = data['gray'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)
        

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            
        if 'gray' in data:
            self.atten = data['gray'].to(self.device)


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        _, nllr_ill, img_nlsr = self.net_g(self.lq, self.atten)
        nllr_gray = self.atten[:,1,:,:].unsqueeze(1)
        if not isinstance(img_nlsr, list):
            img_nlsr = [img_nlsr]
        if not isinstance(nllr_ill, list):
            nllr_ill = [nllr_ill]
        if not isinstance(nllr_gray, list):
            nllr_gray = [nllr_gray]

        self.output = img_nlsr[-1] 

        loss_dict = OrderedDict()
        # pixel loss
        l_total = 0.  
        l_pix_nlsr = 0.
        l_g_percep_nlsr = 0. 
        l_g_style_nlsr = 0.
        l_smooth_ill_nllr = 0.
        l_color_ill_nllr =0.
        
        for pred in img_nlsr:
            l_pix_nlsr += self.cri_pix(pred, self.gt)
            l_g_percep, l_g_style = self.cri_perceptual(pred, self.gt)
            if l_g_percep is not None:
                l_g_percep_nlsr += l_g_percep
            if l_g_style is not None:
                l_g_style_nlsr += l_g_style
        
        for img_atten, img_attengt in zip(nllr_ill, nllr_gray):
            l_smooth_ill_nllr += self.cri_SmoothL1(img_atten, img_attengt)               
        

        for img_atten in nllr_ill:
            l_color_ill_nllr += self.cri_IllColor(img_atten)
 
        l_total = l_pix_nlsr + l_g_percep_nlsr+l_smooth_ill_nllr+l_color_ill_nllr
 
        loss_dict['l_pix_nlsr'] = l_pix_nlsr
        loss_dict['l_g_percep_nlsr'] = l_g_percep_nlsr
        loss_dict['l_smooth_ill_nllr'] = l_smooth_ill_nllr
        loss_dict['l_color_ill_nllr'] = l_color_ill_nllr
        loss_dict['l_total'] = l_total
 
        l_total.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        if self.opt['datasets']['train']['use_grayatten']:
            gray_atten = F.pad(self.atten, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        else:
            gray_atten = None
        if self.opt['datasets']['train']['use_illguidance']:
            atten = F.pad(self.atten, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        else:
            atten = None
        self.nonpad_test(img, gray_atten, atten)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
 
    
    def nonpad_test(self, img=None, gray_atten=None, atten=None):
        if img is None:
            img = self.lq    
        if gray_atten is None and self.opt['datasets']['train']['use_grayatten']:
            gray_atten = self.atten  
        if atten is None and self.opt['datasets']['train']['use_illguidance']:
            atten = self.atten
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                if self.opt['datasets']['train']['use_grayatten']:
                    _, _, pred = self.net_g_ema(img, gray_atten)
                else:
                    _, _, pred = self.net_g_ema(img, atten)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                if self.opt['datasets']['train']['use_grayatten']:
                    _, _, pred = self.net_g(img, gray_atten)
                else:
                    _, _, pred = self.net_g(img, atten)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            } 

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            if self.opt['datasets']['train']['use_grayatten']:
                del self.atten
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                
                if self.opt['is_train']:
                    
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                    
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                else:
                    
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png') 
                    
                imwrite(sr_img, save_img_path) 

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

  
 