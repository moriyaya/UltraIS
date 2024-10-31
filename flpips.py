import os
import cv2
#from skimage.measure import compare_psnr as psnr
#from skimage.measure import compare_ssim as ski_ssim
import numpy as np
from PIL import Image
import lpips
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from ipdb import set_trace


# path_test = "/data2/zyyue/MIROurs/experiments/CSDNFLLSR_v2_scale4_illiter_then_ref_input_seg_illguideref_segguideref_doubleguide_noteInputRef_RELLISUR/results_144000/"
# path_gt = "/data2/zyyue/datasets/RELLISUR-Dataset/Test_crop/NLHR-Duplicates/X4"

# path_test = "/data2/zyyue/MIROurs/experiments/CSDNFLLSR_v2_scale2_illiter_then_ref_input_seg_illguideref_segguideref_doubleguide_noteInputRef_RELLISUR/results_144000/" #/data2/zyyue/MIROurs/experiments/CSDNFLLSR_v2_scale4_illiter_then_ref_input_seg_illguideref_segguideref_doubleguide_noteInputRef_RELLISUR/results_144000/"
# path_gt = "/data2/zyyue/datasets/RELLISUR-Dataset/Test_crop/NLHR-Duplicates/X2"

path_test = "/data2/zyyue/xx/MIROurs0804/experiments/v9_7_5_1_abl_onlyIS+R/results_135000/" #/data2/zyyue/MIROurs/experiments/CSDNFLLSR_v2_scale2_illiter_then_ref_input_seg_illguideref_segguideref_doubleguide_noteInputRef_RELLISUR/results_144000/" #/data2/zyyue/MIROurs/experiments/CSDNFLLSR_v2_scale4_illiter_then_ref_input_seg_illguideref_segguideref_doubleguide_noteInputRef_RELLISUR/results_144000/"
path_gt = "/data2/zyyue/dataset/RELLISUR-Dataset/Test_crop/NLHR-Duplicates/X2"


# path_test = r'F:\00000\data\LOL1\SSIENet'
# path_gt = r"F:\00000\data\LOL1\gt"


tLIP_mean = 0
LIP_mean = 0
count = 0
outputs_tmp = []
outputs = []
gts = []
for files in os.listdir(path_gt): 
    gts.append(os.path.join(path_gt, files))

#针对某次迭代的测试数据完整存在单独文件夹下
for files in os.listdir(path_test): 
    outputs.append(os.path.join(path_test, files))

# 针对各种迭代次数的图片存在了一个文件夹下
# for files in os.listdir(path_test):
#     outputs_tmp.append(os.path.join(path_test, files))
# outputs_tmp.sort()

# for files in outputs_tmp:
#     img_name = files.split('/')[-1]
#     outputs.append(files + '/' + img_name + '_110000.png')

gts.sort()
outputs.sort()
# set_trace()

for output in outputs:
    image_o_y = cv2.imread(output,1) / 127.5 - 1
    image_gt_y = cv2.imread(gts[count],1)/ 127.5 - 1
    # tensor
    image_o_y = np.array(image_o_y)[np.newaxis, :]
    image_o_y = np.transpose(image_o_y, (0, 3, 1, 2)).astype(np.float)
    image_o_y = image_o_y.astype(np.float)

    # print(np.shape(im_input), im_input)
    image_o_y = torch.tensor(image_o_y).type(torch.FloatTensor)
    image_o_y = Variable(image_o_y, requires_grad=False).cuda()

    image_gt_y = np.array(image_gt_y)[np.newaxis, :]
    image_gt_y = np.transpose(image_gt_y, (0, 3, 1, 2)).astype(np.float)
    # print(np.shape(im_input), im_input)
    image_gt_y = torch.tensor(image_gt_y).type(torch.FloatTensor)
    image_gt_y = Variable(image_gt_y, requires_grad=False).cuda()
    d3 = loss_fn_alex(image_o_y, image_gt_y).detach().cpu().numpy()
    LIP_mean += d3
    count += 1
    print(count)

print(LIP_mean/count)