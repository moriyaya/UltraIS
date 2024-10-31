import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# path_test = "E:/UIEB-output/OCM"
# path_gt = "C:/Users/20686/Desktop/UIEB/visualization/GT"
path_test = "/data1/zyyue/LLSR_compare_and_ablation/Ons_stream/results_147000/" #/data2/zyyue/MIROurs/experiments/CSDNFLLSR_v2_scale2_illiter_then_ref_input_seg_illguideref_segguideref_doubleguide_noteInputRef_RELLISUR/results_144000/" #/data2/zyyue/MIROurs/experiments/CSDNFLLSR_v2_scale4_illiter_then_ref_input_seg_illguideref_segguideref_doubleguide_noteInputRef_RELLISUR/results_144000/"
path_gt = "/data2/zyyue/dataset/RELLISUR-Dataset/Test_crop/NLHR-Duplicates/X2"

ssim_mean = 0
count =0
outputs_tmp = []
outputs = []
gts = []
for files in os.listdir(path_gt): 
    gts.append(os.path.join(path_gt, files))

#针对某次迭代的测试数据完整存在单独文件夹下
for files in os.listdir(path_test): 
    outputs.append(os.path.join(path_test, files))

# # 针对各种迭代次数的图片存在了一个文件夹下
# for files in os.listdir(path_test):
#     outputs_tmp.append(os.path.join(path_test, files))
# outputs_tmp.sort()

# for files in outputs_tmp:
#     img_name = files.split('/')[-1]
#     outputs.append(files + '/' + img_name + '_144000.png')

gts.sort()
outputs.sort()

for output in outputs:
    img1 = cv2.imread(output)
    img2 = cv2.imread(gts[count])
    
    ssim = compare_psnr(img1, img2)
    
    ssim_mean += ssim 
    count += 1
    print(count)
    
print(ssim_mean / count)