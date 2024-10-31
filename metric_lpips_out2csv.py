import csv
import os
import cv2
import argparse
from tqdm import tqdm
import lpips

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores

import numpy as np
import torch
from torch.autograd import Variable
from ipdb import set_trace as st

parser = argparse.ArgumentParser(description='test path')
parser.add_argument('--path_test', type=str, default='/data2/zyyue/lolisr_results/metric/codetest_dataset/PRE')
parser.add_argument('--path_gt', type=str, default='/data2/zyyue/lolisr_results/metric/codetest_dataset/GT')
parser.add_argument('--csv_path', type=str, default='/data2/zyyue/lolisr_results/metric/codetest_dataset/testcode1.csv')
parser.add_argument('--csv_list', type=list, default=['imgname', 'lpips'])

args = parser.parse_args()

path_test = args.path_test
path_gt = args.path_gt
csv_path = args.csv_path

csv_list = args.csv_list

# count = 0
outputs = []
gts = []
results_list = []

for files in os.listdir(path_gt): 
    gts.append(os.path.join(path_gt, files))
    
for files in os.listdir(path_test): 
    outputs.append(os.path.join(path_test, files))
    
gts.sort()
outputs.sort()

for i in tqdm(range(len(gts))):
    image_o_y = cv2.imread(outputs[i],1) / 127.5 - 1
    image_gt_y = cv2.imread(gts[i],1)/ 127.5 - 1
    # tensor
    image_o_y = np.array(image_o_y)[np.newaxis, :]
    image_o_y = np.transpose(image_o_y, (0, 3, 1, 2)).astype(np.float32)
    image_o_y = image_o_y.astype(np.float32)

    # print(np.shape(im_input), im_input)
    image_o_y = torch.tensor(image_o_y).type(torch.FloatTensor)
    image_o_y = Variable(image_o_y, requires_grad=False).cuda()

    image_gt_y = np.array(image_gt_y)[np.newaxis, :]
    image_gt_y = np.transpose(image_gt_y, (0, 3, 1, 2)).astype(np.float32)
    # print(np.shape(im_input), im_input)
    image_gt_y = torch.tensor(image_gt_y).type(torch.FloatTensor)
    image_gt_y = Variable(image_gt_y, requires_grad=False).cuda()
    
    lpips_temp = loss_fn_alex(image_o_y, image_gt_y).detach().cpu().numpy()
    img_name = outputs[i].split('/')[-1][:-4]
    
    result = {}
    result['imgname'] = img_name
    result['lpips'] = lpips_temp[0,0,0,0]
    results_list.append(result)
    # count += 1
    # print(count)
    
with open(csv_path,'w') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerow(csv_list)
    for i in tqdm(range(len(results_list))):
        temp = []
        for j in range(len(csv_list)):
            temp.append(results_list[i][csv_list[j]])
        writer.writerow(temp)