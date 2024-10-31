import csv
import os
import argparse
from tqdm import tqdm
from ipdb import set_trace as st
from image_similarity_measures.evaluate import evaluation

parser = argparse.ArgumentParser(description='test path')
parser.add_argument('--path_test', type=str, default='/data1/zyyue/LLSR_compare_and_ablation/HAT-main/results/HAT-S_SRx2_ZeroDCE_ReLLUSUR_test/visualization/ZeroDCE-HAT/')
parser.add_argument('--path_gt', type=str, default='/data2/zyyue/dataset/RELLISUR-Dataset/Test_crop/NLHR-Duplicates/X2')
parser.add_argument('--csv_path', type=str, default='/data2/zyyue/xx/MIROurs0804/Metrics/test_HAT_ZeroDCEcode3.csv')
# parser.add_argument('--metrics', type=list, default=['rmse', 'fsim', 'sre'])
# parser.add_argument('--csv_list', type=list, default=['imgname', 'rmse', 'fsim', 'sre'])
parser.add_argument('--metrics', type=list, default=['rmse', 'fsim', 'sre'])
parser.add_argument('--csv_list', type=list, default=['imgname', 'rmse', 'fsim', 'sre'])

args = parser.parse_args()

path_test = args.path_test
path_gt = args.path_gt
csv_path = args.csv_path

metrics = args.metrics
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

# st()

for i in tqdm(range(len(gts))):
    result = evaluation(org_img_path=gts[i], 
           pred_img_path=outputs[i], 
           metrics=metrics)
    img_name = outputs[i].split('/')[-1][:-4]
    result['imgname'] = img_name
    results_list.append(result)
    # count += 1
    # print(count)
    
with open(csv_path,'w') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerow(csv_list)
    for i in range(len(results_list)):
        temp = []
        for j in range(len(csv_list)):
            temp.append(results_list[i][csv_list[j]])
        writer.writerow(temp)
    