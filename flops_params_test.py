import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import torch
import numpy as np
import cv2
import glob

import time
# from basicsr.archs.rrdbnet_arch import RRDBNet
# from basicsr.archs.mirnetv2_arch import MIRNet_v2_old
# from basicsr.models.archs.csd_gjx_v9_7_4_arch import CSDLLSRNetv9_7_4
from basicsr.models.archs.csd_gjx_v9_7_5_arch import CSDLLSRNetv9_7_5
# from basicsr.models.archs.mirnetv2retinexv8_arch import MIRNetv2Retinexv8
# from basicsr.archs.mirnetv2atten_arch import MIRNetv2atten
# from basicsr.archs.mirnetv2ueatten_arch import MIRNetv2ueatten
# from realesrgan import RealESRGANer
# from realesrgan.archs.srvgg_arch import SRVGGNetCompact
# from ptflops import get_model_complexity_info
# from thop import profile
# from torchstat import stat
from utils_modelsummary import get_model_activation, get_model_flops
from ipdb import set_trace

def main():
    """Inference demo for Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/data2/zyyue/dataset/RELLISUR-Dataset/Test_crop/LLLR', help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='CSDLLSRNetv9_7_5',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus'
              'RealESRGANv2-anime-xsx2 | RealESRGANv2-animevideo-xsx2-nousm | RealESRGANv2-animevideo-xsx2'
              'RealESRGANv2-anime-xsx4 | RealESRGANv2-animevideo-xsx4-nousm | RealESRGANv2-animevideo-xsx4'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--outscale', type=float, default=2, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--half', action='store_true', help='Use half precision during inference')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    args = parser.parse_args()

    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name in ['RealMIRSRGAN_x2plus']:  # x2 MIRNetv2 model
        model = MIRNetv2(inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, n_RRG=4, n_MRB=2, height=3, width=2, scale=2)
        netscale = 2
    elif args.model_name in ['RealMIRSRGAN_x4plus']:  # x4 MIRNetv2 model
        model = MIRNetv2(inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, n_RRG=4, n_MRB=2, height=3, width=2, scale=4)
        netscale = 4
    elif args.model_name in ['RealMIRATTENSRGAN_x2plus']:  # x2 MIRNetv2atten model
        model = MIRNetv2atten(inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, n_RRG=4, n_MRB=2, height=3, width=2, scale=2)
        netscale = 2
    elif args.model_name in ['RealMIRATTENSRGAN_x4plus']:  # x4 MIRNetv2atten model
        model = MIRNetv2atten(inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, n_RRG=4, n_MRB=2, height=3, width=2, scale=4)
        netscale = 4
    elif args.model_name in ['RealMIRUEATTENSRGAN_x2plus']:  # x2 MIRNetv2atten model
        model = MIRNetv2ueatten(inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, n_RRG=4, n_MRB=2, height=3,
                                width=2, scale=2)
        netscale = 2
    elif args.model_name in ['RealMIRUEATTENSRGAN_x4plus']:  # x2 MIRNetv2atten model
        model = MIRNetv2ueatten(inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, n_RRG=4, n_MRB=2, height=3,
                                width=2, scale=4)
        netscale = 4
    elif args.model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['RealESRGAN_x2plus']:  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif args.model_name in ['MIRNet_v2_old']:
        model = MIRNet_v2_old(inp_channels=4, out_channels=12, n_feat=80, chan_factor=1.5, n_RRG=4, n_MRB=2, height=3, width=2, scale=1)
    elif args.model_name in ['MIRRetinexv7']:
        model = MIRNetv2Retinexv7(scale=2)
    elif args.model_name in ['CSDLLSRNetv9_7_5']:
        model = CSDLLSRNetv9_7_5(scale=2)

    ##---------------------------params flops---------------------------##
    # flops, params = get_model_complexity_info(model, (3, 624, 624), as_strings=True, print_per_layer_stat=True)

    # stat(model, (3, 256, 256))

    # dummy_input = torch.randn(1, 3, 256, 256)
    # flops, params = profile(model, (dummy_input,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))


    ################

    # input_dim = (3, 128, 128)
    # flops = get_model_flops(model, input_dim, False)
    # print('FLOPs:', flops/10**9, 'G')
    # num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    # print('#Params', num_parameters/10**6, 'M')


    # ##---------------------------inference---------------------------##
    # device = torch.device("cuda")
    # model.to(device)
    # model.eval()
    # dummy_input = torch.randn(1, 3, 128, 128, dtype=torch.float).to(device)
    # #dummy_input2 = torch.randn(1, 1, 128, 128, dtype=torch.float).to(device)

    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # repetitions = 300
    # timings=np.zeros((repetitions,1))
    # #GPU-WARM-UP
    # for _ in range(10):
    #     _ = model(dummy_input)
    #     #_ = model(dummy_input,dummy_input2)
    # # MEASURE PERFORMANCE
    # with torch.no_grad():
    #     for rep in range(repetitions):
    #         starter.record()
    #         _ = model(dummy_input)
    #         #_ = model(dummy_input,dummy_input2)
    #         ender.record()
    #         # WAIT FOR GPU SYNC
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)
    #         timings[rep] = curr_time
    # mean_syn = np.sum(timings) / repetitions
    # std_syn = np.std(timings)
    # mean_fps = 1000. / mean_syn
    # print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
    # # print(mean_syn)

    # # torch.cuda.synchronize()
    # # time_start = time.time()
    # # predict = model(dummy_input)
    # # torch.cuda.synchronize()
    # # time_end = time.time()
    # # time_sum = time_end - time_start
    # # print(time_sum)

    input_dim = (3, 128, 128)
    flops = get_model_flops(model, input_dim, False)
    print('FLOPs:', flops/10**9, 'G')
    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    print('#Params', num_parameters/10**6, 'M')


    ##---------------------------inference---------------------------##
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, 128, 128, dtype=torch.float).to(device)
    dummy_input2 = torch.randn(1, 2, 128, 128, dtype=torch.float).to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        # _ = model(dummy_input)
        _ = model(dummy_input,dummy_input2)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            # _ = model(dummy_input)
            _ = model(dummy_input,dummy_input2)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
    # print(mean_syn)

    # torch.cuda.synchronize()
    # time_start = time.time()
    # predict = model(dummy_input)
    # torch.cuda.synchronize()
    # time_end = time.time()
    # time_sum = time_end - time_start
    # print(time_sum)

if __name__ == '__main__':
    main()