import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numbers
from einops import rearrange
from basicsr.models.archs.hrseg_lib.models import seg_hrnet
from basicsr.models.archs.hrseg_lib.config import config
from basicsr.models.archs.hrseg_lib.config import update_config
from ipdb import set_trace as st

 

def create_hrnet():
    args = {}
    args['cfg'] = './basicsr/models/archs/hrseg_lib/pascal_ctx/seg_hrnet_w48_cls59_480x480_sgd_lr4e-3_wd1e-4_bs_16_epoch200.yaml'
    args['opt'] = []
    update_config(config, args)
    if torch.__version__.startswith('1'):
        module = eval('seg_hrnet')
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval(config.MODEL.NAME + '.get_seg_model')(config)
 

    pretrained_dict = torch.load('./experiments/pretrained_models/hrnet_w48_pascal_context_cls59_480x480.pth')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('HRNet load')
    return model

 
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))


def gelu(x):
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.power(x,3))))

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

 
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        'define my gelu'
        self.gelu = GELU()

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # x = F.gelu(x1) * x2
        x = self.gelu(x1) * x2
        x = self.project_out(x)
        return x
 
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
 
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', dim2=None):

        super(TransformerBlock, self).__init__()
        if dim2 is not None:
            self.dim2 = dim2
            self.conv_guidence = nn.Conv2d(dim2, dim, 1, 1)
        else:
            self.dim2 = None
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, refl, ill_feat):  
        if self.dim2 is not None:
            ill_feat = self.conv_guidence(ill_feat)
            ill_feat = F.interpolate(ill_feat, [refl.shape[2], refl.shape[3]])

        ill = ill_feat
        refl = self.norm1(refl)
        ill = self.norm1(ill)
        refl = refl + self.attn(refl, ill)
        refl = refl + self.ffn(self.norm2(refl))
        
        return refl
    
 
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3,reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        
        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V     

  
class ContextBlock(nn.Module):
    
    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, bias=bias, padding=1, groups=2),
             
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, bias=bias, padding=1, groups=2)
        )

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
             
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )
        
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, H, W]
        inp = x
        inp = self.head(inp)
        
        # [N, C, 1, 1]
        context = self.modeling(inp)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        inp = inp + channel_add_term
        x = x + self.act(inp)

        return x
 
class RCBdown(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCBdown, self).__init__()
        
        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential( 
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            # nn.BatchNorm2d(n_feat),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            # nn.BatchNorm2d(n_feat),
            act
        )

        self.act = act
        
        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        x = self.body(x)
        res = self.act(self.gcnet(x))
        res += x
        return res
    
    
class RCBup(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCBup, self).__init__()
        
        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(2*n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            # nn.BatchNorm2d(n_feat),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            # nn.BatchNorm2d(n_feat),
            act
        )

        self.act = act
        
        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        x = self.body(x)
        res = self.act(self.gcnet(x))
        res += x
        return res
     
class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels*chan_factor), 1, stride=1, padding=0, bias=bias)
            )

    def forward(self, x):
        return self.bot(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()
        # nn.Conv2d(in_channels, int(in_channels//chan_factor), 1, stride=1, padding=0, bias=bias),
        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            )

    def forward(self, x):
        return self.bot(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x
  
class UpSampleModule(nn.Module):
    def __init__(self, num_kernels):
        super(UpSampleModule, self).__init__()
 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1)
         
        self.conv3 = nn.ModuleList([nn.Conv2d(in_channels=256, out_channels=3*4, kernel_size=3, stride=1, padding=1)
                                    for _ in range(num_kernels)])
 
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=num_kernels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        kernel_weights = self.gate(out)         
        kernel_outputs = [conv(out) for conv in self.conv3]
        weighted_outputs = [kernel_weights[:, i:i+1, :, :] * kernel_output for i, kernel_output in enumerate(kernel_outputs)]
         
        out = torch.stack(weighted_outputs, dim=1).sum(dim=1)          
        out = self.pixel_shuffle(out)
        return out
 


class Illumination(nn.Module):
    def __init__(self, inp_channels=2, out_channels=1, n_feat=64, scale=1, bias=False):
        super(Illumination, self).__init__()
        
        self.scale = scale
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, stride=1, padding=1)
        
        self.conv1 = RCBdown(n_feat=n_feat)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = RCBdown(n_feat=n_feat)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = RCBdown(n_feat=n_feat)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = RCBdown(n_feat=n_feat)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5 = RCBdown(n_feat=n_feat) 
        self.upv6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        self.conv6 = RCBup(n_feat=n_feat) 
        self.upv7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        self.conv7 = RCBup(n_feat=n_feat) 
        self.upv8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        self.conv8 = RCBup(n_feat=n_feat) 
        self.upv9 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        self.conv9 = RCBup(n_feat=n_feat) 
        self.conv10_1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1)
 
        
        self.conv_out_r = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, refl): 
        
        conv1 = self.lrelu(self.conv_in(refl))        
        conv1 = self.conv1(conv1)
        pool1 = self.pool1(conv1)        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)        
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)        
        conv5 = self.conv5(pool4)         
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)        
        out = self.conv10_1(conv9) 
        out = torch.sigmoid(self.conv_out_r(out))
        
        return [conv5, conv6, conv7, conv8, conv9], out 
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
 
class RCBdown(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCBdown, self).__init__()
        
        act = nn.LeakyReLU(0.2)
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act, 
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act
        )

        self.act = act        
        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        inp = x
        x = self.body(x)
        res = self.act(self.gcnet(x))
        res += inp
        return res
    
    
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))
        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]        
        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)        
        return feats_V  


class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()
        
    def forward(self, inp):
        return inp


class bilinearup_block(nn.Module):
    def __init__(self, dim, out_dim, upscale, kernel_size=3, bias=False):
        super(bilinearup_block, self).__init__()
        self.block = nn.Sequential(nn.Upsample(scale_factor=upscale, mode='bilinear', align_corners=bias),
                                   nn.Conv2d(in_channels=dim, out_channels=out_dim, kernel_size=kernel_size, stride=1, padding=1, bias=bias), 
                                   nn.LeakyReLU(0.2))
 
        
    def forward(self, inp):
        output = self.block(inp)  
        return output
    
class bicubicup_block(nn.Module):
    def __init__(self, dim, upscale, kernel_size=3, bias=False) :
        super(bicubicup_block, self).__init__()
        self.block = nn.Sequential(nn.Upsample(scale_factor=upscale, mode='bicubic', align_corners=bias),
                                   nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, stride=1, padding=1, bias=bias), 
                                   nn.LeakyReLU(0.2))
 
        
    def forward(self, inp):
        output = self.block(inp)  
        return output
    

class ms_fea_fusion(nn.Module):
    def __init__(self, dim, scale, bias=False):
        super(ms_fea_fusion, self).__init__()
        
        self.scale = scale
        
        if scale == 2:
            self.dau_top = RCBdown(dim, bias=bias, groups=1)
            self.dau_bot = RCBdown(dim, bias=bias, groups=1) 
            
            self.up_x1_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            self.up_x1_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            self.down_x2_1 = nn.Conv2d(dim, dim, 1, 2, 0, bias=bias)
            self.down_x2_2 = nn.Conv2d(dim, dim, 1, 2, 0, bias=bias)
            
            self.skff_top = SKFF(dim, 2)
            self.skff_bot = SKFF(dim, 2)
            
            self.conv_out_top = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=bias)
            self.conv_out_bot = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=bias)
               
        elif scale == 4:
            self.dau_top = RCBdown(dim, bias=bias, groups=1)
            self.dau_mid = RCBdown(dim, bias=bias, groups=1)
            self.dau_bot = RCBdown(dim, bias=bias, groups=1)
            
            self.up_x1_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            self.up_x1_2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=bias)
            
            self.up_x2_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            self.down_x2_1 = nn.Conv2d(dim, dim, 1, 2, 0, bias=bias)
            
            self.down_x4_1 = nn.Conv2d(dim, dim, 1, 2, 0, bias=bias)
            self.down_x4_2 = nn.Conv2d(dim, dim, 1, 4, 0, bias=bias)
            
            self.skff_top = SKFF(dim, 3)
            self.skff_mid = SKFF(dim, 3)
            self.skff_bot = SKFF(dim, 3)
            
            self.conv_out_top = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=bias)
            self.conv_out_mid = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=bias)
            self.conv_out_bot = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=bias)
        
    def forward(self, inp):
        if self.scale == 2:
            x_top = inp[1].clone()
            x_bot = inp[0].clone()
            
            x_top = self.dau_top(x_top)
            x_bot = self.dau_bot(x_bot)
            
            x_top = self.skff_top([x_top, self.up_x1_1(x_bot)])
            x_bot = self.skff_bot([x_bot, self.down_x2_1(x_top)])
            
 
            out_top = self.conv_out_top(x_top)
            out_bot = self.conv_out_bot(x_bot)
            
            return [out_bot, out_top]
        
        elif self.scale == 4:
            x_top = inp[2].clone()
            x_mid = inp[1].clone()
            x_bot = inp[0].clone()
            
            x_top = self.dau_top(x_top)
            x_mid = self.dau_mid(x_mid)
            x_bot = self.dau_bot(x_bot)
            
            x_top = self.skff_top([x_top, self.up_x2_1(x_mid), self.up_x1_2(x_bot)])
            x_mid = self.skff_mid([self.down_x4_1(x_top), x_mid, self.up_x1_1(x_bot)])
            x_bot = self.skff_bot([self.down_x4_2(x_top), self.down_x2_1(x_mid), x_bot])
            
            out_top = self.conv_out_top(x_top)
            out_mid = self.conv_out_mid(x_mid)
            out_bot = self.conv_out_bot(x_bot)
            
            return [out_bot, out_mid, out_top]
        
        
class RTFU(nn.Module):
    def __init__(self, dim, scale, kernel_size=3, bias=False):
        super(RTFU, self).__init__()
        self.depth = int(np.log2(scale)) + 1
        
        self.up_ps_names = locals()
        for i in range(self.depth):
            self.up_ps_names['self.bilinear_x%s', 2**i] = nn.ModuleList([])
            if i == 0:
                self.up_ps_names['self.bilinear_x%s', 2**i].append(identity())
            else:
                for j in range(i):
                    self.up_ps_names['self.bilinear_x%s', 2**i].append(bilinearup_block(dim, dim, 2).cuda())
         
        self.RTM = ms_fea_fusion(dim=dim, scale=scale, bias=bias)        
        
        self.up_bicu_names = locals()
        for i in range(self.depth):
            self.up_bicu_names['self.bicubic_x%s', 2**i] = nn.ModuleList([])
            if i == (self.depth-1):
                self.up_bicu_names['self.bicubic_x%s', 2**i].append(identity())
            else:
                for j in range(self.depth-i-1):
                    self.up_bicu_names['self.bicubic_x%s', 2**i].append(bicubicup_block(dim=dim, upscale=2, kernel_size=3, bias=bias).cuda())

        self.final1_conv = nn.Conv2d(self.depth*dim, dim, 1, 1, 0, bias=bias)
        self.final_conv = nn.Conv2d(dim, 3, 1, 1, 0, bias=bias)
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, inp):
        # stage1
        out_stage = []
        for i in range(self.depth):
            temp = inp
            for op in self.up_ps_names['self.bilinear_x%s', 2**i]:
                temp = op(temp)
            out_stage.append(temp)
            
        #stage2
        out_stage2 = self.RTM(out_stage)
        
        #stage3
        out_stage3 = []
        for i in range(self.depth):
            temp = out_stage2[i]
            for op in self.up_bicu_names['self.bicubic_x%s', 2**i]:
                temp = op(temp)
            out_stage3.append(temp)
        
        out = torch.cat(out_stage3, dim=1)
        out = self.act(self.final_conv(self.act(self.final1_conv(out))))
                
        return out
 
class Enhancement(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, n_feat=64, scale=1, bias=False, seg_dims=None):
        super(Enhancement, self).__init__()
        
        self.scale = scale
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, stride=1, padding=1)
        
        self.conv1 = RCBdown(n_feat=n_feat)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = RCBdown(n_feat=n_feat)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = RCBdown(n_feat=n_feat)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = RCBdown(n_feat=n_feat)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5 = RCBdown(n_feat=n_feat)
        self.sm5 = TransformerBlock(dim=n_feat, dim2=seg_dims[4])
        self.sm55 = TransformerBlock(dim=n_feat)
        
         
        self.upv6 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv6 = RCBup(n_feat=n_feat)
        self.sm6 = TransformerBlock(dim=n_feat, dim2=seg_dims[3])
        self.sm66 = TransformerBlock(dim=n_feat)
        
 
        self.upv7 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv7 = RCBup(n_feat=n_feat)
        self.sm7 = TransformerBlock(dim=n_feat, dim2=seg_dims[2])
        self.sm77 = TransformerBlock(dim=n_feat)
        
         
        self.upv8 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv8 = RCBup(n_feat=n_feat)
        self.sm8 = TransformerBlock(dim=n_feat, dim2=seg_dims[1])
        self.sm88 = TransformerBlock(dim=n_feat)
        
         
        self.upv9 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv9 = RCBup(n_feat=n_feat)
        self.sm9 = TransformerBlock(dim=n_feat, dim2=seg_dims[0])
        self.sm99 = TransformerBlock(dim=n_feat)
        
        self.conv10_1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1)        
        self.tanh = nn.Tanh()        
        self.RTFU = RTFU(dim=n_feat, scale=scale)
        
 
    def forward(self, refl, seg_orin, seg_fea, ill_fea):
    
        conv1 = self.lrelu(self.conv_in(refl))
        
        conv1 = self.conv1(conv1)
        pool1 = self.pool1(conv1)        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)        
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)        
        conv5 = self.conv5(pool4)     
         
        conv5 = self.sm55(conv5, ill_fea[0]) 
        conv5 = self.sm5(conv5, seg_fea[3])         
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)        
        conv6 = self.sm66(conv6, ill_fea[1]) 
        conv6 = self.sm6(conv6, seg_fea[2])        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)        
        conv7 = self.sm77(conv7, ill_fea[2]) 
        conv7 = self.sm7(conv7, seg_fea[1])        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)        
        conv8 = self.sm88(conv8, ill_fea[3]) 
        conv8 = self.sm8(conv8, seg_fea[0])        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)
        conv9 = self.sm99(conv9, ill_fea[4]) 
        conv9 = self.sm9(conv9, seg_orin)
        out = self.conv10_1(conv9) 
        out = self.RTFU(out)
                            
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
 

 
class CSDLLSRNetv9_7_5(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        n_feat=64,
        scale=1,
        bias=False
    ):
        super(CSDLLSRNetv9_7_5, self).__init__()
        
        kernel_size=3
        self.n_feat = n_feat
        self.scale = scale
        self.seg_dims = [59, 48, 96, 192, 384]       
         
        self.illumination = Illumination(2, 1, n_feat, scale)        
        self.reflectance = Enhancement(inp_channels, n_feat, n_feat, scale, seg_dims=self.seg_dims)   
         
        self.seg = create_hrnet()
        for p in self.seg.parameters():
            p.requires_grad = False
            
 
        
        
    def forward(self, inp_img_lllr, inp_img_gray):

        _, seg_orin, seg_fea = self.seg(inp_img_lllr)
        ill_feas, nllr_ill = self.illumination(inp_img_gray)
        nllr_ill3 = torch.cat((nllr_ill, nllr_ill, nllr_ill), dim=1)
        nllr_ref = inp_img_lllr / nllr_ill3
        nllr_ref = torch.clamp(nllr_ref, 0, 1)
        nlsr_refl = self.reflectance(nllr_ref, seg_orin, seg_fea, ill_feas)
        img_nlsr = nlsr_refl          
        return nlsr_refl, nllr_ill, img_nlsr   
  

if __name__== '__main__':
    inp = torch.randn((1, 3, 128, 128))
    atten = torch.randn((1, 2, 128, 128))
    net = CSDLLSRNetv9_7_5(scale=2).eval()
    
    nlsr_refl, nlsr_ill, img_nlsr = net(inp, atten)
   

