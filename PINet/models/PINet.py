#-*- coding:utf-8 -*-
import time
import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from thop import profile
import torch.nn.functional as F
import numbers
from einops import rearrange

class LocalInformationExtractor(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, padding=0, bias=True),
            nn.Conv2d(dim // reduction, dim // reduction, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        y1 = torch.mean(x, dim=1, keepdim=True)
        y2, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([y1, y2], dim=1)
        y = self.conv1(y)
        y = F.gelu(y)
        return y * x

class FeedForward(nn.Module):
    def __init__(self, dim, bias):
        super().__init__()
        self.project_in = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim, bias=bias)
        self.project_in2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dwconv3 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)
        self.dwconv4 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x1 = self.project_in(x)
        x2 = self.dwconv1(x1)
        x3 = self.project_in2(x)
        x4 = self.dwconv2(x3)
        x5 = self.dwconv3(x4)
        x6 = self.dwconv4(x5)
        x6 = self.project_out(x6)

        x = F.gelu(x2) * x6
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        return out

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

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
        return x / torch.sqrt(sigma + 1e-5) * self.weight

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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class ContextAwareTransformer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm1 = LayerNorm(dim, LayerNorm_type='0')
        self.atten = Attention(dim=dim, num_heads=num_heads, bias=True)
        self.norm2 = LayerNorm(dim, LayerNorm_type='0')
        self.mlp = FeedForward(dim=dim, bias=True)
        self.lce = LocalInformationExtractor(self.dim)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        shortcut = x
        lif = x = self.norm1(x)
        lc = self.lce(lif)

        x = shortcut + self.atten(x+lc)
        x = x + self.mlp(self.norm2(x)+lc)

        x = lc + x
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()

        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            ContextAwareTransformer(dim=dim, num_heads=num_heads)
            for i in range(depth)])

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size)
        return x

class ContextAwareTransformerBlock(nn.Module):
    def __init__(self, dim, depth, num_heads):
        super().__init__()
        self.dim = dim
        self.residual_group = BasicLayer(dim=dim,depth=depth,num_heads=num_heads)

    def forward(self, x, x_size):
        res = self.residual_group(x, x_size)
        res = res + x
        return res

class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self,embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

class SpatialAttentionModule(nn.Module):
    def __init__(self, dim):
        super(SpatialAttentionModule, self).__init__()
        self.att1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        f_cat = torch.cat((x1, x2), 1)
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        return att_map

class make_dilation_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3, dilation=2):
        super(make_dilation_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2+dilation-1,
                              bias=True, dilation=dilation)

    def forward(self, x):
        out = F.gelu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class DRDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, dilation):
        super(DRDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dilation_dense(nChannels_, growthRate, dilation=dilation))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = F.gelu(out)
        out = out + x
        return out

class HDRTransformer(nn.Module):
    def __init__(self, in_chans=9, embed_dim=64, depths=[6, 6, 6], num_heads=[8, 8, 8]):
        super(HDRTransformer, self).__init__()
        num_in_ch = in_chans
        num_out_ch = 3
        ################################### 1. Feature Extraction Network ###################################
        self.conv_f1 = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.conv_f2 = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.conv_f3 = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.att_module_l = SpatialAttentionModule(embed_dim)
        self.att_module_h = SpatialAttentionModule(embed_dim)
        self.conv_first = nn.Conv2d(embed_dim * 3, embed_dim, 3, 1, 1)
        ################################### 2. HDR Reconstruction Network ###################################
        self.num_layers = len(depths)

        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim=embed_dim)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ContextAwareTransformerBlock(dim=embed_dim,
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         )
            self.layers.append(layer)

        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, padding=1)
        self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, padding=1)

        self.apply(self._init_weights)
        self.drdb = DRDB(nChannels=embed_dim, nDenselayer=8, growthRate=16, dilation=2)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, F_0, branch):
        F_0 = F_0 + branch
        x_size = (F_0.shape[2], F_0.shape[3])
        F_0 = self.patch_embed(F_0)

        F_1 = self.layers[0](F_0, x_size)
        F_1 = self.patch_unembed(F_1, x_size)
        F_1 = F_1 + branch
        F_1 = self.patch_embed(F_1)
        F_2 = self.layers[1](F_1, x_size)
        F_2 = self.patch_unembed(F_2, x_size)
        F_2 = F_2 + branch
        F_2 = self.patch_embed(F_2)
        F_3 = self.layers[2](F_2, x_size)
        F_3 = self.patch_unembed(F_3, x_size)

        return F_3

    def forward(self, x1, x2, x3):
        z1 = self.conv_f1(x1)
        z2 = self.conv_f2(x2)
        z3 = self.conv_f3(x3)

        z1_att_m = self.att_module_h(z1, z2)
        z1_att = z1 * z1_att_m
        z3_att_m = self.att_module_l(z3, z2)
        z3_att = z3 * z3_att_m
        F_0 = self.conv_first(torch.cat((z1_att, z2, z3_att), dim=1))

        F_branch = self.drdb(F_0)
        F_3 = self.forward_features(F_0 ,F_branch) + F_0

        F_4 = self.conv_after_body(F_3+F_branch)
        F_out = self.conv_last(F_4+z2)
        F_out = torch.sigmoid(F_out)
        return F_out

def flops():
    height, width = 1000, 1500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x1 = torch.randn((1, 9, height, width)).to(device)
    x2 = torch.randn((1, 9, height, width)).to(device)
    x3 = torch.randn((1, 9, height, width)).to(device)
    model = HDRTransformer().to(device)
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for i in range(100):
            x = model(x1, x2, x3)
    end_time = time.time()
    print("Infer time: {}".format(end_time-start_time))

    flops, params = profile(model, inputs=(x1, x2, x3), verbose=False)
    print('model(%s): flops: %.3f G, params: %.3f M' % (
        'HDR-Transformer', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

if __name__ == '__main__':
    flops()
