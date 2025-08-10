import torch
import math
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import block as B

class SRCNN(nn.Module):
    """A light-weighted SRCNN.

    The architecture is much like SRResNet, except that we downscale the
    feature by `downscale` times before residual blocks. In upscaling phase,
    we need to upscale the feature `downscale * upscale` times instead.
    """
    def __init__(self, num_in_ch, num_out_ch, nf, nb, downscale=2, scale=2,
                 norm_type=None, act_type='relu', mode='CNA', res_scale=1,
                 upsample_mode='upconv'):
        super(SRCNN, self).__init__()
        in_nc = num_in_ch
        out_nc = num_out_ch
        n_downscale = int(math.log(downscale, 2))
        self.scale = scale
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        upscale=scale
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_block
            # upsample_block = partial(B.upconv_block, mode='bilinear')
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError(
                'upsample mode [{:s}] is not found'.format(upsample_mode))

        def _unet(d):
            resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type,
                                           act_type=act_type, mode=mode,
                                           res_scale=res_scale)
                             for _ in range(nb)]

            if d == 0:
                return B.sequential(*resnet_blocks)

            downsampler = B.conv_block(nf, nf, kernel_size=3,
                                       stride=2, norm_type=None,
                                       act_type=act_type)
            upsampler = upsample_block(nf, nf, act_type=act_type)
            return B.sequential(*resnet_blocks,
                                B.ShortcutBlock(
                                    B.sequential(downsampler,
                                                 _unet(d - 1),
                                                 upsampler)))

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3,
                                norm_type=None, act_type=act_type)
        ########
        fea_conv0 = B.conv_block(nf, nf, kernel_size=3, stride=upscale, norm_type=None, act_type=act_type)
        LR_conv = B.conv_block(nf, nf, kernel_size=3,
                               norm_type=None, act_type=None, mode=mode)

        if upscale == 1:
            upsampler = [B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)]
        elif upscale == 3:
            upsampler = [upsample_block(nf, nf, 3, act_type=act_type)]
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type)
                         for _ in range(n_upscale)]

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None,
                                act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None,
                                act_type=None)

        self.model = B.sequential(fea_conv,fea_conv0,
                                  B.ShortcutBlock(
                                      B.sequential(_unet(n_downscale),
                                                   LR_conv)),
                                  *upsampler,
                                  HR_conv0,
                                  HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x