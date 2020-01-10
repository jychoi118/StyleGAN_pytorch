import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

from math import sqrt

import random

def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)



def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super().__init__()

        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)

'''
class FusedUpsample(nn.Module):
    def __init__(self):

    def forward(self, input):


class FusedDownSample(nn.Module):
    def __init__(self):

    def forward(self, input):
'''


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

# keepdim: avoid squeeze
# **: element-wise square
'''
class BlurFunction(Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):

    @staticmethod
    def backward(ctx, *grad_outputs):
        class BlurFunctionBackward(Function):
            @staticmethod
            def forward(ctx, *args, **kwargs):

            @staticmethod
            def backward(ctx, *grad_outputs):


blur = BlurFunction.apply
'''
# The constant input in synthesis network is initialized to one.
class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.ones(1, channel, size, size)) # randn in github

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

# The biases and noise scaling factors are initialized to zero
# learned per-channel scaling factors to the noise input
class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        #print(image.size())
        #print(noise.size())
        return image + self.weight * noise  # broadcast automatically?


# The biases and noise scaling factors are initialized to zero,
# except for the biases associated with ys that we initialize to
# one
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1) # batchxin_channelx1x1

        out = self.norm(input)
        out = gamma * out + beta

        return out


# Bilinear upsampling
class StyleBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size=3,
            stride = 1,
            padding = 1,
            style_dim=512,
            resolution = 4):

        super().__init__()
        if resolution == 4:
            self.conv1 = ConstantInput(channel=in_channel, size=4)
        else:
            self.conv1 = nn.Sequential(
                F.interpolate(scale_factor=2, mode='bilinear'),
                EqualConv2d(in_channel, out_channel, kernel_size, stride, padding),
            )

        self.noiseInject1 = equal_lr(NoiseInjection(channel=out_channel))
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.AdaIN1 = AdaptiveInstanceNorm(out_channel, style_dim)

        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.noiseInject2 = equal_lr(NoiseInjection(channel=out_channel))
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.AdaIN2 = AdaptiveInstanceNorm(out_channel, style_dim)


    def forward(self, input, style, noise):  # same block, same noise?
        out = self.conv1(input)
        out = self.noiseInject1(out, noise)  # more easy to define noise outside, for device
        out = self.lrelu1(out)  # leaky relu every layer
        out = self.AdaIN1(out, style)

        out = self.conv2(out)
        out = self.noiseInject2(out, noise)
        out = self.lrelu2(out)
        out = self.AdaIN2(out, style)

        return out


class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, n_mlp = 8):
        super().__init__()
        layers = []
        layers.append(PixelNorm())
        for i in range(n_mlp):
            if i==0:
                layers.append(EqualLinear(in_dim=z_dim, out_dim=w_dim))
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(EqualLinear(in_dim=w_dim, out_dim=w_dim))
                layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(self, input): # input z
        return self.style(input)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.style = MappingNetwork(z_dim=512, w_dim=512, n_mlp=8)

        self.progressive = nn.ModuleList(
            [
                StyleBlock(in_channel=512, out_channel= 512, kernel_size=3,stride = 1,padding = 1,style_dim=512,resolution = 4),
                StyleBlock(512, 512, 3, 1, 1, 512, 8),
                StyleBlock(512, 512, 3, 1, 1, 512, 16),
                StyleBlock(512, 512, 3, 1, 1, 512, 32),
                StyleBlock(512, 256, 3, 1, 1, 512, 64),
                StyleBlock(256, 128, 3, 1, 1, 512, 128),
                StyleBlock(128, 64, 3, 1, 1, 512, 256),
                StyleBlock(64, 32, 3, 1, 1, 512, 512),
                StyleBlock(32, 16, 3, 1, 1, 512, 1024)
            ]

        )
        self.to_rgb = nn.ModuleList(   # no activation function
            [
                EqualConv2d(in_channels=512, out_channels=3, kernel_size=1),  #4
                EqualConv2d(512, 3, 1),  #8
                EqualConv2d(512, 3, 1),  #16
                EqualConv2d(512, 3, 1),  #32
                EqualConv2d(256, 3, 1),  #64
                EqualConv2d(128, 3, 1),  #128
                EqualConv2d(64, 3, 1),  #256
                EqualConv2d(32, 3, 1),  #512
                EqualConv2d(16, 3, 1),  #1024
            ]
        )

    def forward(self, z_codes, noises=None, step=0, mixing_range=(-1, -1), alpha = 1):  # step start with 0
        styles = []
        n_batch = z_codes[0].size(0)

        # in case z is not list or tuple
        if type(z_codes) not in (list, tuple):
            z_codes = [z_codes]

        # create noise for stochastic variation
        if noises is None:
            noises = []
            for i in range(step + 1):
                size = 4 * 2 ** i
                noises.append(torch.randn(n_batch, 1, size, size, device=z_codes[0].device))

        # initialize out
        out = noises[0]

        # inject index for second style
        if len(z_codes) == 1:
            inject_idx = len(self.progressive)  # no injection
        elif len(z_codes) == 2:
            inject_idx = random.sample(list(range(step)), 1)
        else:
            raise NotImplementedError('too many z_codes')

        # two styles
        # first style is main source
        for z in z_codes:
            styles.append(self.style(z))

        for i, (style_block, to_rgb) in enumerate(zip(self.progressive, self.to_rgb)):

            # when training
            if mixing_range == (-1, -1):
                if i < inject_idx:
                    style_step = styles[0]

                # second style from inject_idx
                else:
                    style_step = styles[1]

            # when generating samples
            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = styles[1]
                else:
                    style_step = styles[0]

            # for residual connection
            out_prev = out
            # class StyleBlock
            out = style_block(input= out, style = style_step, noise = noises[i])

            # last layer
            if i == step:
                out = to_rgb(out)
                # residual connection from PG-GAN
                if i > 0 and 0 <= alpha < 1:
                    skip = to_rgb(F.interpolate(out_prev, scale_factor = 2, mode='bilinear'))
                    out = alpha * out + (1-alpha) * skip

                break

        return out

class DownBlock(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 kernel_size2=None,
                 padding2=None):
        super().__init__()
        layers = []

        kernel2 = kernel_size2
        pad2 = padding2
        # for last conv layer of D
        if kernel_size2 is not None:
            kernel2 = kernel_size2
        if padding2 is not None:
            pad2 = padding2

        layers.append(
            EqualConv2d(in_channel, out_channel, kernel_size, stride, padding),
            nn.LeakyReLU(0.2),
            EqualConv2d(out_channel, out_channel, kernel2, stride, pad2),
            F.interpolate(scale_factor=0.5, mode='bilinear'),
            nn.LeakyReLU(0.2)
        )
        self.block = nn.Sequential(*layers)

    def forward(self, input):
        return self.block(input)


class Discriminator(nn.Module):
    def __init__(self,
                 ):
        super().__init__()
        self.progression = nn.ModuleList(
            [
                DownBlock(in_channel=16, out_channel=32, kernel_size=3, stride=1, padding=1),  # resolution = 512
                DownBlock(32, 64, 3, 1, 1),  # 256
                DownBlock(64, 128, 3, 1, 1),  # 128
                DownBlock(128, 256, 3, 1, 1),  # 64
                DownBlock(256, 512, 3, 1, 1),  # 32
                DownBlock(512, 512, 3, 1, 1),  # 16
                DownBlock(512, 512, 3, 1, 1),  # 8
                DownBlock(512, 512, 3, 1, 1),  # 4
                DownBlock(512, 512, 3, 1, 1, kernel_size2=4, padding2=0),  # 1
            ]

        )
        def make_from_rgb(out_channel):
            return nn.Sequential(EqualConv2d(3, out_channel, 1), nn.LeakyReLU(0.2))

        self.from_rgb = nn.ModuleList(
            [
                make_from_rgb(16), #1024
                make_from_rgb(32),  #512
                make_from_rgb(64),  #256
                make_from_rgb(128),  #128
                make_from_rgb(256),  #64
                make_from_rgb(512),  #32
                make_from_rgb(512),  #16
                make_from_rgb(512),  #8
                make_from_rgb(512),  #4
            ]
        )

        self.n_layer = len(self.progression)
        self.linear = EqualLinear(512, 1)

    def forward(self, input, step, alpha):

        for i in range(step + 1):
            # first layer
            if i == 0:
                out = self.from_rgb[self.n_layer - step - 1](input)
                out = self.progression[self.n_layer - step - 1](out)

                if 0 <= alpha < 1:
                    skip_input = F.interpolate(input, scale_factor=0.5, mode='bilinear')
                    skip_input = self.from_rgb[self.n_layer - step](skip_input)
                    out = alpha * out + (1 - alpha) * skip_input
                continue

            # last layer, minibatch standard deviation from PG-GAN
            elif i == step:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[self.n_layer - step - 1 + i](out) # batchx512x1x1

        out = out.squeeze(2).sequeeze(2) # batchx512

        return self.linear(out)