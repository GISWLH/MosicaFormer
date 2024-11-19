import torch
from torch import nn

import torch
import torch.nn as nn
from torch.nn.modules.utils import _quadruple
import math
import torch.nn.functional as F

class Conv4d(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:[int, tuple],
                 stride:[int, tuple] = (1, 1, 1, 1),
                 padding:[int, tuple] = (0, 0, 0, 0),
                 dilation:[int, tuple] = (1, 1, 1, 1),
                 groups:int = 1,
                 bias=False,
                 padding_mode:str ='zeros'):
        super(Conv4d, self).__init__()
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))

        # Assertions for constructor arguments
        assert len(kernel_size) == 4, '4D kernel size expected!'
        assert len(stride) == 4, '4D Stride size expected!!'
        assert len(padding) == 4, '4D Padding size expected!!'
        assert len(dilation) == 4, '4D dilation size expected!'
        assert groups == 1, 'Groups other than 1 not yet implemented!'

        # Store constructor arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.groups = groups
        self.padding_mode = padding_mode

        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        # # # # # self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 3)

        # Construct weight and bias of 4D convolution
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(self.kernel_size[0]):
            # Initialize a Conv3D layer
            conv3d_layer = nn.Conv3d(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=self.kernel_size[1::],
                                     padding=self.padding[1::],
                                     dilation=self.dilation[1::],
                                     stride=self.stride[1::],
                                     bias=False)
            conv3d_layer.weight = nn.Parameter(self.weight[:, :, i, :, :])

            # Store the layer
            self.conv3d_layers.append(conv3d_layer)

        del self.weight


    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        # Define shortcut names for dimensions of input and kernel
        (Batch, _, l_i, d_i, h_i, w_i) = tuple(input.shape)
        (l_k, d_k, h_k, w_k) = self.kernel_size
        (l_p, d_p, h_p, w_p) = self.padding
        (l_d, d_d, h_d, w_d) = self.dilation
        (l_s, d_s, h_s, w_s) = self.stride

        # Compute the size of the output tensor based on the zero padding
        l_o = (l_i + 2 * l_p - (l_k) - (l_k-1) * (l_d-1))//l_s + 1
        d_o = (d_i + 2 * d_p - (d_k) - (d_k-1) * (d_d-1))//d_s + 1
        h_o = (h_i + 2 * h_p - (h_k) - (h_k-1) * (h_d-1))//h_s + 1
        w_o = (w_i + 2 * w_p - (w_k) - (w_k-1) * (w_d-1))//w_s + 1

        # Pre-define output tensors
        out = torch.zeros(Batch, self.out_channels, l_o, d_o, h_o, w_o).to(input.device)

        # Convolve each kernel frame i with each input frame j
        for i in range(l_k):
            # Calculate the zero-offset of kernel frame i
            zero_offset = - l_p + (i * l_d)
            # Calculate the range of input frame j corresponding to kernel frame i
            j_start = max(zero_offset % l_s, zero_offset)
            j_end = min(l_i, l_i + l_p - (l_k-i-1)*l_d)
            # Convolve each kernel frame i with corresponding input frame j
            for j in range(j_start, j_end, l_s):
                # Calculate the output frame
                out_frame = (j - zero_offset) // l_s
                # Add results to this output frame
                out[:, :, out_frame, :, :, :] += self.conv3d_layers[i](input[:, :, j, :, :])

        # Add bias to output
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1, 1)

        return out


class ImageToPatch2D(nn.Module):
    """
    将2D图像转换为Patch Embedding。

    参数:
        img_dims (tuple[int]): 图像尺寸。
        patch_dims (tuple[int]): Patch的尺寸。
        in_channels (int): 输入图像的通道数。
        out_channels (int): 投影后的通道数。
        normalization_layer (nn.Module, optional): 归一化层，默认为None。
    """

    def __init__(self, img_dims, patch_dims, in_channels, out_channels, normalization_layer=None):
        super().__init__()
        self.img_dims = img_dims
        height, width = img_dims
        patch_h, patch_w = patch_dims

        padding_top = padding_bottom = padding_left = padding_right = 0

        # 计算高度和宽度的余数
        height_mod = height % patch_h
        width_mod = width % patch_w

        if height_mod:
            pad_height = patch_h - height_mod
            padding_top = pad_height // 2
            padding_bottom = pad_height - padding_top

        if width_mod:
            pad_width = patch_w - width_mod
            padding_left = pad_width // 2
            padding_right = pad_width - padding_left

        # 添加填充层
        self.padding = nn.ZeroPad2d((padding_left, padding_right, padding_top, padding_bottom))
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=patch_dims, stride=patch_dims)

        # 可选的归一化层
        if normalization_layer is not None:
            self.normalization = normalization_layer(out_channels)
        else:
            self.normalization = None

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert H == self.img_dims[0] and W == self.img_dims[1], \
            f"输入图像尺寸 ({H}x{W}) 与模型预期 ({self.img_dims[0]}x{self.img_dims[1]}) 不符。"
        x = self.padding(x)
        x = self.projection(x)
        if self.normalization is not None:
            x = self.normalization(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class ImageToPatch3D(nn.Module):
    """
    将3D图像转换为Patch Embedding。

    参数:
        img_dims (tuple[int]): 图像尺寸。
        patch_dims (tuple[int]): Patch的尺寸。
        in_channels (int): 输入图像的通道数。
        out_channels (int): 投影后的通道数。
        normalization_layer (nn.Module, optional): 归一化层，默认为None。
    """

    def __init__(self, img_dims, patch_dims, in_channels, out_channels, normalization_layer=None):
        super().__init__()
        self.img_dims = img_dims
        depth, height, width = img_dims
        patch_d, patch_h, patch_w = patch_dims

        padding_front = padding_back = padding_top = padding_bottom = padding_left = padding_right = 0

        # 计算深度、高度和宽度的余数
        depth_mod = depth % patch_d
        height_mod = height % patch_h
        width_mod = width % patch_w

        if depth_mod:
            pad_depth = patch_d - depth_mod
            padding_front = pad_depth // 2
            padding_back = pad_depth - padding_front

        if height_mod:
            pad_height = patch_h - height_mod
            padding_top = pad_height // 2
            padding_bottom = pad_height - padding_top

        if width_mod:
            pad_width = patch_w - width_mod
            padding_left = pad_width // 2
            padding_right = pad_width - padding_left

        # 添加填充层
        self.padding = nn.ZeroPad3d(
            (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
        )
        self.projection = nn.Conv3d(in_channels, out_channels, kernel_size=patch_dims, stride=patch_dims)

        # 可选的归一化层
        if normalization_layer is not None:
            self.normalization = normalization_layer(out_channels)
        else:
            self.normalization = None

    def forward(self, x: torch.Tensor):
        B, C, D, H, W = x.shape #
        assert C == self.img_dims[0] and H == self.img_dims[1] and W == self.img_dims[2], \
            f"输入图像尺寸 ({D}x{H}x{W}) 与模型预期 ({self.img_dims[0]}x{self.img_dims[1]}x{self.img_dims[2]}) 不符。"
        x = self.padding(x)
        x = self.projection(x)
        if self.normalization:
            x = self.normalization(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x


class ImageToPatch4D(nn.Module):
    """
    将4D图像转换为Patch Embedding。

    参数:
        img_dims (tuple[int]): 图像尺寸（时间、深度、高度、宽度）。
        patch_dims (tuple[int]): Patch的尺寸。
        in_channels (int): 输入图像的通道数。
        out_channels (int): 投影后的通道数。
        normalization_layer (nn.Module, optional): 归一化层，默认为None。
    """

    def __init__(self, img_dims, patch_dims, in_channels, out_channels, normalization_layer=None):
        super().__init__()
        self.img_dims = img_dims
        time, depth, height, width = img_dims
        patch_t, patch_d, patch_h, patch_w = patch_dims

        # 初始化填充变量
        padding_time_front = padding_time_back = padding_depth_front = padding_depth_back = 0
        padding_top = padding_bottom = padding_left = padding_right = 0

        # 计算每个维度的余数并设置填充
        time_mod = time % patch_t
        depth_mod = depth % patch_d
        height_mod = height % patch_h
        width_mod = width % patch_w

        if time_mod:
            pad_time = patch_t - time_mod
            padding_time_front = pad_time // 2
            padding_time_back = pad_time - padding_time_front

        if depth_mod:
            pad_depth = patch_d - depth_mod
            padding_depth_front = pad_depth // 2
            padding_depth_back = pad_depth - padding_depth_front

        if height_mod:
            pad_height = patch_h - height_mod
            padding_top = pad_height // 2
            padding_bottom = pad_height - padding_top

        if width_mod:
            pad_width = patch_w - width_mod
            padding_left = pad_width // 2
            padding_right = pad_width - padding_left

        # 填充层
        self.padding = nn.ConstantPad3d(
            (padding_left, padding_right, padding_top, padding_bottom,
             padding_depth_front, padding_depth_back, padding_time_front, padding_time_back),
            0
        )

        # Conv4d 投影层
        self.projection = Conv4d(in_channels, out_channels, kernel_size=patch_dims, stride=patch_dims)

        # 可选归一化层
        if normalization_layer is not None:
            self.normalization = normalization_layer(out_channels)
        else:
            self.normalization = None

    def forward(self, x: torch.Tensor):
        B, C, T, D, H, W = x.shape
        assert C == self.img_dims[0] and T == self.img_dims[1] and H == self.img_dims[2] and W == self.img_dims[3], \
            f"输入图像尺寸 ({C}x{T}x{H}x{W}) 与模型预期 ({self.img_dims[0]}x{self.img_dims[1]}x{self.img_dims[2]}x{self.img_dims[3]}) 不符。"
        x = self.padding(x)
        x = self.projection(x)
        if self.normalization:
            x = self.normalization(x.permute(0, 2, 3, 4, 5, 1)).permute(0, 5, 1, 2, 3, 4)
        return x