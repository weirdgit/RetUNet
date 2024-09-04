import math
from functools import partial

import torch
from torch import nn
import numpy as np
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from torch.utils import checkpoint

from retunet.dynamic_network_architectures.attentions.RetSA import BasicLayer, PatchEmbed
from retunet.dynamic_network_architectures.attentions.ViT import Transformer

def containsNan(tensor):
    nan_mask = torch.isnan(tensor)
    # 使用torch.any()检查是否有任何NaN值
    contains_nan = torch.any(nan_mask)
    return contains_nan.item()


class PlainConvEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = 'conv',
                 patch_size: Union[int, Tuple[int, ...]] = (0,0),
                 grid_diff: dict = None,
                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        masks = {}


        stages = []
        RetLayers = []
        RetLayers_reverse = []
        ViTLayers = []
        SwinTLayers = []
        embed_dims = []
        depths = (2, 2, 8, 2)
        num_heads = (4, 4, 8, 16)
        init_values = (2, 2, 2, 2)
        heads_ranges = (4, 4, 6, 6)
        chunkwise_recurrent = (True, True, False, False)
        self.num_layers = 4
        patch_sizes = []
        for i in range(n_stages):
            patch_sizes.append(patch_size)
            patch_size = (patch_size[0]//2,patch_size[1]//2)
        for index in range(self.num_layers):
            dim = features_per_stage[n_stages + index - self.num_layers]
            embed_dims.append(dim)
            RetLayers_reverse.append(BasicLayer(embed_dim=embed_dims[index],
                                         out_dim=embed_dims[index] if index < self.num_layers - 1 else None,
                                         depth=depths[index],
                                         num_heads=num_heads[index], init_value=init_values[index],
                                         heads_range=heads_ranges[index],
                                         chunkwise_recurrent=chunkwise_recurrent[index], ffn_dim=3 * input_channels,
                                         drop_path=0.3, grid_diff=grid_diff))
            ViTLayers.append(Transformer(dim=embed_dims[index], depth=depths[index], heads=num_heads[index], dim_head =64, mlp_dim=96, dropout = 0.3))
            # SwinTLayers.append(SwinTransformerLayer(dim=embed_dims[index], input_resolution=patch_sizes[-4+index],
            #                                         depth=depths[index], num_heads=num_heads[index], window_size=7,
            #      mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.3,
            #      drop_path=0.3, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
            #      fused_window_process=False))
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules.append(StackedConvBlocks(
                n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ))
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
        self.ret_layers = nn.Sequential(*RetLayers)
        self.reverse_ret_layers = nn.Sequential(*RetLayers_reverse)
        self.forward_features = PatchEmbed(in_chans=1, embed_dim=embed_dims[0])
        self.call_count = 0
        self.ViTLayers = nn.Sequential(*ViTLayers)
        # self.SwinTLayers = nn.Sequential(*SwinTLayers)

    def forward(self, x):
        ret = []
        for i, s in enumerate(self.stages):
            # print(f"stage{i}:{x.shape}")
            n = len(self.stages)
            x = s(x)
            if self.call_count == 0:
                print(f"stage{i}:{x.shape}")
            if n - self.num_layers < i < n:
                x = self.reverse_ret_layers[i - (n - self.num_layers)](x)
                # x = self.ViTLayers[i-(n-4)](x)
                # x = self.SwinTLayers[i-(n-4)](x)
            ret.append(x)

        self.call_count = 1
        if self.return_skips:

            assert containsNan(ret[-1]) == False, f"stage_conv:containsNan"
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output
