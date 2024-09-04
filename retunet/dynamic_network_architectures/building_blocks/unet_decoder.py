from functools import partial

import numpy as np
import torch
from timm.layers import DropPath
from torch import nn
from typing import Union, List, Tuple, Type

from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

from retunet.dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from retunet.dynamic_network_architectures.attentions.RetSA import BasicLayer
import torch.nn.functional as F
from torch.utils import checkpoint
from retunet.dynamic_network_architectures.attentions.ViT import Transformer
from retunet.dynamic_network_architectures.attentions.window_attention import SwinTransformerLayer


def containsNan(tensor):
    nan_mask = torch.isnan(tensor)
    # 使用torch.any()检查是否有任何NaN值
    contains_nan = torch.any(nan_mask)
    return contains_nan.item()
class UNetDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None,
                 patch_size: Union[int, Tuple[int, ...], List[int]] = None,
                 grid_diff:dict =None
                 ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs


        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]

            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            # print(f"input_features_skip:{input_features_skip}")
            # print(f"input_features_below:{input_features_below}")
            # attentions.append(BCA(input_features_below,input_features_below,input_features_below))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))
        attentions = []
        embed_dims = []
        ViTLayers = []

        depths = (2, 8, 2)
        num_heads = (16, 8, 4)
        init_values = (2, 2, 2)
        heads_ranges = (6, 6, 4)
        chunkwise_recurrent = (False, True, True)
        self.num_layers = 3
        # for i in range(n_stages_encoder):
        #     patch_sizes.append(patch_size)
        #     patch_size = (patch_size[0] // 2, patch_size[1] // 2)
        for index in range(self.num_layers):

            embed_dims.append(encoder.output_channels[-(index+2)])

            attentions.append(BasicLayer(embed_dim=embed_dims[index],
                                         out_dim=embed_dims[index] if index < self.num_layers - 1 else None,
                                         depth=depths[index],
                                         num_heads=num_heads[index], init_value=init_values[index],
                                         heads_range=heads_ranges[index],
                                         chunkwise_recurrent=chunkwise_recurrent[index], ffn_dim=3 * embed_dims[index],
                                         drop_path=0.3,grid_diff = grid_diff))
            ViTLayers.append(
                Transformer(dim=embed_dims[index], depth=depths[index], heads=num_heads[index], dim_head=64, mlp_dim=96,
                            dropout=0.3))
        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.attentions = nn.Sequential(*attentions)
        self.ViTLayers = nn.Sequential(*ViTLayers)
        # self.SwinTLayers = nn.Sequential(*SwinTLayers)
        self.call_num = 0
    def forward(self, skips):
        # 从最深层的编码器输出开始
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            # 对最深层的输入进行上采样
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.call_num == 0:
                print(x.shape)
            if s < self.num_layers:
                x = self.attentions[s](x)
                # x = self.ViTLayers[s](x)
                # x = self.SwinTLayers[s](x)
            # 如果启用深度监督或者在最后一层，添加
            # 分割输出
            if self.deep_supervision or s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[s if self.deep_supervision else -1](x))

            # 准备下一层的输入
            lres_input = x
        self.call_num = 1
        # 调整输出顺序，最大的分割预测在前
        seg_outputs = seg_outputs[::-1]

        return seg_outputs[0] if not self.deep_supervision else seg_outputs


    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output