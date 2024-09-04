import copy
from functools import partial

import numpy as np
import torch
from sklearn.neighbors import KDTree, BallTree
from timm.layers import DropPath
from torch import nn
from typing import Tuple

import torch.nn.functional as F
from torch.utils import checkpoint


def containsNan(tensor):
    nan_mask = torch.isnan(tensor)
    contains_nan = torch.any(nan_mask)
    return contains_nan.item()
class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2)  # (b h w c)
        x = self.conv(x)  # (b c h w)
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        return x

class RelPos2d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range,grid_diff):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)
        self.cache = {}
        self.grid_diff = copy.deepcopy(grid_diff)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_2d_decay(self, H, W):

        key = (H, W)
        grid_diff = self.grid_diff[key]
        mask = -grid_diff * self.decay[:, None, None]
        return mask
        # index_h = torch.arange(H).to(self.decay)
        # index_w = torch.arange(W).to(self.decay)
        # grid = torch.meshgrid([index_h, index_w])
        # grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  # (H*W 2)
        # mask = grid[:, None, :] - grid[None, :, :]  # (H*W H*W 2)
        # mask = (mask ** 2).sum(dim=-1).sqrt()
        # mask = -mask * self.decay[:, None, None]  # (n H*W H*W)
        return mask

    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]  # (l l)
        mask = mask.abs()  # (l l)
        mask = -mask * self.decay[:, None, None]  # (n l l)
        return mask

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:

            retention_rel_pos = self.decay.exp()

        elif chunkwise_recurrent:
            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])

            retention_rel_pos = (mask_h, mask_w)

        else:
            mask = self.generate_2d_decay(slen[0], slen[1])  # (n l l)
            retention_rel_pos = mask

        return retention_rel_pos
class MaSAd(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)

        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        '''
        bsz, h, w, _ = x.size()

        mask_h, mask_w = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        qr = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        kr = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)

        '''
        qr: (b n h w d1)
        kr: (b n h w d1)
        v: (b h w n*d2)
        '''

        qr_w = qr.transpose(1, 2)  # (b h n w d1)
        kr_w = kr.transpose(1, 2)  # (b h n w d1)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)  # (b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)  # (b h n w w)
        qk_mat_w = qk_mat_w + mask_w  # (b h n w w)
        qk_mat_w = torch.softmax(qk_mat_w, -1)  # (b h n w w)
        v = torch.matmul(qk_mat_w, v)  # (b h n w d2)

        qr_h = qr.permute(0, 3, 1, 2, 4)  # (b w n h d1)
        kr_h = kr.permute(0, 3, 1, 2, 4)  # (b w n h d1)
        v = v.permute(0, 3, 2, 1, 4)  # (b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)  # (b w n h h)
        qk_mat_h = qk_mat_h + mask_h  # (b w n h h)
        qk_mat_h = torch.softmax(qk_mat_h, -1)  # (b w n h h)
        output = torch.matmul(qk_mat_h, v)  # (b w n h d2)

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)  # (b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output

class RetSA(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        rel_pos: mask: (n l l)
        '''
        bsz, h, w, _ = x.size()
        mask = rel_pos

        assert (h * w == mask.size(1),f'h*w={h*w} but mask={mask}')

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        qr = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        kr = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d1)

        qr = qr.flatten(2, 3)  # (b n l d1)
        kr = kr.flatten(2, 3)  # (b n l d1)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d2)
        vr = vr.flatten(2, 3)  # (b n l d2)
        qk_mat = qr @ kr.transpose(-1, -2)  # (b n l l)
        qk_mat = qk_mat + mask  # (b n l l)
        qk_mat = torch.softmax(qk_mat, -1)  # (b n l l)
        output = torch.matmul(qk_mat, vr)  # (b n l d2)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1)  # (b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output




class FeedForwardNetwork(nn.Module):
    def __init__(
            self,
            embed_dim,
            ffn_dim,
            activation_fn=F.gelu,
            dropout=0.0,
            activation_dropout=0.0,
            layernorm_eps=1e-6,
            subln=False,
            subconv=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        if self.dwconv is not None:
            residual = x
            x = self.dwconv(x)
            x = x + residual
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


class RetBlock(nn.Module):

    def __init__(self, retention: str, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False,
                 layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        assert retention in ['chunk', 'whole']
        if retention == 'chunk':
            self.retention = MaSAd(embed_dim, num_heads)
        else:
            self.retention = RetSA(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)

    def forward(
            self,
            x: torch.Tensor,
            incremental_state=None,
            chunkwise_recurrent=False,
            retention_rel_pos=None
    ):

        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(
                self.gamma_1 * self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent,
                                              incremental_state))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(
                self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim),
            nn.Conv2d(embed_dim, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        x = self.proj(x)
        assert containsNan(x) == False, f"stage_proj:containsNan{containsNan(x)}"
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        '''
        x: B H W C
        '''
        x = x.permute(0, 3, 1, 2).contiguous()  #(b c h w)
        x = self.reduction(x) #(b oc oh ow)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) #(b oh ow oc)
        return x
class BasicLayer(nn.Module):

    def __init__(self, embed_dim, out_dim, depth, num_heads,
                 init_value: float, heads_range: float,
                 ffn_dim=96., drop_path=0., norm_layer=nn.LayerNorm, chunkwise_recurrent=False,
                 downsample: PatchMerging = None, use_checkpoint=False,
                 layerscale=False, layer_init_values=1e-5,grid_diff=None):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.chunkwise_recurrent = chunkwise_recurrent
        if chunkwise_recurrent:
            flag = 'chunk'
        else:
            flag = 'whole'
        self.Relpos = RelPos2d(embed_dim, num_heads, init_value, heads_range,grid_diff)

        # build blocks
        self.blocks = nn.ModuleList([
            RetBlock(flag, embed_dim, num_heads, ffn_dim,
                     drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_values)
            for i in range(depth)])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):

        assert containsNan(x)==False,f"stage_conv:containsNan{containsNan(x)}"
        x = x.permute(0, 2, 3, 1)  # B,H,W,C
        b, h, w, d = x.size()
        rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
        for i,blk in enumerate(self.blocks):
            if self.use_checkpoint:
                tmp_blk = partial(blk, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent,
                                  retention_rel_pos=rel_pos)
                x = checkpoint.checkpoint(tmp_blk, x)
            else:
                x = blk(x, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent,
                        retention_rel_pos=rel_pos)
                if containsNan(x):
                    print(f"stage{i}:containsNan{containsNan(x)}")
        if self.downsample is not None:
            x = self.downsample(x)
        x = x.permute(0, 3, 1, 2)
        return x