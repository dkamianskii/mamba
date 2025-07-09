from dataclasses import dataclass, field
from typing import Optional, Tuple
from abc import ABC

import torch
from torch.distributed import ProcessGroup


@dataclass
class AttentionConfig:
    pass

class SSMConfig(ABC):
    def __init__(self,
                 layer: str,
                 d_state: int,
                 d_conv: int,
                 expand: float,
                 dt_min: float,
                 dt_max: float,
                 dt_init_floor: float,
                 conv_bias: bool,
                 bias: bool):
        self.layer = layer
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.bias = bias
        self.conv_bias = conv_bias


class Mamba1Config(SSMConfig):
    def __init__(self,
                 d_state: int=16,
                 d_conv: int=4,
                 expand: float=2,
                 dt_min: float=0.001,
                 dt_max: float=0.1,
                 dt_init_floor: float=1e-4,
                 conv_bias=True,
                 bias: bool=False,
                 dt_rank="auto",
                 dt_init="random",
                 dt_scale=1.0,
                 use_fast_path=True):
        super().__init__("Mamba1", d_state, d_conv, expand, dt_min, dt_max, dt_init_floor, conv_bias, bias)
        self.use_fast_path = use_fast_path
        self.dt_scale = dt_scale
        self.dt_init = dt_init
        self.dt_rank = dt_rank

class Mamba2Config(SSMConfig):
    def __init__(self,
                 d_state: int=128,
                 d_conv: int=4,
                 expand: float=2,
                 dt_min: float=0.001,
                 dt_max: float=0.1,
                 dt_init_floor: float=1e-4,
                 conv_bias=True,
                 bias: bool=False,
                 headdim: int=64,
                 d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
                 ngroups: int=1,
                 A_init_range: Tuple[int, int]=(1, 16),
                 D_has_hdim: bool=False,
                 rmsnorm: bool=True,
                 norm_before_gate: bool=False,
                 dt_limit: Tuple[float, float]=(0.0, float("inf")),
                 chunk_size: int=256,
                 use_mem_eff_path: bool=True,
                 process_group: Optional[ProcessGroup]=None,
                 sequence_parallel: bool=True
                 ):
        super().__init__("Mamba2", d_state, d_conv, expand, dt_min, dt_max, dt_init_floor, conv_bias, bias)
        self.headdim = headdim
        self.sequence_parallel = sequence_parallel
        self.process_group = process_group
        self.use_mem_eff_path = use_mem_eff_path
        self.chunk_size = chunk_size
        self.dt_limit = dt_limit
        self.norm_before_gate = norm_before_gate
        self.rmsnorm = rmsnorm
        self.D_has_hdim = D_has_hdim
        self.A_init_range = A_init_range
        self.ngroups = ngroups
        self.d_ssm = d_ssm


@dataclass
class MambaGeneralConfig:

    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: SSMConfig = field(default_factory=Mamba1Config)
    # attn_layer_idx: list = field(default_factory=list)
    # attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True


@dataclass
class MixerArgs:
    seqlen: Optional[int] = None
    seq_idx: Optional[int] = None
    cu_seqlens: Optional[torch.Tensor] = None
