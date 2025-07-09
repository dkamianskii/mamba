import torch
import torch.nn.functional as F

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaGeneralConfig


batch = 10
seqlen = 64
device = "cpu"
dtype = torch.float32

config = MambaGeneralConfig(
    d_model=512,
    n_layer=4,
    vocab_size=256,
    ssm_cfg=dict(layer="Mamba2"),
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    pad_vocab_size_multiple=16,
)
torch.manual_seed(2357)
model = MambaLMHeadModel(config, device=device, dtype=dtype)

traced_script_module = torch.jit.script(model)

traced_script_module.save("model_name.pt")