# Modified from PyTorch nn.Transformer

from typing import List, Callable, Optional, Tuple
import numpy as np, torch, torch.nn.functional as F, torch.nn as nn
from torch import Tensor
from matanyone2.model.channel_attn import CAResBlock
from torch.nn.functional import scaled_dot_product_attention as SDPA
from torch.nn.attention import sdpa_kernel, SDPBackend
from einops.layers.torch import Rearrange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
THETA = 30000.0
def _setup_tf32() -> None:
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.enable_flash_sdp(enabled=True)
            torch.backends.cuda.enable_cudnn_sdp(enabled=True)
         
_setup_tf32()

def gammatone(dims, head, min_freq=200.0, max_freq=8000.0):
    head_dim = dims // head
    f = torch.pow(max_freq / min_freq, torch.linspace(0, 1, head_dim // 2, device=device, dtype=dtype)) * min_freq
    return f / 1000

def wideband(dims, head, max_freq=8000.0):
    head_dim = dims // head
    mel_max = 2595 * torch.log10(torch.tensor(1 + max_freq / 700, device=device, dtype=dtype))
    mel_scale = torch.pow(10, torch.linspace(0, mel_max, head_dim // 2, device=device, dtype=dtype) / 2595) - 1
    return 700 * mel_scale / 1000

def have(a):
    return a is not None  

def aorb(a, b):
    return a if have(a) else b

def aborc(a, b, c):
    return aorb(a, aorb(b, c))

def abcord(a, b, c, d):
    return aorb(a, aborc(b, c, d))

def no_none(x):
    return x.apply(lambda tensor: tensor if tensor is not None else None)

def l2norm(t):
    return F.normalize(t, dim = -1)

def exact_div(x, y):
    assert x % y == 0
    return x // y

class rotary(nn.Module):
    def __init__(n, dims, head):
        super().__init__()

        n.head_dim = dims // head
        n.head = head
        n.dims = dims
        n.lin = nn.Linear(dims, n.head_dim // 2, bias=True)

    def gammatone(n, min_freq=200.0, max_freq=8000.0):
        head_dim = n.dims // n.head
        freqs = torch.pow(max_freq / min_freq, torch.linspace(0, 1, head_dim // 2, device=device, dtype=dtype)) * min_freq
        return freqs / 1000

    def wideband(n, max_freq=8000.0):
        head_dim = n.dims // n.head
        mel_max = 2595 * torch.log10(torch.tensor(1 + max_freq / 700, device=device, dtype=dtype))
        mel_scale = torch.pow(10, torch.linspace(0, mel_max, head_dim // 2, device=device, dtype=dtype) / 2595) - 1
        return 700 * mel_scale / 1000

    def compute_f(n, x=None, mask=None):
        if mask is None:
            scale = gammatone(n.dims, n.head)
            return x.mean(dim=-1) * scale / 1000 if x is not None else 200 * scale / 1000
        else: 
            return torch.arange(0, n.head_dim, 2, device=device, dtype=dtype, requires_grad=False) / n.head_dim * torch.log(torch.tensor(x.mean(dim=-1) * THETA if x is not None else THETA, requires_grad=False))

    def forward(n, x=None, xa=None, mask=None): 
        t = torch.arange(x.shape[2], device=device, dtype=dtype).float()
        f = torch.einsum('i,j->ij', t,  n.compute_f(mask=mask))
        m = torch.norm(xa, dim=-1, keepdim=True)
        # m = n.lin(xa)
        # m = torch.sigmoid(n.lin(xa)) ** t
        # m = torch.sigmoid(n.lin(xa)) 

        if mask is None:
            f = torch.polar(m, f)
        else: 
            f = torch.polar(m, f)

        x1 = x[..., :f.shape[-1]*2]
        x2 = x[..., f.shape[-1]*2:]
        s = x1.shape
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
        x1 = torch.view_as_complex(x1) * f
        x1 = torch.view_as_real(x1).flatten(-2)
        x1 = x1.view(s)
        return torch.cat([x1.type_as(x), x2], dim=-1)

class AbbyNormal(nn.Module):
    def __init__(n, dims, size: int = 5, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0, threshold: float = 0.8):
        super().__init__()
        n.size = size
        n.alpha = alpha
        n.beta = beta
        n.k = k
        n.tx = threshold
        
        n.mode_router = nn.Sequential(
            nn.Linear(dims, dims),
            nn.SiLU(),
            nn.Linear(dims, 3) 
        )

    def forward(n, x: Tensor, confidence=None) -> Tensor:
        if x.numel() == 0:
            return x

        size = max(3, int(x.size(-1) * 0.05))
        if size % 2 == 0:
            size += 1
        pad_len = size // 2
        
        div = x.mul(x)
        logits = n.mode_router(x)
        mean_val = x.abs().mean(dim=-1, keepdim=True)
        std_val = x.std(dim=-1, keepdim=True)
        cv = std_val / (mean_val + 1e-6)

        decisions = F.gumbel_softmax(logits + cv, tau=1.0, hard=True) 
        avg_d = F.avg_pool1d(div.squeeze(0), kernel_size=size, stride=1, padding=pad_len)
        max_d = F.max_pool1d(div.squeeze(0), kernel_size=size, stride=1, padding=pad_len)
  
        div_mode1 = avg_d
        condition = (max_d > 2.0 * avg_d).float()
        div_mode2 = (condition * max_d) + ((1 - condition) * avg_d)
        
        if confidence is None:
            div_mode3 = avg_d
        else:
            conf_mask = (confidence > n.tx).float().unsqueeze(1)
            div_mode3 = (conf_mask * avg_d) + ((1 - conf_mask) * max_d)

        d0 = decisions[..., 0:1] 
        d1 = decisions[..., 1:2] 
        d2 = decisions[..., 2:3] 
        
        div = (d0 * div_mode1) + (d1 * div_mode2) + (d2 * div_mode3)
        denom = div.mul(n.alpha).add(n.k).pow(n.beta)
        out = x / denom
        return out

class ConvLite(nn.Module):
    def __init__(n, dims, kernel_size=15): 
        super().__init__()
        n.point1 = nn.Conv1d(dims, dims * 2, kernel_size=1)
        n.glu = nn.GLU(dim=1) 
        
        n.depth = nn.Conv1d(
            dims, dims, kernel_size=kernel_size, 
            padding=(kernel_size - 1) // 2, groups=dims
        )
        n.bn = nn.BatchNorm1d(dims) 
        n.swish = nn.SiLU() 
        
        n.point2 = nn.Conv1d(dims, dims, kernel_size=1)
        n.dropout = nn.Dropout(0.1)

    def forward(n, x):
        residual = x
        x = n.point1(x)
        x = n.glu(x)
        x = n.depth(x)
        x = n.bn(x)
        x = n.swish(x)
        x = n.point2(x)
        x = n.dropout(x)
        return residual + x

class LocalNorm(nn.Module):
    def __init__(self, size: int = 5, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0, mode: str = '1', threshold: float = 0.8):
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.mode = mode
        self.threshold = threshold

    def forward(self, input: Tensor, confidence=None) -> Tensor:
        if input.numel() == 0:
            return input

        div = input.mul(input).unsqueeze(1) 
        
        pad_len = self.size // 2

        if self.mode == "1":
            div = F.avg_pool1d(div, kernel_size=self.size, stride=1, padding=pad_len)

        elif self.mode == "2":
            avg_d = F.avg_pool1d(div, kernel_size=self.size, stride=1, padding=pad_len)
            max_d = F.max_pool1d(div, kernel_size=self.size, stride=1, padding=pad_len)
            condition = (max_d > 2.0 * avg_d).float()
            div = (condition * max_d) + ((1 - condition) * avg_d)

        elif self.mode == "3":
            avg_d = F.avg_pool1d(div, kernel_size=self.size, stride=1, padding=pad_len)
            max_d = F.max_pool1d(div, kernel_size=self.size, stride=1, padding=pad_len)
            
            if confidence is None:
                div = avg_d
            else:
                conf_mask = (confidence > self.threshold).float().unsqueeze(1)
                div = (conf_mask * avg_d) + ((1 - conf_mask) * max_d)

        div = div.narrow(2, 0, input.size(1)).squeeze(1)
        denom = div.mul(self.alpha).add(self.k).pow(self.beta)
        return input / denom

class GlobalNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x

class LinearNorm(nn.Module):
    def __init__(n, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, n).__init__()
        n.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(n.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(n, x):
        return n.linear_layer(x)

class LayerNorm(nn.Module):
    def __init__(n, dims, eps=1e-5):
        super().__init__()
        n.dims = dims
        n.eps = eps
        n.gamma = nn.Parameter(torch.ones(dims))
        n.beta = nn.Parameter(torch.zeros(dims))

    def forward(n, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (n.dims,), n.gamma, n.beta, n.eps)
        return x.transpose(1, -1)

class AdaLN(nn.Module):
    def __init__(self, dims):
        super().__init__()

        self.norm = nn.LayerNorm(dims, elementwise_affine=False)

        self.mlp = nn.Sequential(
            nn.Linear(dims, dims),
            nn.SiLU(), 
            nn.Linear(dims, 2 * dims)
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, condition=None):
        if condition is None:
             return self.norm(x)

        scale_bias = self.mlp(condition)
        gamma, beta = torch.chunk(scale_bias, 2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return self.norm(x) * (1 + gamma) + beta

class AbbyNormal(nn.Module):
    def __init__(n, dims, size: int = 5, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0, threshold: float = 0.8):
        super().__init__()
        n.size = size
        n.alpha = alpha
        n.beta = beta
        n.k = k
        n.tx = threshold
        
        n.mode_router = nn.Sequential(
            nn.Linear(dims, dims),
            nn.SiLU(),
            nn.Linear(dims, 3) 
        )

    def forward(n, x: Tensor, confidence=None) -> Tensor:
        if x.numel() == 0:
            return x

        size = max(3, int(x.size(-1) * 0.05))
        if size % 2 == 0:
            size += 1
        pad_len = size // 2
        
        div = x.mul(x)
        logits = n.mode_router(x)
        mean_val = x.abs().mean(dim=-1, keepdim=True)
        std_val = x.std(dim=-1, keepdim=True)
        cv = std_val / (mean_val + 1e-6)

        decisions = F.gumbel_softmax(logits + cv, tau=1.0, hard=True) 
        avg_d = F.avg_pool1d(div.squeeze(0), kernel_size=size, stride=1, padding=pad_len)
        max_d = F.max_pool1d(div.squeeze(0), kernel_size=size, stride=1, padding=pad_len)
  
        div_mode1 = avg_d
        condition = (max_d > 2.0 * avg_d).float()
        div_mode2 = (condition * max_d) + ((1 - condition) * avg_d)
        
        if confidence is None:
            div_mode3 = avg_d
        else:
            conf_mask = (confidence > n.tx).float().unsqueeze(1)
            div_mode3 = (conf_mask * avg_d) + ((1 - conf_mask) * max_d)

        d0 = decisions[..., 0:1] 
        d1 = decisions[..., 1:2] 
        d2 = decisions[..., 2:3] 
        
        div = (d0 * div_mode1) + (d1 * div_mode2) + (d2 * div_mode3)
        denom = div.mul(n.alpha).add(n.k).pow(n.beta)
        out = x / denom
        return out

def get_norm(n_type: str, dims: Optional[int] = None, num_groups: Optional[int] = None)-> nn.Module:

    if n_type in ["batchnorm", "instancenorm"] and dims is None:
        raise ValueError(f"'{n_type}' requires 'dims'.")
    if n_type == "groupnorm" and num_groups is None:
        raise ValueError(f"'{n_type}' requires 'num_groups'.")

    norm_map = {
        # "layernorm": lambda: nn.LayerNorm(normalized_shape=dims, bias=False),
        "layernorm": lambda: LayerNorm(dims=dims),
        "linearnorm": lambda: LinearNorm(in_dim=dims, out_dim=dims, bias=False),
        "adanorm": lambda: AdaLN(dims=dims),
        "instancenorm": lambda: nn.InstanceNorm1d(num_features=dims, affine=False, track_running_stats=False),     
        "rmsnorm": lambda: nn.RMSNorm(normalized_shape=dims),        
        "batchnorm": lambda: nn.BatchNorm1d(num_features=dims),
        "instancenorm2d": lambda: nn.InstanceNorm2d(num_features=dims),
        "groupnorm": lambda: nn.GroupNorm(num_groups=num_groups, num_channels=dims),
        "localnorm": lambda: LocalNorm(size=5),
        "AbbyNormal": lambda: AbbyNormal(dims, size = 5, alpha = 1e-4, beta = 0.75, k = 1.0, threshold = 0.8),
        }
   
    norm_func = norm_map.get(n_type)
    if norm_func:
        return norm_func()
    else:
        print(f"Warning: Norm type '{n_type}' not found. Returning LayerNorm.")
        return nn.LayerNorm(dims) 

def get_activation(act: str) -> nn.Module:

    act_map = {
        "gelu": nn.GELU(), 
        "relu": nn.ReLU(), 
        "sigmoid": nn.Sigmoid(), 
        "tanh": nn.Tanh(), 
        "swish": nn.SiLU(), 
        "tanhshrink": nn.Tanhshrink(), 
        "softplus": nn.Softplus(), 
        "softshrink": nn.Softshrink(), 
        "leaky_relu": nn.LeakyReLU(), 
        "elu": nn.ELU(),
    }
    return act_map.get(act, nn.GELU())


def create_attention_mask(batch_size, ctx, is_causal=True, padding_mask=None, device=None):
    if is_causal:
        mask = torch.triu(torch.ones((ctx, ctx), device=device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, ctx, ctx)
    else:
        mask = torch.zeros((batch_size, 1, ctx, ctx), device=device).bool()
    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        mask = mask | (~padding_mask)
    return mask.contiguous()

class BaseAttention(nn.Module):
    use_sdpa = True
    
    def __init__(self, dims: int, head: int, max_dist: int = 512):
        super().__init__()
        assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.max_dist = max_dist
        self.scale = self.head_dim ** -0.25
        
    def _shape(self, tensor: torch.Tensor, ctx: int, batch: int):
        return tensor.view(batch, ctx, self.head, self.head_dim).transpose(1, 2).contiguous()
        
    def _reshape_to_output(self, attn_output, batch, ctx):
        return attn_output.permute(0, 2, 1, 3).reshape(batch, ctx, self.dims)

def calculate_attention(q, k, v, mask=None, temperature=1.0, use_sdpa=True, is_causal=True):
    batch_size = q.shape[0]
    ctx = q.shape[2]
    attn_mask = None
    if mask is not None:
        if mask.dim() <= 3:
            attn_mask = create_attention_mask(
                batch_size=batch_size, 
                ctx=ctx, 
                is_causal=is_causal, 
                padding_mask=mask if mask.dim() > 1 else None,
                device=q.device)
        else:
            attn_mask = mask
    scaled_q = q
    if temperature != 1.0 and temperature > 0:
        scaled_q = q * (1.0 / temperature)**.5

    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION) as context:
        out = torch.nn.functional.scaled_dot_product_attention(
            scaled_q,
            k,
            v,
            attn_mask=attn_mask,
        )

    # with torch.nn.attention.sdpa_kernel(
    #             enable_math=False, enable_flash=False, enable_mem_efficient=True
    #         ):
    #     out = torch.nn.functional.scaled_dot_product_attention(
    #         scaled_q,
    #         k,
    #         v,
    #         attn_mask=attn_mask,
    #     )

        # with sdpa_kernel([SDPBackend.MATH]):
        #     out = scaled_dot_product_attention(scaled_q, k, v, attn_mask=attn_mask, is_causal=False)

    return out

def compute_attention(self, norm_x, mask=None, kv_cache=None, is_causal=True):

    batch, ctx = norm_x.shape[:2]
    
    q = norm_x.view(batch, ctx, self.head, -1).transpose(1, 2)
    k = norm_x.view(batch, ctx, self.head, -1).transpose(1, 2)
    v = norm_x.view(batch, ctx, self.head, -1).transpose(1, 2)

    attn_output, _ = calculate_attention(q, k, v, mask, 1.0, BaseAttention.use_sdpa, is_causal=is_causal)
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch, ctx, -1)
    return attn_output

class AdaptiveSpan(BaseAttention):

    def __init__(self, dims, head, max_dist, sharpen=True, temp_scale=0.01):
        super().__init__(dims, head, max_dist)
        self.sharpen = sharpen
        self.temp_scale = temp_scale
        self.span_scale = nn.Parameter(torch.tensor(1.0))

    def standby_attention(self, norm_x, mask=None, kv_cache=None, is_causal=True):

        batch, ctx = norm_x.shape[:2]
        
        q = norm_x.view(batch, ctx, self.head, -1).transpose(1, 2)
        k = norm_x.view(batch, ctx, self.head, -1).transpose(1, 2)
        v = norm_x.view(batch, ctx, self.head, -1).transpose(1, 2)

        attn_output, _ = calculate_attention(q, k, v, mask, 1.0, BaseAttention.use_sdpa, is_causal=is_causal)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, ctx, -1)
        return attn_output

    def forward(self, x, query=None, key=None, value=None, max_dist=None, max_span=None, span_scale=None, is_causal=True):

        batch, ctx = x.shape[:2]
        query = x.view(batch, ctx, self.head, -1).transpose(1, 2)
        key = x.view(batch, ctx, self.head, -1).transpose(1, 2)
        value = x.view(batch, ctx, self.head, -1).transpose(1, 2)

        if max_dist is None:
            max_dist = self.max_dist
        if max_span is None:
            max_span = query.shape[1]
        if span_scale is None:
            span_scale = self.span_scale
            
        span_mean = span_scale.mean().item()
        span_len = min(int(max_span * span_mean), query.shape[1], key.shape[1], value.shape[1])
        eff_span = min(span_len, max_dist)
        
        if eff_span == 0:
            batch = query.shape[0]
            return (torch.zeros(batch, eff_span, self.dims, device=query.device), None)
            
        q_span = query[:, :eff_span, :]
        k_span = key[:, :eff_span, :]
        v_span = value[:, :eff_span, :]

        batch = q_span.shape[0]

        q = self._shape(q_span, q_span.size(1), batch)
        k = self._shape(k_span, k_span.size(1), batch)
        v = self._shape(v_span, v_span.size(1), batch)

        temperature = (1.0 + self.temp_scale * (1.0 - span_mean)
            if self.sharpen
            else 0.5 + self.temp_scale * span_mean)
        
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            attn_output, weights = calculate_attention(
                q, k, v, None, temperature, BaseAttention.use_sdpa, is_causal=is_causal)
            out = self._reshape_to_output(attn_output, batch, eff_span)
        return out, weights


class attention(nn.Module):
    def __init__(n, dims, head, layer=None, n_type=None, modal=False): 
        super().__init__()
        n.layer = layer
        n.scale = (dims // head) ** -0.25
        n.modal = modal

        n.q   = nn.Sequential(get_norm(n_type, dims), nn.Linear(dims, dims), Rearrange('b c (h d) -> b h c d', h = head))
        n.kv  = nn.Sequential(get_norm(n_type, dims), nn.Linear(dims, dims * 2), Rearrange('b c (kv h d) -> kv b h c d', kv = 2, h = head))
        n.c   = nn.Sequential(get_norm(n_type, dims) , nn.Linear(dims, dims), Rearrange('b c (h d) -> b h c d', h = head))
        n.out = nn.Sequential(Rearrange('b h c d -> b c (h d)'), nn.Linear(dims, dims))

        n.conv = nn.Conv2d(head, head, 1, bias=False) if modal else nn.Identity()
        n.ln = get_norm(n_type, dims // head)
        n.rot = rotary(dims, head)

    def taylor_softmax(x, order=2):
        ta = 1.0
        for i in range(1, order + 1):
            F_i = torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
            ta += x**i / F_i
        return ta / torch.sum(ta, dim=-1, keepdim=True)

    def forward(n, x, xa=None, mask=None, pt=None, window=3, pitch_bias=None): 
        
        b, c, d = x.shape
        k, v = n.kv(aorb(xa, x))
        q = n.q(x)

# potential = ion.mean() + 0.2 * w_metric.mean()

# jump_g = 1.0
# if potential < 0.1 and i < n.layer - 1:
#     action = 1  

        if pitch_bias is not None:
            qk = n.rbf_scores(q * n.scale, k * n.scale, rbf_sigma=1.0, rbf_ratio=0.3)
            pb = pitch_bias(xa) 
            if pb is not None:
                qk = qk + pb[:,:,:q,:q]

            ids = k[:, :, :, 0]
            scale = torch.ones_like(ids)
            fz = torch.clamp(F.softplus(n.fz), n.minz, n.maxz)
            scale[ids.float() == n.pad] = fz
            
            if mask is not None:
                if mask.dim() == 4:
                    mask = mask[0, 0]
                mask = mask[:q, :k] if xa is not None else mask[:q, :q]
                qk = qk + mask * scale.unsqueeze(-2).expand(qk.shape)

            qk = qk * scale.unsqueeze(-2)
            w = F.softmax(qk, dim=-1).to(q.dtype)
            wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

        if pt is not None:
            c = n.c(pt)
            b, h, c, d = q.shape 
            t = torch.zeros(b, h, c, c, device=device, requires_grad=False)

            for i in range(c):
                for j in range(c):
                    start = max(0, min(i, j) - window)
                    end = min(c, max(i, j) + window)
                    
                    for k in range(start, end): 
                        score = (q[:, :, i, :] * k[:, :, j, :] * c[:, :, k, :]).sum(dim=-1)
                        t[:, :, i, j] += score

            q = q * n.scale + t
            k = k * n.scale + t

        else:
            q = q * n.scale 
            k = k * n.scale

        q, k = n.rot(q, xa=x if pt is None else pt, mask=mask), n.rot(k, xa=xa if xa is not None else x, mask=mask)  

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION], set_priority=True):
            a = SDPA(n.ln(q), n.ln(k), v, is_causal=have(mask))

        if n.modal and xa is not None:
            (ka, va), (kb, vb) = n.kv(x), n.kv(xa)
            qa, qb = n.q(x), n.q(xa)
            qa, qb, ka, kb = n.rot(qa), n.rot(qb), n.rot(ka), n.rot(kb)
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION], set_priority=True):
                b = SDPA(n.ln(qa), n.ln(kb), vb, is_causal=have(mask))
                c = SDPA(n.ln(qb), n.ln(ka), va, is_causal=have(mask))
            return n.out(a), n.out(n.conv(b)), n.out(n.conv(c))
        else:
            return n.out(a)

class Multihead1(nn.Module):  # encoder - decoder attention
    cosa = False
    magnitude = False
    sdpa = True
    use_qkv = False # removing qkv projection for now
    
    def __init__(self, dims, head, layer_idx, decoder, dropout=0.0, bias=False):
        tox = {"device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
               "dtype": torch.float32}
        super().__init__()
        # self.qkv = qkv_proj(dims, bias=bias, **tox) if Multihead.use_qkv else n_proj(dims, bias=bias, **tox)
        # super().__init__(embed_dim=dims, num_heads=head, dropout=dropout, bias=bias, kdim=dims, vdim=dims, batch_first=True)
        self.dims = dims
        self.head = head

        self.layer_idx = layer_idx
        self.decoder = decoder
        self.dropout = dropout
        self.bias = bias
        self.head_dim = dims // head
        assert self.dims % self.head == 0, f"{self.dims} must be divisible by {self.head}"
        self.scale = self.head_dim ** -0.5


        if Multihead1.use_qkv:
            self.qkv = nn.Linear(dims, dims * 3, bias=bias, **tox)
        else:
            self.q = nn.Linear(dims, dims, **tox)
            self.k = nn.Linear(dims, dims, bias=bias, **tox)
            self.v = nn.Linear(dims, dims, **tox)
        self.o = nn.Linear(dims, dims, bias=bias, **tox)

        self.print_once = False

    def cos_attention(self, q: Tensor, k: Tensor, v: Tensor, mask) -> Tensor:
        q_norm = torch.nn.functional.normalize(q, dim=-1, eps=1e-12)
        k_norm = torch.nn.functional.normalize(k, dim=-1, eps=1e-12)
        qk_cosine = torch.matmul(q_norm, k_norm.transpose(-1, -2))
        
        if Multihead1.magnitude:
            q_magnitude = torch.norm(q, dim=-1, keepdim=True)
            k_magnitude = torch.norm(k, dim=-1, keepdim=True)
            magnitude_scaling = (q_magnitude * k_magnitude.transpose(-1, -2)) ** 0.5
            magnitude_scaling = torch.clamp(magnitude_scaling, min=1e-8)
            qk_cosine = qk_cosine * magnitude_scaling

        qk_cosine = qk_cosine + mask
        weights = F.softmax(qk_cosine, dim=-1)
        out = torch.matmul(weights, v)
        return out


    def forward(self, x, xa=None, kv_cache=None, mask=None, decoder=False):
        B, L, D = x.shape
        batch, ctx, dims = B, L, D
        if xa is not None:
            batch, ctx, dims = xa.shape

        scale = self.scale 
        is_causal=decoder

        x = xa if xa is not None else x

        if Multihead1.use_qkv:
            result = self.qkv(x)
            q, k, v = torch.chunk(result, 3, dim=-1)
            q_w, k_w, v_w = self.qkv.weight.chunk(3, dim=0)

            if self.bias:
                q_bias, k_bias, v_bias = torch.chunk(self.qkv.bias, 3, dim=0)
            else:
                q_bias, k_bias, v_bias = None, None, None
            q, k, v = (
                F.linear(q, q_w, q_bias),
                F.linear(k, k_w, k_bias),
                F.linear(v, v_w, v_bias))
        else:
            q = self.q(x)
            if kv_cache is None or xa is None or self.k not in kv_cache:
                k = self.k(x if xa is None else xa)
                v = self.v(x if xa is None else xa)
            else:
                k = kv_cache[self.k]
                v = kv_cache[self.v]

            if kv_cache is not None:
                kv_cache[self.k] = k
                kv_cache[self.v] = v

        B, L, D = q.shape
        
        q = q.unflatten(-1, [self.head, self.head_dim]).transpose(1, 2)  # (B, L, D) -> (B, head, L, head_dim)
        k = k.unflatten(-1, [self.head, self.head_dim]).transpose(1, 2)
        v = v.unflatten(-1, [self.head, self.head_dim]).transpose(1, 2)

        if decoder and not xa:
            causal_mask = torch.empty(L, L, device=q.device).fill_(-np.inf).triu_(1)
            self.register_buffer("causal_mask", causal_mask, persistent=False)
            causal_mask = causal_mask.expand(B, self.head, L, L)
            mask = causal_mask if mask is None else mask + causal_mask
            is_causal = True

        else:
            self.register_buffer("causal_mask", None, persistent=False)
            mask = None
            is_causal = False

        if Multihead1.cosa:
            if not self.print_once:
                print("cosa")
            output = self.cos_attention(q, k, v, mask=mask)
            qk = None

        if Multihead1.sdpa:
            if not self.print_once:
                print("sdpa")
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION], set_priority=True):
                output = SDPA(q, k, v, attn_mask=mask, dropout_p=self.dropout, is_causal=is_causal)  # (B, head, L, head_dim)
            qk = None

        else:
            if not self.print_once:
                print("regular")
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:ctx, :ctx]
            qk = qk.float()
            w = F.softmax(qk, dim=-1).to(q.dtype)
            output = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        self.print_once = True
        output = output.transpose(1, 2).flatten(-2)  # (B, L, head * head_dim) -> (B, L, D)
        out = self.o(output)
        return out, qk

      
class MultiheadB(nn.Module):
    scaling = True 
    def __init__(self, dims: int, head: int):
        super().__init__()
        self.head = head
        self.dims = dims
        self.head_dim = dims // head
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dims, dims)
        self.k = nn.Linear(dims, dims, bias=False)
        self.v = nn.Linear(dims, dims)
        self.o = nn.Linear(dims, dims)
        
    def cos_attention(self, q: Tensor, k: Tensor, v: Tensor, mask) -> Tensor:
        q_norm = torch.nn.functional.normalize(q, dim=-1, eps=1e-12)
        k_norm = torch.nn.functional.normalize(k, dim=-1, eps=1e-12)
        qk_cosine = torch.matmul(q_norm, k_norm.transpose(-1, -2))
        
        if MultiheadB.scaling:
            q_magnitude = torch.norm(q, dim=-1, keepdim=True)
            k_magnitude = torch.norm(k, dim=-1, keepdim=True)
            magnitude_scaling = (q_magnitude * k_magnitude.transpose(-1, -2)) ** 0.5
            magnitude_scaling = torch.clamp(magnitude_scaling, min=1e-8)
            qk_cosine = qk_cosine * magnitude_scaling

        qk_cosine = qk_cosine + mask
        weights = F.softmax(qk_cosine, dim=-1)
        out = torch.matmul(weights, v)
        return out
        
    def _shape(self, tensor: torch.Tensor, ctx: int, batch: int):
        return tensor.view(batch, ctx, self.head, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(self, x: Tensor, xa = None, mask = None, kv_cache = None, is_causal=True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, ctx = x.shape[:2]
        q = self.q(x)

        if kv_cache is None or xa is None or self.k not in kv_cache:
            k = self.k(x if xa is None else xa)
            v = self.v(x if xa is None else xa)
        else:
            k = kv_cache[self.k]
            v = kv_cache[self.v]

        wv = self._attention(q, k, v, mask, is_causal=is_causal)
        return self.o(wv), None

    def _attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor, is_causal: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = q.shape
        batch, ctx, dims = B, L, D  # noqa: F841
        q = q * self.scale
        k = k * self.scale

        if is_causal:
            causal_mask = torch.empty(ctx, ctx, device=q.device).fill_(-np.inf).triu_(1)
            self.register_buffer("causal_mask", causal_mask, persistent=False)

            causal_mask = causal_mask.expand(batch, self.head, ctx, ctx)
            mask = causal_mask if mask is None else mask + causal_mask

        else:
            self.register_buffer("causal_mask", None, persistent=False)

        q = self._shape(q, ctx, batch)
        k = self._shape(k, k.size(1), batch)
        v = self._shape(v, v.size(1), batch)

        out = self.cos_attention(q, k, v, mask=mask)
        out = out.permute(0, 2, 1, 3).flatten(start_dim=2)
        return out

class MultiheadC(nn.Module):
    def __init__(self, dims: int, head: int):
        super().__init__()
        self.head = head
        self.dims = dims
        self.head_dim = dims // head
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dims, dims)
        self.k = nn.Linear(dims, dims, bias=False)
        self.v = nn.Linear(dims, dims)
        self.o = nn.Linear(dims, dims)
            
    def _shape(self, tensor: torch.Tensor, ctx: int, batch: int):
        return tensor.view(batch, ctx, self.head, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(self, x: Tensor, xa: Optional[Tensor], kv_cache = None, is_causal=True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, ctx = x.shape[:2]

        if kv_cache is None or xa is None or self.k not in kv_cache:
            k = self.k(x if xa is None else xa)
            v = self.v(x if xa is None else xa)
        else:
            k = kv_cache[self.k]
            v = kv_cache[self.v]

        q = self.q(x)
        wv = self._attention(q, k, v, is_causal=is_causal)
        return self.o(wv), None

    def _attention(self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool, attn_mask: Optional[torch.Tensor] = None, need_weights=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, ctx, dims = q.shape

        q = q * self.scale
        k = k * self.scale
        self.dims = dims

        q = self._shape(q, ctx, batch)
        k = self._shape(k, k.size(1), batch)
        v = self._shape(v, v.size(1), batch)
        # B, L, D = q.shape
        # print("q shape:", q.shape, "k shape:", k.shape, "v shape:", v.shape)

        if need_weights:

            qk = (q) @ (k).transpose(-1, -2)
            if attn_mask is not None:
                qk = qk + attn_mask[:ctx, :ctx]
            qk = qk.float()
            w = F.softmax(qk, dim=-1).to(q.dtype)
            output = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2).contiguous()
            qk = qk.detach()
            output = output.transpose(1, 2).flatten(-2).contiguous()   # (B, L, head * head_dim) -> (B, L, D)
            out = self.o(output)
            return out, qk

        else:
            qk=None
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
                a = SDPA(q, k, v, attn_mask=None, is_causal=have(attn_mask), enable_gqa=False) # sdpa folds weights into the attention computation
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2).contiguous()
            out = self.o(out)
            return out, qk


class SelfAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 nhead: int,
                 dropout: float = 0.0,
                 batch_first: bool = True,
                 add_pe_to_qkv: List[bool] = [True, True, False]):
        super().__init__()
# 
        self.self_attn = MultiheadC(dim, nhead)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.add_pe_to_qkv = add_pe_to_qkv

    def forward(self,
                x: torch.Tensor,
                pe: torch.Tensor,
                attn_mask: bool = None,
                key_padding_mask: bool = None) -> torch.Tensor:
        x = self.norm(x)
        if any(self.add_pe_to_qkv):
            x_with_pe = x + pe
            q = x_with_pe if self.add_pe_to_qkv[0] else x # 
            k = x_with_pe if self.add_pe_to_qkv[1] else x
            v = x_with_pe if self.add_pe_to_qkv[2] else x
        else:
            q = k = v = x
        r = x
        x, qk = self.self_attn._attention(q, k, v, is_causal=False, attn_mask=attn_mask, need_weights=False)
        return r + self.dropout(x)


# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
class CrossAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 nhead: int,
                 dropout: float = 0.0,
                 batch_first: bool = True,
                 add_pe_to_qkv: List[bool] = [True, True, False],
                 residual: bool = True,
                 norm: bool = True):
        super().__init__()
        self.cross_attn = MultiheadC(dim, nhead)
        # self.cross_attn = nn.MultiheadAttention(dim,
        #                                         nhead,
        #                                         dropout=dropout,
        #                                         batch_first=batch_first)
        if norm:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.add_pe_to_qkv = add_pe_to_qkv
        self.residual = residual

    def forward(self,
                x: torch.Tensor,
                mem: torch.Tensor,
                x_pe: torch.Tensor,
                mem_pe: torch.Tensor,
                attn_mask: bool = None,
                *,
                need_weights: bool = False) -> (torch.Tensor, torch.Tensor): # type: ignore
        x = self.norm(x)
        if self.add_pe_to_qkv[0]:
            q = x + x_pe
        else:
            q = x

        if any(self.add_pe_to_qkv[1:]):
            mem_with_pe = mem + mem_pe
            k = mem_with_pe if self.add_pe_to_qkv[1] else mem
            v = mem_with_pe if self.add_pe_to_qkv[2] else mem
        else:
            k = v = mem
        r = x

        # print("attn_mask:", attn_mask)
        # print("q:", q.shape, "k:", k.shape, "v:", v.shape, "x:", x.shape, "attn_mask:", attn_mask, "need_weights:", need_weights)
        x, weights = self.cross_attn._attention(q, k, v, is_causal=attn_mask, attn_mask=None, need_weights=False)

        if self.residual:
            return r + self.dropout(x), weights
        else:
            return self.dropout(x), weights


class FFN(nn.Module):
    def __init__(self, dim_in: int, dim_ff: int, activation=F.relu):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_ff)
        self.linear2 = nn.Linear(dim_ff, dim_in)
        self.norm = nn.LayerNorm(dim_in)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm(x)
        x = self.linear2(self.activation(self.linear1(x)))
        x = r + x
        return x


class PixelFFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.conv = CAResBlock(dim, dim)

    def forward(self, pixel: torch.Tensor, pixel_flat: torch.Tensor) -> torch.Tensor:
        # pixel: batch_size * num_objects * dim * H * W
        # pixel_flat: (batch_size*num_objects) * (H*W) * dim
        bs, num_objects, _, h, w = pixel.shape
        pixel_flat = pixel_flat.view(bs * num_objects, h, w, self.dim)
        pixel_flat = pixel_flat.permute(0, 3, 1, 2).contiguous()

        x = self.conv(pixel_flat)
        x = x.view(bs, num_objects, self.dim, h, w)
        return x


class OutputFFN(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation=F.relu):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_out)
        self.linear2 = nn.Linear(dim_out, dim_out)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.activation(self.linear1(x)))
        return x


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
