import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False

class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int = None,
        # MLA Specifics
        d_latent_kv: int = 512, 
        d_latent_q: int = None, 
        d_rope: int = 64,    
        bias: bool = False,
        dropout: float = 0.0,
        layer_idx: int = None,  
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head if d_head is not None else d_model // n_heads
        self.d_latent_kv = d_latent_kv
        self.d_rope = d_rope
        self.layer_idx = layer_idx
        
        self.d_latent_q = d_latent_q if d_latent_q is not None else d_model
        
        self.q_down = nn.Linear(d_model, self.d_latent_q, bias=bias)
        self.q_up_content = nn.Linear(self.d_latent_q, n_heads * self.d_head, bias=bias)
        self.q_up_rope = nn.Linear(self.d_latent_q, n_heads * d_rope, bias=bias)


        self.kv_down = nn.Linear(d_model, d_latent_kv, bias=bias)
        self.k_up_content = nn.Linear(d_latent_kv, n_heads * self.d_head, bias=bias)
        self.k_up_rope = nn.Linear(d_latent_kv, n_heads * d_rope, bias=bias)
        self.v_up = nn.Linear(d_latent_kv, n_heads * self.d_head, bias=bias)

        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, inference_params=None, **kwargs):
        batch, seq_len, _ = x.shape

        c_q = self.q_down(x)
        q_content = self.q_up_content(c_q) 
        q_rope = self.q_up_rope(c_q)     
        
        c_kv = self.kv_down(x) 
        
        k_content = self.k_up_content(c_kv)
        k_rope = self.k_up_rope(c_kv)
        v = self.v_up(c_kv)

        q_content = rearrange(q_content, 'b l (h d) -> b l h d', h=self.n_heads)
        q_rope = rearrange(q_rope, 'b l (h d) -> b l h d', h=self.n_heads)
        k_content = rearrange(k_content, 'b l (h d) -> b l h d', h=self.n_heads)
        k_rope = rearrange(k_rope, 'b l (h d) -> b l h d', h=self.n_heads)
        v = rearrange(v, 'b l (h d) -> b l h d', h=self.n_heads)

        q_rope, k_rope = self._apply_rope(q_rope, k_rope)
        q = torch.cat([q_content, q_rope], dim=-1) # (B, L, H, D+R)
        k = torch.cat([k_content, k_rope], dim=-1) # (B, L, H, D+R)
        

        if FLASH_AVAILABLE:
            out = flash_attn_func(q, k, v, causal=True)
        else:
            scale = 1.0 / math.sqrt(q.size(-1))
            scores = torch.einsum('bthd,bshd->bhts', q, k) * scale
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores.masked_fill_(mask[None, None, :, :], float('-inf'))
            attn = F.softmax(scores, dim=-1)
            out = torch.einsum('bhts,bshd->bthd', attn, v)

        out = rearrange(out, 'b l h d -> b l (h d)')
        return self.out_proj(out)

    def step(self, x, state, **kwargs):
        batch, _, _ = x.shape
        
        c_q = self.q_down(x) 
        q_content = self.q_up_content(c_q)
        q_rope = self.q_up_rope(c_q)
        
        c_kv_t = self.kv_down(x) 
        k_rope_t = self.k_up_rope(c_kv_t) 
        
       
        q_content = rearrange(q_content, 'b l (h d) -> b l h d', h=self.n_heads)
        q_rope = rearrange(q_rope, 'b l (h d) -> b l h d', h=self.n_heads)
        k_rope_t = rearrange(k_rope_t, 'b l (h d) -> b l h d', h=self.n_heads)
        
        q_rope, k_rope_t = self._apply_rope(q_rope, k_rope_t, seq_idx=state[0].shape[1])

       
        if state is None:
            cached_c_kv = c_kv_t
            cached_k_rope = k_rope_t
        else:
            cached_c_kv, cached_k_rope = state
            cached_c_kv = torch.cat([cached_c_kv, c_kv_t], dim=1)
            cached_k_rope = torch.cat([cached_k_rope, k_rope_t], dim=1)
            
        new_state = (cached_c_kv, cached_k_rope)
        
        k_content_history = self.k_up_content(cached_c_kv) 
        v_history = self.v_up(cached_c_kv)                 
        
        k_content_history = rearrange(k_content_history, 'b l (h d) -> b l h d', h=self.n_heads)
        v_history = rearrange(v_history, 'b l (h d) -> b l h d', h=self.n_heads)
        
        k_history = torch.cat([k_content_history, cached_k_rope], dim=-1)
        q = torch.cat([q_content, q_rope], dim=-1)

       
        if FLASH_AVAILABLE:
            out = flash_attn_func(q, k_history, v_history, causal=False)
        else:
            scale = 1.0 / math.sqrt(q.size(-1))
            scores = torch.einsum('bthd,bshd->bhts', q, k_history) * scale
            attn = F.softmax(scores, dim=-1)
            out = torch.einsum('bhts,bshd->bthd', attn, v_history)
            
        out = rearrange(out, 'b l h d -> b l (h d)')
        return self.out_proj(out), new_state

    def _apply_rope(self, q_rope, k_rope, seq_idx=None):
        # Placeholder for RoPE rotation logic.
        # In MAD-lab, you likely import this from `..utils.rope` or similar.
        # For now, this is a pass-through to ensure shape compatibility.
        return q_rope, k_rope