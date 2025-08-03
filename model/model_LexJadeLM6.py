# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             LexJadeLM6 Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
import math
from transformers import PretrainedConfig
class LexJadeLM6Config(PretrainedConfig):
    """
    LexJadeLM6 系列模型的配置类。
    遵循针对不同规模进行独立结构优化的原则。
    """
    model_type = "lexjadelm6"
    def __init__(
        self,
        # 基础模型配置
        vocab_size: int = 15500, # 默认为600M模型的词汇表大小
        hidden_size: int = 1024, # 默认为600M模型 (Ultra) 的隐藏层大小
        intermediate_size: int = None, # 通常由 hidden_size 和门控机制推导
        num_hidden_layers: int = 12, # 示例层数
        num_attention_heads: int = 12, # 默认为600M模型 (Ultra) 的头数
        num_key_value_heads: int = 4, # 使用GQA，键值头数通常少于查询头数
        hidden_act: str = "silu", # SLM常用激活函数
        max_position_embeddings: int = 4096, # 示例最大位置编码
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5, # RMSNorm epsilon
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pretraining_tp: int = 1,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        # Dropout
        dropout: float = 0.0,
        # 滑动窗口注意力 (适用于 Tiny/Flash/Cube/Large)
        sliding_window: int = None, # e.g., 1024 for Tiny/Flash/Cube/Large
        global_attn_every_n_layers: int = None, # e.g., 5 for Tiny/Flash/Cube/Large
        ####################################################
        # MoE 配置 (适用于 Extreme-M / Ultra-M)
        # 当 use_moe 为 False 时，以下配置无效
        ####################################################
        use_moe: bool = False,
        moe_layer_freq: int = 1, # MoE层出现的频率
        num_experts_per_tok: int = 4, # 每个token激活的专家数 K (来自文档: 6路由专家，每token激活4个)
        n_routed_experts: int = 6, # 路由专家总数 N (来自文档)
        n_shared_experts: int = 1, # 共享专家数量 (来自文档)
        expert_capacity_factor: float = 1.35, # 专家容量因子 (来自文档)
        moe_aux_loss_alpha: float = 0.05, # MoE辅助损失系数 (动态增长的最终值, 来自文档)
        moe_aux_loss_alpha_init: float = 0.01, # MoE辅助损失系数初始值 (来自文档)
        moe_scoring_func: str = 'softmax', # 专家评分函数
        moe_norm_topk_prob: bool = True, # 是否标准化top-k概率
        moe_seq_aux: bool = True, # 是否在序列级别计算辅助损失
        # MoGE 特定配置 (专家分组) - 文档建议 M = N / K, 但需为整数
        moe_num_expert_groups: int = None, # 专家组数 M, 例如 6专家4激活，可能分2组(每组3专家，每组激活2个)
        # 专家隐藏层大小 (通常比普通FFN大以补偿稀疏性) - 文档建议增加约15%
        moe_expert_hidden_size_factor: float = 1.15, # 专家层隐藏大小相对于标准FFN的倍数
        ####################################################
        # PEMA 微调优化 (配置信息通常在训练/微调脚本中处理)
        ####################################################
        use_pema: bool = False, # 微调时是否启用PEMA
        **kwargs,
    ):
        # --- 基础参数 ---
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        # 计算或设置 intermediate_size
        if intermediate_size is None:
            # 常见的门控FFN中间大小计算方式
            intermediate_size = int(8 * self.hidden_size / 3)
            # 向上取整到256的倍数，符合SLM实践
            self.intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        else:
            self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        # 检查 head_dim 是否整除
        if self.hidden_size % self.num_attention_heads != 0:
             raise ValueError(
                 f"`hidden_size` must be divisible by `num_attention_heads` (got `hidden_size`: {self.hidden_size}"
                 f" and `num_attention_heads`: {self.num_attention_heads})."
             )
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        # 检查 GQA 配置是否有效
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                f"`num_key_value_heads` ({self.num_key_value_heads}) cannot be larger than `num_attention_heads` "
                f"({self.num_attention_heads})."
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"`num_attention_heads` ({self.num_attention_heads}) must be divisible by "
                f"`num_key_value_heads` ({self.num_key_value_heads})."
            )
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        if hidden_act != "silu":
             print(f"警告: LexJade6 通常使用 'silu' 激活函数, 得到的是 '{hidden_act}'.")
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        # --- 注意力机制特定参数 ---
        self.sliding_window = sliding_window
        self.global_attn_every_n_layers = global_attn_every_n_layers
        # --- MoE 参数 ---
        self.use_moe = use_moe
        self.moe_layer_freq = moe_layer_freq
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.expert_capacity_factor = expert_capacity_factor
        self.moe_aux_loss_alpha = moe_aux_loss_alpha
        self.moe_aux_loss_alpha_init = moe_aux_loss_alpha_init
        self.moe_scoring_func = moe_scoring_func
        self.moe_norm_topk_prob = moe_norm_topk_prob
        self.moe_seq_aux = moe_seq_aux
        self.moe_num_expert_groups = moe_num_expert_groups
        self.moe_expert_hidden_size_factor = moe_expert_hidden_size_factor
        # --- 其他参数 ---
        self.use_pema = use_pema
        self.dropout = dropout
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                          LexJadeLM6 Model Variants Configurations
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
# --- LexJade6-Tiny (5M) ---
class LexJade6TinyConfig(LexJadeLM6Config):
    """LexJade6-Tiny (5M参数) 配置"""
    def __init__(self, **kwargs):
        super().__init__(
            vocab_size=10000,
            hidden_size=256,
            num_hidden_layers=4, # 示例层数，可根据需要调整
            num_attention_heads=4,
            num_key_value_heads=2, # GQA
            intermediate_size=512, # 示例，实际会由基类计算覆盖
            sliding_window=1024, # 启用滑动窗口
            global_attn_every_n_layers=5, # 滑动窗口配置
            use_moe=False,
            **kwargs,
        )
# --- LexJade6-Flash (25M) ---
class LexJade6FlashConfig(LexJadeLM6Config):
    """LexJade6-Flash (25M参数) 配置"""
    def __init__(self, **kwargs):
        super().__init__(
            vocab_size=10000,
            hidden_size=256,
            num_hidden_layers=8, # 示例层数
            num_attention_heads=4,
            num_key_value_heads=2, # GQA
            intermediate_size=512, # 示例
            sliding_window=1024, # 启用滑动窗口
            global_attn_every_n_layers=5, # 滑动窗口配置
            use_moe=False,
            **kwargs,
        )
# --- LexJade6-Cube (50M) ---
class LexJade6CubeConfig(LexJadeLM6Config):
    """LexJade6-Cube (50M参数) 配置"""
    def __init__(self, **kwargs):
        super().__init__(
            vocab_size=10000,
            hidden_size=512,
            num_hidden_layers=8, # 示例层数
            num_attention_heads=8,
            num_key_value_heads=4, # GQA
            intermediate_size=1024, # 示例
            sliding_window=1024, # 启用滑动窗口
            global_attn_every_n_layers=5, # 滑动窗口配置
            use_moe=False,
            **kwargs,
        )
# --- LexJade6-Large (100M) ---
class LexJade6LargeConfig(LexJadeLM6Config):
    """LexJade6-Large (100M参数) 配置"""
    def __init__(self, **kwargs):
        super().__init__(
            vocab_size=10000,
            hidden_size=512,
            num_hidden_layers=12, # 示例层数
            num_attention_heads=8,
            num_key_value_heads=4, # GQA
            intermediate_size=1024, # 示例
            sliding_window=1024, # 启用滑动窗口
            global_attn_every_n_layers=5, # 滑动窗口配置
            use_moe=False,
            **kwargs,
        )
# --- LexJade6-Extreme (300M) ---
class LexJade6ExtremeConfig(LexJadeLM6Config):
    """LexJade6-Extreme (300M参数) 配置"""
    def __init__(self, **kwargs):
        super().__init__(
            vocab_size=11400,
            hidden_size=768,
            num_hidden_layers=16, # 示例层数
            num_attention_heads=8,
            num_key_value_heads=4, # GQA
            intermediate_size=2048, # 示例
            use_moe=False, # 密集模型
            **kwargs,
        )
# --- LexJade6-Ultra (600M) ---
class LexJade6UltraConfig(LexJadeLM6Config):
    """LexJade6-Ultra (600M参数) 配置"""
    def __init__(self, **kwargs):
        super().__init__(
            vocab_size=15500,
            hidden_size=1024,
            num_hidden_layers=16, # 示例层数
            num_attention_heads=12,
            num_key_value_heads=4, # GQA
            intermediate_size=2816, # 示例 (参考Qwen2结构)
            use_moe=False, # 密集模型
            **kwargs,
        )
# --- LexJade6-Extreme-M (300M-MoE) ---
class LexJade6ExtremeMConfig(LexJadeLM6Config):
    """LexJade6-Extreme-M (300M-MoE参数) 配置"""
    def __init__(self, **kwargs):
        super().__init__(
            vocab_size=11400,
            hidden_size=768,
            num_hidden_layers=16, # 示例层数
            num_attention_heads=8,
            num_key_value_heads=4, # GQA
            intermediate_size=2048, # 基础FFN中间大小示例
            use_moe=True, # 启用MoE
            moe_layer_freq=1, # 示例：每层都是MoE
            num_experts_per_tok=4, # 文档配置
            n_routed_experts=6, # 文档配置
            n_shared_experts=1, # 文档配置
            expert_capacity_factor=1.35, # 文档配置
            moe_aux_loss_alpha=0.05, # 文档配置
            moe_aux_loss_alpha_init=0.01, # 文档配置
            moe_num_expert_groups=2, # 示例分组 (6专家分2组)
            moe_expert_hidden_size_factor=1.15, # 文档配置
            **kwargs,
        )
# --- LexJade6-Ultra-M (600M-MoE) ---
class LexJade6UltraMConfig(LexJadeLM6Config):
    """LexJade6-Ultra-M (600M-MoE参数) 配置"""
    def __init__(self, **kwargs):
        super().__init__(
            vocab_size=15500,
            hidden_size=1024,
            num_hidden_layers=16, # 示例层数
            num_attention_heads=12,
            num_key_value_heads=4, # GQA
            intermediate_size=2816, # 基础FFN中间大小示例
            use_moe=True, # 启用MoE
            moe_layer_freq=1, # 示例：每层都是MoE
            num_experts_per_tok=4, # 文档配置
            n_routed_experts=6, # 文档配置
            n_shared_experts=1, # 文档配置
            expert_capacity_factor=1.35, # 文档配置
            moe_aux_loss_alpha=0.05, # 文档配置
            moe_aux_loss_alpha_init=0.01, # 文档配置
            moe_num_expert_groups=2, # 示例分组 (6专家分2组)
            moe_expert_hidden_size_factor=1.15, # 文档配置
            **kwargs,
        )
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             LexJadeLM6 Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
import math
import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
# --- 修复点 1: 导入 BaseModelOutputWithPast ---
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
# --- 修复点 1 结束 ---
class RMSNorm(torch.nn.Module):
    """RMSNorm 归一化层，计算更轻量"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        # 计算均方根并取倒数
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        # 保持输入类型一致
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """
    预计算 RoPE (Rotary Position Embedding) 的频率和复数表示。
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # cos和sin用于旋转嵌入
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2): # --- 修复点 2: 默认参数改为 2 ---
    """
    应用 RoPE 到查询(Q)和键(K)向量。
    Args:
        q (torch.Tensor): 查询向量, 形状 [..., seq_len, num_heads, head_dim]
        k (torch.Tensor): 键向量, 形状 [..., seq_len, num_heads, head_dim]
        cos (torch.Tensor): 预计算的余弦值, 形状 [..., seq_len, head_dim]
        sin (torch.Tensor): 预计算的正弦值, 形状 [..., seq_len, head_dim]
        position_ids (torch.Tensor, optional): 位置ID (未在此函数中使用, 保留以兼容性).
        unsqueeze_dim (int): 在哪个维度上对 cos/sin 增加维度以进行广播。对于 [B, S, H, D] 输入，应为 2。
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 旋转后的位置嵌入 q_embed 和 k_embed。
    """
    def rotate_half(x):
        # 将向量的后半部分取负并交换前后两半
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    # 应用旋转
    # q/k shape: [..., seq_len, num_heads, head_dim]
    # cos/sin shape: [..., seq_len, head_dim]
    # cos.unsqueeze(unsqueeze_dim) shape: [..., seq_len, 1, head_dim] (if unsqueeze_dim=2)
    # 广播: [..., seq_len, num_heads, head_dim] * [..., seq_len, 1, head_dim] -> OK
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    复制键值头以匹配查询头的数量 (用于GQA)。
    与 MiniMind 实现完全一致。
    期望输入 x 的形状为 [batch_size, seq_len, num_key_value_heads, head_dim]。
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 使用 repeat_interleave 在 head 维度上复制
    # 这与 MiniMind 的原始实现完全一致
    return torch.repeat_interleave(x, dim=2, repeats=n_rep)
class LexJadeAttention(nn.Module):
    """
    LexJade 注意力层，支持 GQA 和滑动窗口注意力。
    """
    def __init__(self, config: LexJadeLM6Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.attention_dropout = config.attention_dropout
        # 滑动窗口配置
        self.sliding_window = config.sliding_window if hasattr(config, 'sliding_window') else None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor], # cos, sin
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        # --- 修复点 1: 确保在使用缓存时只处理最新 token ---
        # 在生成模式下，每次只应处理一个 token
        if past_key_value is not None and past_key_value[0] is not None and q_len > 1:
            # 只保留最后一个 token (最新 token)
            hidden_states = hidden_states[:, -1:, :]
            _, q_len, _ = hidden_states.size()  # 更新 q_len 为 1
        # --- 修复点 1 结束 ---
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # --- 1. 重塑为 [B, S, Num_Heads/Num_KV_Heads, Head_Dim] ---
        # 这是与 MiniMind 保持一致的关键格式
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
        # --- 2. 应用 RoPE ---
        cos, sin = position_embeddings
        # 确保 RoPE 的 seq_len 与当前 q_len 匹配
        if cos.dim() == 2: # [seq_len, dim]
            cos = cos[:q_len]
            sin = sin[:q_len]
        elif cos.dim() == 3: # [bsz, seq_len, dim] or [1, seq_len, dim]
            cos = cos[:, :q_len, :]
            sin = sin[:, :q_len, :]
        
        # --- 修复点 3: 修改 unsqueeze_dim 参数 ---
        # 应用 RoPE
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2) # --- 修改 ---
        # --- 修复点 3 结束 ---
        
        # --- 3. KV Cache 处理 ---
        if past_key_value is not None:
            # past_key_value 包含 [past_key, past_value]
            past_key, past_value = past_key_value
            # 在序列维度 (dim=1) 上拼接
            key_states = torch.cat([past_key, key_states], dim=1)
            value_states = torch.cat([past_value, value_states], dim=1)
        
        # 保存新的 KV 缓存
        past_key_value = (key_states, value_states) if use_cache else None
        
        # --- 4. GQA: 重复 K 和 V 以匹配 Q 的头数 ---
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # --- 5. 转置为 [B, Num_Heads, S, Head_Dim] 以进行注意力计算 ---
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # --- 6. 标准缩放点积注意力 (SDPA) ---
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # --- 7. 滑动窗口注意力掩码处理 ---
        if self.sliding_window is not None and q_len > 1:
            current_len = key_states.shape[2]
            query_indices = torch.arange(q_len, device=attn_weights.device)
            key_indices = torch.arange(current_len, device=attn_weights.device)
            relative_positions = key_indices.unsqueeze(0) - query_indices.unsqueeze(1)
            sw_mask = (relative_positions < -self.sliding_window + 1) | (relative_positions > 0)
            sw_mask = sw_mask.unsqueeze(0).unsqueeze(0).expand(bsz, self.num_heads, q_len, current_len)
            attn_weights = attn_weights.masked_fill(sw_mask, torch.finfo(attn_weights.dtype).min)
        
        # --- 8. 应用传入的通用注意力掩码 ---
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=attn_weights.dtype)
            attn_weights = attn_weights + attention_mask
        
        # --- 9. softmax 归一化并计算输出 ---
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # --- 10. 转置并重塑 ---
        attn_output = attn_output.transpose(1, 2).contiguous()
        # --- 修复点 12: 使用正确的维度进行重塑 ---
        # 确保使用 num_heads * head_dim 而不是 hidden_size
        # 因为在某些配置中 (如 Tiny/Flash/Cube)，num_heads * head_dim != hidden_size
        # 但在 LexJade6-Flash 中，num_heads * head_dim = hidden_size = 256
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        # --- 修复点 12 结束 ---
        attn_output = self.o_proj(attn_output) # 线性投影
        
        return attn_output, past_key_value
class LexJadeMLP(nn.Module):
    """
    LexJade 前馈网络 (FFN)，采用门控结构 (Gated FFN) 和 SiLU 激活函数。
    """
    def __init__(self, config: LexJadeLM6Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # 门控投影
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        # 上投影
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        # 下投影
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        # 激活函数
        self.act_fn = ACT2FN[config.hidden_act]
    def forward(self, x):
        # Gated FFN: down_proj(act(gate_proj(x)) * up_proj(x))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
class MoEGate(nn.Module):
    """
    MoE 路由门控网络。
    """
    def __init__(self, config: LexJadeLM6Config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.scoring_func = config.moe_scoring_func
        self.alpha = config.moe_aux_loss_alpha # 最终的alpha值
        self.seq_aux = config.moe_seq_aux
        self.norm_topk_prob = config.moe_norm_topk_prob
        self.gating_dim = config.hidden_size
        # 门控权重矩阵
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()
    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h) # [B*S, H]
        logits = F.linear(hidden_states, self.weight, None) # [B*S, N]
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1) # [B*S, N]
        else:
            raise NotImplementedError(f'不支持的MoE门控评分函数: {self.scoring_func}')
        # 选择 Top-K 专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False) # [B*S, K]
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator # 标准化
        aux_loss = 0.0
        # 计算辅助负载均衡损失
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1) # [B, S*K]
            if self.seq_aux:
                # 序列级别辅助损失
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1) # [B, S, N]
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # Token级别辅助损失
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        return topk_idx, topk_weight, aux_loss
class MOEFeedForward(nn.Module):
    """
    MoE 前馈网络，包含路由专家和共享专家。
    """
    def __init__(self, config: LexJadeLM6Config):
        super().__init__()
        self.config = config
        # 计算专家网络的中间大小 (补偿性扩展)
        base_intermediate_size = config.intermediate_size
        expert_intermediate_size = int(base_intermediate_size * config.moe_expert_hidden_size_factor)
        # 确保是256的倍数
        expert_intermediate_size = 256 * ((expert_intermediate_size + 256 - 1) // 256)
        # 创建路由专家 (使用扩展后的中间大小)
        expert_config = LexJadeLM6Config(**config.to_dict())
        expert_config.intermediate_size = expert_intermediate_size
        self.experts = nn.ModuleList([
            LexJadeMLP(expert_config) # 使用 LexJadeMLP 作为专家
            for _ in range(config.n_routed_experts)
        ])
        # 创建门控网络
        self.gate = MoEGate(config)
        # 创建共享专家 (使用基础中间大小)
        if config.n_shared_experts > 0:
            shared_expert_config = LexJadeLM6Config(**config.to_dict())
            # 共享专家通常使用基础大小，但也可以调整
            self.shared_experts = nn.ModuleList([
                LexJadeMLP(shared_expert_config)
                for _ in range(config.n_shared_experts)
            ])
    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1]) # [B*S, H]
        flat_topk_idx = topk_idx.view(-1) # [B*S*K]
        if self.training:
            # 训练模式：复制token并分配给专家
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0) # [B*S*K, H]
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                # 将对应token分配给专家i
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
            # 加权求和
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1) # [B*S, H]
            y = y.view(*orig_shape) # [B, S, H]
        else:
            # 推理模式：高效路由 (参考MiniMind)
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        # 添加共享专家的输出
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        高效的MoE推理路由。
        """
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)
        return expert_cache
class LexJadeDecoderLayer(nn.Module):
    """
    LexJade 解码器层，包含自注意力和前馈网络。
    """
    def __init__(self, config: LexJadeLM6Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LexJadeAttention(config=config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 根据配置决定是否使用MoE层
        use_moe_layer = (
            config.use_moe and
            (layer_idx % config.moe_layer_freq == 0) # 例如，如果 freq=1，每层都是MoE
        )
        if use_moe_layer:
            self.mlp = MOEFeedForward(config)
        else:
            self.mlp = LexJadeMLP(config)
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # 自注意力
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        # 前馈网络 (FFN 或 MoE)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs # hidden_states, present_key_value
class LexJadeModel(PreTrainedModel):
    """
    LexJade 主干模型，包含嵌入层、解码器层堆叠和最终归一化层。
    """
    config_class = LexJadeLM6Config
    def __init__(self, config: LexJadeLM6Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LexJadeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # 初始化权重
        self.post_init()
        # RoPE 频率预计算
        self.rope_theta = config.rope_theta
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim, # 注意是 head_dim
            end=config.max_position_embeddings,
            theta=self.rope_theta
        )
        # 注册为 buffer，不参与梯度更新
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
    def get_input_embeddings(self):
        return self.embed_tokens
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]: # --- 修复点 4: 修改返回类型注解 ---
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("不能同时指定 input_ids 和 inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("必须指定 input_ids 或 inputs_embeds")
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # 位置ID处理
        if position_ids is None:
             device = input_ids.device if input_ids is not None else inputs_embeds.device
             # 计算当前序列的起始位置 (考虑历史缓存)
             # past_key_values[0] 是第一层的past_kv, past_key_values[0][0] 是key缓存
             # key缓存的形状是 [B, S_past, Num_KV_Heads, Head_Dim] (根据Attention的实现)
             past_seq_len = past_key_values[0][0].shape[1] if past_key_values[0] is not None and len(past_key_values[0]) > 0 else 0
             position_ids = torch.arange(
                 past_seq_len,
                 seq_length + past_seq_len,
                 dtype=torch.long,
                 device=device
             )
             position_ids = position_ids.unsqueeze(0) # [1, seq_len]
        # 获取 RoPE 位置嵌入 (根据 position_ids 切片)
        # freqs_cos/freqs_sin shape: [max_seq_len, dim]
        # position_ids shape: [batch_size, seq_len]
        # 我们需要 gather 正确的 cos/sin 值
        cos = self.freqs_cos[position_ids] # [batch_size, seq_len, dim]
        sin = self.freqs_sin[position_ids] # [batch_size, seq_len, dim]
        # apply_rotary_pos_emb 期望 cos/sin 是 [*, seq_len, dim]，这已经满足
        position_embeddings = (cos, sin) # [batch_size, seq_len, dim]
        # --- 修复点 5: 正确处理 attention_mask，兼容 past_key_values ---
        # 准备注意力掩码 (处理因果性和padding)
        # attention_mask (来自 tokenizer) 形状通常是 [B, S_input]
        # 我们需要将其处理成适用于当前前向传播的掩码
        # 如果 use_cache=True (生成模式)，S_total = past_seq_len + S_input
        # 我们需要一个形状为 [B, 1, S_input, S_total] 的掩码
        if attention_mask is not None:
            # 1. 确保 attention_mask 是正确的数据类型
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype) # [B, S_input]
            
            # 2. 获取当前输入序列长度 - 修复：从 input_ids/inputs_embeds 获取，而不是 attention_mask
            # attention_mask 的长度可能包含历史信息，而我们只需要当前输入的长度
            if input_ids is not None:
                batch_size, input_seq_len = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, input_seq_len = inputs_embeds.shape[:2]
            else:
                raise ValueError("必须指定 input_ids 或 inputs_embeds")

            # 3. 计算总的 key/value 序列长度 (包括历史缓存)
            # past_seq_len 是从 past_key_values 中获取的
            # past_key_values[0] 是第一层的缓存, [0] 是 key 缓存
            # key 缓存形状: [B, S_past, Num_KV_Heads, Head_Dim]
            past_seq_len = 0
            if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None and len(past_key_values[0]) > 0:
                past_seq_len = past_key_values[0][0].shape[1] # [B, S_past, ...] -> S_past
            
            # 在生成模式下，input_seq_len 应为 1
            if past_seq_len > 0 and input_seq_len > 1:
                # 修正：在生成模式下，只应处理最新 token
                input_seq_len = 1
            
            total_seq_len = past_seq_len + input_seq_len # S_total
            
            # 4. 创建适用于当前输入的因果掩码 [S_input, S_input]
            # 这只考虑当前输入内部的因果关系
            current_causal_mask = torch.tril(torch.ones((input_seq_len, input_seq_len), dtype=inputs_embeds.dtype, device=inputs_embeds.device))
            
            # 5. 如果有历史缓存，需要允许当前输入关注所有历史token
            # 创建一个块 [S_input, S_past] 填充为1 (允许关注)
            if past_seq_len > 0:
                historical_ones = torch.ones((input_seq_len, past_seq_len), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                # 将历史部分和当前因果部分拼接
                # 最终的未反转掩码形状 [S_input, S_total]
                combined_uninverted_mask = torch.cat([historical_ones, current_causal_mask], dim=1) # [S_input, S_total]
            else:
                # 如果没有历史，只使用当前因果掩码
                combined_uninverted_mask = current_causal_mask # [S_input, S_total] where S_total == S_input
            
            # 6. 将 combined_uninverted_mask 扩展到 [B, S_input, S_total]
            expanded_uninverted_mask = combined_uninverted_mask.unsqueeze(0).expand(batch_size, -1, -1) # [B, S_input, S_total]
            
            # 7. 应用原始的 padding mask
            # attention_mask: [B, S_input] (1 for valid, 0 for pad)
            # expanded_uninverted_mask: [B, S_input, S_total]
            # 我们需要将 padding mask 应用于 expanded_uninverted_mask 的第二个维度 (S_input)
            # expanded_padding_mask: [B, S_input, 1]
            expanded_padding_mask = attention_mask
            if past_seq_len > 0 and attention_mask.shape[1] > input_seq_len:
                # 在生成模式下，只取最后一个 token 的 padding mask
                expanded_padding_mask = attention_mask[:, -input_seq_len:]
            expanded_padding_mask = expanded_padding_mask.unsqueeze(-1) # [B, S_input, 1]
            
            # 广播相乘: [B, S_input, 1] * [B, S_input, S_total] -> [B, S_input, S_total]
            combined_mask = expanded_padding_mask * expanded_uninverted_mask # [B, S_input, S_total]
            
            # 8. 转换为 inverted 形式 (0 for valid, min for masked)
            inverted_mask = 1.0 - combined_mask
            inverted_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool),
                torch.finfo(inputs_embeds.dtype).min
            )
            
            # 9. 扩展维度以匹配注意力权重的期望输入 [B, 1, S_input, S_total]
            attention_mask = inverted_mask.unsqueeze(1) # [B, 1, S_input, S_total]
        # --- 修复点 5 结束 ---
        hidden_states = inputs_embeds
        # 解码器层循环
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        # 添加最后一层的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        # --- 修复点 6: 聚合并存储 aux_loss ---
        # 聚合所有 MoE 层的辅助损失
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if hasattr(layer.mlp, 'aux_loss') and layer.mlp.aux_loss is not None # 检查是否是MoE层且有损失
        )
        # 将 aux_loss 存储为模型的一个属性，供 LexJadeForCausalLM 访问
        # 注意：这不是线程安全的，但对于单GPU训练是可行的
        self._aux_loss = aux_loss
        # --- 修复点 6 结束 ---
        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)
        # --- 修复点 7: 返回 BaseModelOutputWithPast ---
        # LexJadeModel 应该返回 BaseModelOutputWithPast，
        # 因为它输出的是最后一层的隐藏状态 (norm 之后)，
        # 而不是最终的 logits。
        # logits 的计算是在 LexJadeForCausalLM 中完成的。
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, # <-- 这是正确的字段名
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            # aux_loss 不是 BaseModelOutputWithPast 的标准字段
        )
        # --- 修复点 7 结束 ---
class LexJadeForCausalLM(PreTrainedModel, GenerationMixin):
    """
    LexJade 因果语言模型，包装主干模型并添加语言模型头。
    """
    config_class = LexJadeLM6Config
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config: LexJadeLM6Config):
        super().__init__(config)
        self.model = LexJadeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 绑定权重：LM头的权重 = 嵌入层的权重
        self.lm_head.weight = self.model.embed_tokens.weight
        # 初始化权重
        self.post_init()
    def get_input_embeddings(self):
        return self.model.embed_tokens
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    def get_output_embeddings(self):
        return self.lm_head
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    def set_decoder(self, decoder):
        self.model = decoder
    def get_decoder(self):
        return self.model
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # --- 修复点 8: 调用主干模型 ---
        # 调用主干模型
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        # hidden_states = outputs[0] # 如果 outputs 是元组
        # 如果 outputs 是 BaseModelOutputWithPast, hidden_states 是 .last_hidden_state
        # --- 修复点 8 结束 ---
        # --- 修复点 9: 从主干模型获取隐藏状态 ---
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        # --- 修复点 9 结束 ---
        logits = self.lm_head(hidden_states)
        logits = logits.float() # 转为 float32 以提高数值稳定性
        loss = None
        if labels is not None:
            # 计算因果语言建模损失
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            # --- 修复点 10: 从主干模型获取并添加 MoE 辅助损失 ---
            # 添加 MoE 辅助损失 (如果存在)
            # outputs.aux_loss 可能不存在 (因为 BaseModelOutputWithPast 没有)
            # 我们从 self.model 实例中获取
            model_aux_loss = getattr(self.model, '_aux_loss', None) # <-- 修改：从 self.model 获取
            if model_aux_loss is not None and isinstance(model_aux_loss, torch.Tensor) and model_aux_loss.requires_grad:
                 loss = loss + model_aux_loss
            # --- 修复点 10 结束 ---
        if not return_dict:
            output = (logits,) + outputs[1:] # (logits, past_key_values, hidden_states, attentions)
            # 如果需要，可以手动添加 aux_loss 到 output 元组中，但这不标准
            # output = output + (model_aux_loss,) # 不推荐
            return (loss,) + output if loss is not None else output
        # --- 修复点 11: 返回标准的 CausalLMOutputWithPast ---
        # 确保只传递 CausalLMOutputWithPast 接受的参数
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # aux_loss 不是标准字段，如果需要，考虑自定义输出类或在 loss 中包含
        )
        # --- 修复点 11 结束 ---
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        为生成过程准备输入。
        """
        # 处理已由 past_key_values 覆盖的token
        if past_key_values is not None:
            # past_key_values is a tuple of tuples/lists: (layer_0_past, layer_1_past, ...)
            # layer_i_past is (key_cache, value_cache), each tensor is [B, S_past, H_kv, D]
            past_length = past_key_values[0][0].shape[1] if past_key_values[0] is not None and len(past_key_values[0]) > 0 else 0
            # 如果 attention_mask 比 input_ids 长，说明部分输入是通过 embeds 传递的
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 如果 past_length 小于 input_ids 长度，说明 input_ids 包含新token
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # 根据 attention_mask 动态创建 position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        # 如果传入了 inputs_embeds 且没有 past_key_values，使用 embeds
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        为 beam search 重新排序缓存。
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past