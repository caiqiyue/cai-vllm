import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,# 2560
        num_heads: int,#32
        num_kv_heads: int,#8
        max_position: int = 4096 * 32,# 262144
        head_dim: int | None = None, # 128
        rms_norm_eps: float = 1e-06, #
        qkv_bias: bool = False, #
        rope_theta: float = 10000, #
        rope_scaling: dict | None = None, # {'rope_theta': 5000000, 'rope_type': 'default'}
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()#1 进程数,可以理解为就只有一个gpu
        self.total_num_heads = num_heads# 32
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size#32
        self.total_num_kv_heads = num_kv_heads # kv 
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size# 8
        self.head_dim = head_dim or hidden_size // self.total_num_heads#128
        self.q_size = self.num_heads * self.head_dim#q 4096
        self.kv_size = self.num_kv_heads * self.head_dim#1024
        self.scaling = self.head_dim ** -0.5 #0.0883
        self.qkv_bias = qkv_bias# false

        self.qkv_proj = QKVParallelLinear( # 创建并行的权重空间，qkv的映射权重
            hidden_size,#2560 2560是embedding的维度
            self.head_dim,# 128
            self.total_num_heads,#32
            self.total_num_kv_heads,#8
            bias=qkv_bias,#false
        )
        """
        假设激活值 是 128,2560 现在 qkv的映射矩阵是   32x128，2560
        那么乘法应该是 128,2560 x 2560,32x128 得到 128,32x128
        现在要把权重矩阵 32x128,2560 切分，
        如果是按列切分 那么就是 32x128，1280  和 32x128,1280 
        那怎么并行计算？ 128，2560 不能喝 32x128,1280 计算， 一般来说激活值不切分

        所以只有 按行切  128,2560 和 16x128,2560 计算 得到 128,16x128 
        最后合并两个 128,16x128 得到 128,32x128
        
        attention 出来之后，应该是 128,32x128  权重是 2560,32x128
        假设一个gpu，那出来就是 128,16x128 
        原始 o 权重是 2560,32x128，所以得按列切， 2560,16x128 
        最后 128，16x128 和 2560,16x128 相乘，得到 128,2560

        然后两个gpu 再通信，对  两个 128,2560 进行求和  
        """
        self.o_proj = RowParallelLinear(#attention 合并头之后的输出投影。 但你要注意 有的是 横着切，有的是竖着切，这你得懂矩阵乘法的尺寸变化
            self.total_num_heads * self.head_dim,#32 x 128
            hidden_size,#2560
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(#旋转位置矩阵
            self.head_dim,#128
            rotary_dim=self.head_dim,#128
            max_position=max_position,#262144
            base=rope_theta,
        )
        self.attn = Attention(#attention qkv计算，attention中没有权重，所有没有所谓的并行
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,#2560
            [intermediate_size] * 2, # 9728 * 2
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,#9728
            hidden_size,#2560
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size, #2560
            num_heads=config.num_attention_heads,#q: 32个头
            num_kv_heads=config.num_key_value_heads,#k v : 8  qwen3 系列就是 4个q 对 一个 k
            max_position=config.max_position_embeddings,#262144 
            rms_norm_eps=config.rms_norm_eps,#1 x 10(-6)
            qkv_bias=getattr(config, 'attention_bias', True),#false
            head_dim=getattr(config, 'head_dim', None),#128
            rope_theta=getattr(config, "rope_theta", 1000000),#1x10(6)
            rope_scaling=getattr(config, "rope_scaling", None),#{'rope_theta': 5000000, 'rope_type': 'default'}
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,#2560
            intermediate_size=config.intermediate_size,#9728
            hidden_act=config.hidden_act,#silu
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)#2560 1x10 -6
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)#输入时候rms norm
        hidden_states = self.self_attn(positions, hidden_states)#2，2560
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)#2,2560
        hidden_states = self.mlp(hidden_states)#2,2560
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)#151936个词，hidden stat 2560
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])#36层decode layer
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)#预热阶段 input_ids 是512 个 0，   512，2560    input_ids decode [8597,3730]  hid 2,2560
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)#hidden 512,2560  residual 512,2560  decode 2，2560  2代表2个样本
        hidden_states, _ = self.norm(hidden_states, residual)#512,2560
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)#现在 nano vllm 就支持这一个模型
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)#分类头，和前面的 embedding 一致，因为这里是 tied embedding
        if config.tie_word_embeddings:                
            self.lm_head.weight.data = self.model.embed_tokens.weight.data #使用分类头，也就是 分类头使用 embedding 那一层

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)#计算qwen3model返回的应该是特征值 返回 512,2560

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)#1，151936  decode阶段是 2，151936 每个样本一个词一个词的预测
