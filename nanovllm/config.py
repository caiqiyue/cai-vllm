import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self): # 这是一种什么写法？
        assert os.path.isdir(self.model)#判断模型是否存在
        assert self.kvcache_block_size % 256 == 0#判断kv cache 的存储长度必须是256 的倍数
        assert 1 <= self.tensor_parallel_size <= 8#并法数不能超过8
        self.hf_config = AutoConfig.from_pretrained(self.model) # 根据模型去 huggingface 中找出对应模型的具体结构
        """
        
                Qwen3Config {
                "architectures": [
                    "Qwen3ForCausalLM"
                ],
                "attention_bias": false,
                "attention_dropout": 0.0,
                "bos_token_id": 151643,
                "dtype": "bfloat16",
                "eos_token_id": 151645,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 2560,
                "initializer_range": 0.02,
                "intermediate_size": 9728,
                "layer_types": [
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention",
                    "full_attention"
                ],
                "max_position_embeddings": 262144,
                "max_window_layers": 36,
                "model_type": "qwen3",
                "num_attention_heads": 32,
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "pad_token_id": null,
                "rms_norm_eps": 1e-06,
                "rope_parameters": {
                    "rope_theta": 5000000,
                    "rope_type": "default"
                },
                "sliding_window": null,
                "tie_word_embeddings": true,
                "transformers_version": "5.20.2",
                "use_cache": true,
                "use_sliding_window": false,
                "vocab_size": 151936
                }
        
        """
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)#词表 是 262144
