import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


# def load_model(model: nn.Module, path: str):
#     packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
#     for file in glob(os.path.join(path, "*.safetensors")):#'/mnt/public/gw/LLM_model/Qwen3-4B-Instruct-2507/model-00003-of-00003.safetensors'
#         with safe_open(file, "pt", "cpu") as f:
#             for weight_name in f.keys():
#                 for k in packed_modules_mapping:
#                     if k in weight_name:
#                         v, shard_id = packed_modules_mapping[k]
#                         param_name = weight_name.replace(k, v)
#                         param = model.get_parameter(param_name)
#                         weight_loader = getattr(param, "weight_loader")
#                         weight_loader(param, f.get_tensor(weight_name), shard_id)
#                         break
#                 else:
#                     param = model.get_parameter(weight_name)
#                     weight_loader = getattr(param, "weight_loader", default_weight_loader)
#                     weight_loader(param, f.get_tensor(weight_name))

"""
print("A: before safe_open", flush=True)

with safe_open(file, "pt", "cpu") as f:
    print("B: safe_open completed", flush=True)

    keys = f.keys()
    print("C: keys count =", len(keys), flush=True)

    for weight_name in keys:
        print("D: loading", weight_name, flush=True)
        # 原来的权重加载逻辑
"""

def load_model(model: nn.Module, path: str):
    """
    读取 Safetensors 权重文件
          ↓
遍历每一个权重名称
          ↓
判断是否属于需要合并的权重
          ↓
    ┌─────┴─────────┐
    ↓               ↓
特殊权重          普通权重
Q/K/V、Gate/Up    RMSNorm、Embedding 等
    ↓               ↓
映射目标参数       找到同名参数
    ↓               ↓
调用专用 Loader   调用默认/专用 Loader
    └─────┬─────────┘
          ↓
将真实权重复制到模型参数空间
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    """
    因为 Hugging Face 模型权重和 Nano-vLLM 内部的模块组织方式不同。

        例如，原始 Qwen3 模型可能有三个独立权重：
        model.layers.0.self_attn.q_proj.weight
        model.layers.0.self_attn.k_proj.weight
        model.layers.0.self_attn.v_proj.weight


        但是 Nano-vLLM 使用： qkv_proj
    """
    for file in glob(os.path.join(path, "*.safetensors")):#'/mnt/public/gw/LLM_model/Qwen3-4B-Instruct-2507/model-00003-of-00003.safetensors'
        print("A: before safe_open", flush=True)

        with safe_open(file, "pt", "cpu") as f:
            print("B: safe_open completed", flush=True)

            keys = f.keys()
            print("C: keys count =", len(keys), flush=True)
            for weight_name in f.keys():#model.layers.35.input_layernorm.weight
                for k in packed_modules_mapping:#q_proj
                    """
                    q_proj' =
                            ('qkv_proj', 'q')
                            special variables
                            function variables
                            0 =
                            'qkv_proj'
                            1 =
                            'q'
                            len() =
                            2
                            'k_proj' =
                            ('qkv_proj', 'k')
                            special variables
                            function variables
                            0 =
                            'qkv_proj'
                            1 =
                            'k'
                            len() =
                            2
                            'v_proj' =
                            ('qkv_proj', 'v')
                            special variables
                            function variables
                            0 =
                            'qkv_proj'
                            1 =
                            'v'
                            len() =
                            2
                            'gate_proj' =
                            ('gate_up_proj', 0)
                            special variables
                            function variables
                            0 =
                            'gate_up_proj'
                            1 =
                            0
                            len() =
                            2
                            'up_proj' =
                            ('gate_up_proj', 1)
                            special variables
                            function variables
                            0 =
                            'gate_up_proj'
                            1 =
                            1
                            len() =
                            2
                    
                    """
                    if k in weight_name:#这里只是因为 huggingface 和 vllm 的一些命名方式问题，不是多gpu并行导致
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)#model.layers.35.input_layernorm.weight  2560 全 1 ，是因为之前的gpu初始化权重空间
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)#权重复制到gpu的权重空间中的 函数
                    weight_loader(param, f.get_tensor(weight_name))#把真实权重复制到 gpu分配到那部分权重空间中