import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator
    """
    假设：
            input_size = 4096output_size = 8192tp_size = 2

            那么：
            divide(8192, 2)  # 4096
            实际传给父类的是：
            super().__init__(4096, 4096, False, 0)

            二、父类做了什么？
            回顾 LinearBase：
            class LinearBase(nn.Module):    def __init__(self, input_size, output_size, bias=False, tp_dim=None):        super().__init__()        self.tp_dim = tp_dim        self.tp_rank = dist.get_rank()        self.tp_size = dist.get_world_size()        self.weight = nn.Parameter(            torch.empty(output_size, input_size)        )

            因此，每个 GPU 对应的进程都会执行：
            self.weight = nn.Parameter(torch.empty(4096, 4096))

            这时：
                完整模型	GPU 0	GPU 1
            权重 Shape	[8192,4096]	[4096,4096]	[4096,4096]
            是否已有真实权重	模型文件中有	否	否
            到这里，只是创建了两个局部权重张量，还没有把真实权重切分进去。
            三、真正切分权重发生在哪里？
            在你之前分析的 weight_loader()：    
    """


class LinearBase(nn.Module):
    """
    初始化权重：创建形状为 [output_size, input_size] 的权重矩阵。

    保存 TP 信息：记录当前进程的 rank、world_size 和权重切分维度 tp_dim。

    定义公共接口：为子类提供权重加载接口和 Forward 接口。

    tp_dim 指示沿哪个维度进行张量并行切分。
        tp_dim	切分方向	含义
        0	第 0 维（行）	按输出特征切分，常见于 Column Parallel Linear
        1	第 1 维（列）	按输入特征切分，常见于 Row Parallel Linear
        None	不切分	每个进程保留完整权重
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim#0 按照哪一个维度 去 切分
        self.tp_rank = dist.get_rank()#0
        self.tp_size = dist.get_world_size()#1
        self.weight = nn.Parameter(torch.empty(output_size, input_size))#创建一个局部权重空间，每个gpu被分到的部分权重
        self.weight.weight_loader = self.weight_loader#这里的 weight_loader 让子类自己实现
        if bias:  #是否➕偏置矩阵
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        raise NotImplementedError 
        #让子类自己实现


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()#1
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)#这里就开始准备切分了：super().__init__(4096, 4096, False, 0) 会教给父类创建一个局部权重空间，但是没有加载真实权重

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data#每个gpu被分配到权重表，param.data 不是已经装好真实权重的矩阵，而是提前分配好的局部权重空间
        shard_size = param_data.size(self.tp_dim)# tp_dim 是切分的维度方向，从而获取到每个gpu分配到这个权重多少
        start_idx = self.tp_rank * shard_size#根据自己的进程号 和 分配到多少量 算出 切分的起始位置
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)#去真实的权重表中 找到 那一批真实矩阵值
        param_data.copy_(loaded_weight)#把那一批真实举证值 复制到 gpu中创建好的权重表中，即param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
                    假设完整 Linear：
            input_size = 4096output_size = 8192tp_size = 2



            那么：
                GPU 0	GPU 1
            输入 X	[1,4096]	[1,4096]
            局部权重 W	[4096,4096]	[4096,4096]
            输出 Y	[1,4096]	[1,4096]
            每张 GPU 分别执行：
            \[
            Y_0=XW_0^T
            \]

            \[
            Y_1=XW_1^T
            \]

            如果把两个结果沿最后一维拼接：
            \[
            Y=\operatorname{concat}(Y_0,Y_1)
            \]

            就得到完整的 [1,8192] 输出。
        """
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)#

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()#1
        total_num_kv_heads = total_num_kv_heads or total_num_heads#8
        self.head_size = head_size#128
        self.num_heads = divide(total_num_heads, tp_size)#32
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)#8
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size #6144 (32 + 2 * 8)*128
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()#1
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)#按行切

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
