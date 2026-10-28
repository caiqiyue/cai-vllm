import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()# 0 
        self.tp_size = dist.get_world_size() # 1
        """
        tp_rank：当前 GPU 进程在进程组中的编号。

        tp_size：参与张量并行的进程数。
        """
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings#151936
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size#对embedding 进行切分
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank # 0
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition #151936
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))#创建embedding，每个词用2560维表示;torch.empty() 申请张量空间，但不初始化为有效的模型权重
        self.weight.weight_loader = self.weight_loader#embedding模型的权重加载

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        param_data：当前 GPU 上已经分配好的权重空间，但还没有填入真实权重。

        loaded_weight：从模型文件中读取的完整权重数据，尚未按照 GPU 切分。

        shard_size：告诉程序取多少行。



            start_idx：告诉程序从第几行开始取。



            narrow()：从完整权重中截取属于当前 GPU 的部分。



            copy_()：把截取出的真实权重填进之前分配的空间。
        """
        param_data = param.data#这是切分好的gpu词表
        shard_size = param_data.size(0)#每个gpu的词表负责多少个词
        start_idx = self.tp_rank * shard_size# gpu 的 rank 号 乘以 自己负责多少个词，作为 完整词表权重 中 截取的开始位置
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)#从第0维，也就是按行切，初始位置开始，截取多少个token
        param_data.copy_(loaded_weight)# 把切分好的 词表权重 复制到 param.data gpu 被分配的张量中

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()#获取上下文，在prefill阶段 装配好的上下文
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1#计算出q 的最后一个索引
            x = x[last_indices].contiguous()#1,2560 x一开始是512，2560 然后取最后一个token 变成了 1，2560
        logits = F.linear(x, self.weight)#512,2560 和 llmhead 151936，2560 相乘 得到，1，151936
        if self.tp_size > 1:# 这里关注 的是 多gpu 的时候，切分的权重 算出 部分 logits 最后要合并
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
