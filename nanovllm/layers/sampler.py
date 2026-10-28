import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))#先除以温度
        probs = torch.softmax(logits, dim=-1)#softmax  转化成概率
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        #根据，最终概率，按照概率分布生成一些随机数，也就是说不是贪心策略，有一定的随机选择，但随机选择依旧是基于概率分布的
        return sample_tokens
