from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        """
        @dataclass：自动生成 __init__() 等常用方法。



        slots=True：限制实例可设置的属性，减少实例的内存开销等。



        __post_init__()：自动生成的 __init__() 完成属性赋值后，自动调用这个方法。
        """