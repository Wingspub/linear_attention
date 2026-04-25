# 任务说明

## 1 人工合成任务(synthetic_task)

该任务是用于做模型收敛性分析，即，模型是否能够在这样的数据上进行拟合。

即给定一组符号集合，在该符号集合中随机生成一组固定长度的序列，让模型针对这个序列进行自回归训练以验证模型的收敛性，内存占用和计算量等信息。

参考材料：
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](http://arxiv.org/abs/2006.16236)


## 2 人工合成的复制任务(synthetic_copy_task)

该任务是用于做模型性能分析。

给定一组符号集合，采用0w0w的形式，前者是输入，后者是重复，期望通过前者给定的任意符号，让其能够在后续重复，以验证模型对前者输入的阅读能力。


参考材料
- [Reformer: The Efficient Transformer](http://arxiv.org/abs/2001.04451)


## 3 序列图像生成任务


## 4 文本生成任务


