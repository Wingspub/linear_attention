# 任务说明

## 1 人工合成的复制任务(synthetic_copy_task)

该任务可用于做模型收敛性分析、显存占用、计算量等指标计算。

给定一组符号集合 $C$，采用 $0w0w$ 形式。期望通过前面给定字符，在后面重复，以验证模型的长程依赖关系。

参考材料
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](http://arxiv.org/abs/2006.16236)
- [Reformer: The Efficient Transformer](http://arxiv.org/abs/2001.04451)


## 2 序列图像生成任务



## 3 文本生成任务

数据集[主页](http://prize.hutter1.net/): [enwik8](http://mattmahoney.net/dc/enwik8.zip)

给定文本数据集enwik8，将其拆分成训练集和验证集，利用next token prediction任务拟合训练数据，并在验证集上计算拟合损失值。最后通过生成观察生成效果。
