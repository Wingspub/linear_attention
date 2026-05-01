# linear_attention

该仓库是用于探索线性注意力的内容。

## Usage

1. 环境安装
```bash
uv venv
uv sync
```

2. 任务启动
```bash
uv run -m task.synthetic_copy_task # 人工合成任务
uv run -m task.enwik8_task         # 文本拟合任务
```

## Reference

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

```bibtex
@inproceedings{shen2021efficient,
  title={Efficient attention: Attention with linear complexities},
  author={Shen, Zhuoran and Zhang, Mingyuan and Zhao, Haiyu and Yi, Shuai and Li, Hongsheng},
  booktitle={Proceedings of the IEEE/CVF winter conference on applications of computer vision},
  pages={3531--3539},
  year={2021}
}
```

```bibtex
@inproceedings{katharopoulos2020transformers,
  title={Transformers are rnns: Fast autoregressive transformers with linear attention},
  author={Katharopoulos, Angelos and Vyas, Apoorv and Pappas, Nikolaos and Fleuret, Fran{\c{c}}ois},
  booktitle={International conference on machine learning},
  pages={5156--5165},
  year={2020},
  organization={PMLR}
}
```