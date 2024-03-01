# TriBodyQA-LLM
基于三体原文的LLM检索问答模型

# 文件目录

```
. 
├── src # 代码文件
│   ├── bm25.py   # BM25召回模型，依赖jieba分词与rank_bm25库中的BM250kapi模型
│   ├── embeddings.py  # 向量召回嵌入模型，包括对bge和其他模型的嵌入方法，bge使用指令的方式需加前缀
│   ├── LLM.py # 大模型推理，主要包括vllm的初始化，prompt模板构建，chat推理
│   ├── runsanti.py # 主函数入口
│   ├── santi.json # 调试代码用
│   └── qwen_generation_utils.py # qwen工具，直接从qwen开源代码中复制来，主要为了构建qwen的batch推理输入形式
├── models # 模型文件，本地调试时将模型全放在此文件夹中
├── requirements.txt 相关版本信息
└── README.md # 本文件

# 运行结果及流程
运行src/runsanti.py，得到测试santi.json的结果如图
![596b704a6b24bfb3ededed02be14acb](https://github.com/1IsMaple/TriBodyQA-LLM/assets/137876510/592db9ac-be00-4578-bbd2-d052b4ce295c)
