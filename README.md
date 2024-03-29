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
├── models # 模型文件，因大小未上传，下载地址如下，本地调试时将模型全放在此文件夹中
├── requirements.txt 相关版本信息
└── README.md # 本文件
```

# 运行流程及结果
运行src/runsanti.py，得到测试santi.json的结果如图
![596b704a6b24bfb3ededed02be14acb](https://github.com/1IsMaple/TriBodyQA-LLM/assets/137876510/592db9ac-be00-4578-bbd2-d052b4ce295c)

# 使用模型

1、Qwen-7B-Chat

使用官方线上模型

2、gte-large-zh

下载地址：[thenlper/gte-large-zh · Hugging Face](https://huggingface.co/thenlper/gte-large-zh)

3、bge-large-zh

下载地址：[BAAI/bge-large-zh · Hugging Face](https://huggingface.co/BAAI/bge-large-zh)

4、bge-reranker-large

下载地址：[BAAI/bge-reranker-large · Hugging Face](https://huggingface.co/BAAI/bge-reranker-large)


# 主要流程

1.加载LLM、加载embedding模型、加载reranker模型

2.向量知识库构建、BM25知识库构建

3.多路召回与排序，包括bm25召回、bge召回、gte召回，然后使用bge-reranker进行精排，选取得分最高的top-3与问题同时作为输入到llm的上下文。并使用jieba分词对于问题进行分词，加入一层关键词判断，提高匹配精度，同时可根据关键词判断是否有答案。

# 优点

1.使用vllm并且根据Qwen官方脚本完成batch推理，推理速度有较大提升

2.使用了bm25召回、bge召回、gte召回进行多路召回，并使用bge-reranker进行精排，选取top-3作为llm的输入。

3.进行召回时将question复制三倍，使得question长度与召回文档长度不会相差太多，有利于提升召回效果
