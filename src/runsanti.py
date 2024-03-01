import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from LLM import LLMPredictor
from embeddings import BGEpeftEmbedding
from langchain import FAISS
from pdfparser import extract_page_text
from bm25 import BM25Model
import torch
import jieba
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids


####对文档进行重新排序（rerank）的功能。####
def rerank(docs, query, rerank_tokenizer, rerank_model, k=5):
    #将 docs 中的文档内容提取出来，存储在 docs_ 列表中。
    docs_ = []
    for item in docs:
        if isinstance(item, str):docs_.append(item)
        else:docs_.append(item.page_content)
    
    docs = list(set(docs_))
    pairs = []
    for d in docs:pairs.append([query, d])
    
    #使用 rerank_tokenizer 对查询和文档内容进行标记化，并组成查询-文档对。将标记化后的查询-文档对输入到 rerank_model 中进行评分。
    with torch.no_grad():
        inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to('cuda')
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().tolist()
    docs = [(docs[i], scores[i]) for i in range(len(docs))]
    #将评分结果与对应的文档内容按照评分值降序排序。返回评分最高的前 k 个文档的内容。
    docs = sorted(docs, key = lambda x: x[1], reverse = True)
    docs_ = []
    for item in docs:docs_.append(item[0])
    return docs_[:k]


####导入了 vllm 模块中的 SamplingParams 类，并创建了一个 sampling_params 对象####
#temperature：温度参数，控制生成文本的多样性。较高的温度会导致更加随机的生成结果。
#top_p：top-p 参数，用于控制采样分布的范围。表示在累积概率达到该值时停止采样。较低的值会使得采样结果更加保守。
#max_tokens：生成文本的最大长度，限制了生成结果的长度。
from vllm import SamplingParams
sampling_params = SamplingParams(temperature=1.0, top_p=0.5, max_tokens=512) #temperature=1.0, top_p=0.5, max_tokens=512

####用于批量推断文本。####
def infer_by_batch(all_raw_text, llm, system="请根据以下给出的背景知识回答问题，对于不知道的信息，直接回答“未找到相关答案”。"):
    #首先遍历all_raw_text中的每个文本，并使用make_context函数生成推断的上下文。然后，将生成的上下文添加到 batch_raw_text 列表中。
    batch_raw_text = []
    for q in all_raw_text:
        raw_text, _ = make_context(llm.tokenizer,q,system=system,max_window_size=6144,chat_format='chatml',)
        batch_raw_text.append(raw_text)
    #接下来，调用 LLM 模型的 generate 方法对 batch_raw_text 中的文本进行生成。
    res = llm.model.generate(batch_raw_text, sampling_params, use_tqdm = False)
    res = [output.outputs[0].text.replace('<|im_end|>', '').replace('\n', '') for output in res]
    return res

def post_process(answer):
    if '抱歉' in answer or '无法回答' in answer or '无答案' in answer:
        return "无答案"
    return answer


#batch_size 变量定义了批处理大小，即一次推断的文本数量。num_input_docs 变量定义了输入文档的数量。
batch_size = 4
num_input_docs = 4
#model 变量指定了主模型的路径，如果 submit 为 True，则使用指定的提交路径，否则使用默认路径。
model = "../models/Qwen-7B-Chat" 
#embedding_path 和 embedding_path2 变量指定了嵌入模型的路径，用于文本嵌入操作。
embedding_path = "../models/gte-large-zh" 
embedding_path2 = "../models/bge-large-zh" 
#reranker_model_path 变量指定了重新排序模型的路径。
reranker_model_path = "../models/bge-reranker-large" 

llm = LLMPredictor(model_path=model, is_chatglm=False, device='cuda:0') 
# llm.model.config.use_flash_attn = True

#加载用于重新排序的模型和相应的分词器，并将模型设置为评估模式（eval），使用半精度浮点数（half）进行计算，并将模型移动到 GPU 上。
rerank_tokenizer = AutoTokenizer.from_pretrained(reranker_model_path)
rerank_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_path)
rerank_model.eval()  #eval() 方法将模型设置为评估模式，这意味着在推理时不会进行梯度计算。
rerank_model.half()  #half() 方法将模型参数转换为半精度浮点数，以减少内存占用和加速计算。
rerank_model.cuda()  #cuda() 方法将模型移动到 GPU 上进行加速计算。
###############################################################################################################################################################################

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter


tokenizer = AutoTokenizer.from_pretrained(model,trust_remote_code=True)

def get_token_len(text: str) -> int:

    tokens = tokenizer.encode(text)
    return len(tokens)

# 1. 从文件读取本地数据集
loader1 = TextLoader("../data/三体1疯狂年代.txt",encoding='gbk')
documents1 = loader1.load()
loader2 = TextLoader("../data/三体2黑暗森林.txt",encoding='gbk')
documents2 = loader2.load()
loader3 = TextLoader("../data/三体3死神永生.txt",encoding='gbk')
documents3 = loader3.load()

documents=documents1+documents2+documents3

# 2. 拆分文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=8, length_function=get_token_len,  separators = ["\n\n", "\n", "。", " ", ""])

docs = text_splitter.split_documents(documents)
corpus = [item.page_content for item in docs]
print(docs[0:2])


# embedding database创建了两个嵌入模型 BGEpeftEmbedding，并使用它们来构建两个嵌入数据库 db 和 db2，然后基于语料库 corpus 构建了一个 BM25 模型 BM25。
embedding_model = BGEpeftEmbedding(model_path=embedding_path)
db = FAISS.from_documents(docs, embedding_model)

embedding_model2 = BGEpeftEmbedding(model_path=embedding_path2)
db2 = FAISS.from_documents(docs, embedding_model2)

BM25 = BM25Model(corpus)


###############################################################################################################################################################################
result_list = []
test_file = "../data/santiQ.json"
with open(test_file, 'r', encoding='utf-8') as f:
    result = json.load(f)

#prompts[x]是控制batch推理用的，all_prompts[x]是给res3用的，来判断关键词在不在里面，ress[x]用于存结果
prompts1, prompts2, prompts3 = [], [], []
all_prompts0,all_prompts1  = [],[]
ress1, ress2, ress3 = [], [], []

for i, line in tqdm(enumerate(result)):
    # bm25 召回，使用 BM25 检索相关文档。
    search_docs1 = BM25.bm25_similarity(line['question']*3, 10)
    # bge 召回，使用嵌入数据库 db2 和 db 进行文档相似性搜索。
    search_docs2 = db2.similarity_search(line['question']*3, k=10)
    # gte 召回
    search_docs3 = db.similarity_search(line['question']*3, k=10)
    # rerank，使用重新排名模型 rerank 对检索结果进行重新排序。
    search_docs4 = rerank(search_docs1 + search_docs2 + search_docs3, line['question'], rerank_tokenizer, rerank_model, k=num_input_docs)

    #prompt1用reranker召回的前4个文档合并起来作为上下文提示，prompt2直接用bm25召回的前4文档合并起来作为上下文提示生成prompt
    prompt1 = llm.get_prompt("\n".join(search_docs4[::-1]), line['question'], bm25=True)
    #prompt2 = llm.get_prompt("\n".join(search_docs1[:num_input_docs][::-1]), line['question'], bm25=True)
    prompts1.append(prompt1)
    #prompts2.append(prompt2)

    #all_prompts0是使用bm25召回3个结果合并作为提示，all_prompts1是使用rerank的1个结果和bm25召回2个结果合并作为提示，
    #all_prompts0.append(search_docs1[0]+'\n'+search_docs1[1]+'\n'+search_docs1[2])
    #all_prompts1.append(search_docs4[0]+'\n'+search_docs1[1]+'\n'+search_docs1[2]+'\n')#

    if len(prompts1)==batch_size:
        ress1.extend(infer_by_batch(prompts1, llm))
        prompts1 = []


if len(prompts1)>0:
    ress1.extend(infer_by_batch(prompts1, llm))

###############################################################################################################################################################################
#对结果进行处理，这里的res[x]是最终的答案，这里result_list是但问题答案汇总
for i, line in enumerate(result):
    res1 = post_process(ress1[i])
    #res2 = post_process(ress2[i])
    line['answer_1'] = res1
    print(line['question'],res1,)
    result_list.append(line)
###############################################################################################################################################################################

