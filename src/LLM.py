from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationConfig
import torch
from vllm import LLM
import warnings


def build_template():
    prompt_template = "请你基于以下材料回答用户问题。回答要清晰准确，包含正确关键词。不要胡编乱造。如果所给材料与用户问题无关，只输出：无答案。\n" \
                      "以下是材料：\n---" \
                        "{}\n" \
                        "用户问题：\n" \
                        "{}\n" \
                        "务必注意，如果所给材料无法回答用户问题，只输出无答案，不要自己回答。"
    return prompt_template



class LLMPredictor(object):
    def __init__(self, model_path, adapter_path=None, is_chatglm=False, device="cuda", **kwargs):


        self.model = LLM(model=model_path, trust_remote_code=True, dtype = 'bfloat16', gpu_memory_utilization = 0.80)
        self.tokenizer = self.model.get_tokenizer()
        self.tokenizer.pad_token='<|extra_0|>'
        self.tokenizer.eos_token='<|endoftext|>'
        self.tokenizer.padding_side='left'

        self.max_token = 4096
        self.prompt_template = build_template()
        self.kwargs = kwargs
        self.device = torch.device(device)
        print('successful load LLM', model_path)

        self.model_path = model_path

    def get_prompt(self, context, query, bm25 = False, is_yi = False):

        if bm25:
            content = context
        else:
            content = "\n".join(doc.page_content for doc in context)
            
        content = self.prompt_template.format(content, query)
        return content



 
