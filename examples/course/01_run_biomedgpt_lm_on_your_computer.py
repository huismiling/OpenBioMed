
import torch
import torch_mlu
import transformers  # 导入transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
# 加载模型
# 模型和tokenizer文件存放路径
model_path = '/projs/AE/zhangxiao1/work/huggingface_models/BioMedGPT-LM-7B'
model = AutoModelForCausalLM.from_pretrained(model_path, 
            torch_dtype=torch.float16,
            device_map="auto")
#print(model)
model.config
# 加载模型对应的tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
text = ["What's the function of Aspirin?"]
# 使用tokenizer处理文本
input = tokenizer(text,
            truncation=True,
            return_tensors="pt")

output = model.generate(inputs=input.input_ids.mlu(), max_new_tokens=128, early_stopping=True)

# 解码成文字
output = tokenizer.decode(output[0])
print(text[0], output)

