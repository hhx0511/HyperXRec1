import sys
import os
import torch
import torch.nn as nn
import transformers
import numpy as np
from peft import PeftModel
from datasets import load_from_disk
from transformers import LlamaConfig
from transformers import AutoModel  # ← 你前面用到了 AutoModel，需要确保导入
from model.vamoe import Vmoe_llama3
from transformers import AutoTokenizer
from bert_score import BERTScorer
from model.vae_cluster import Vae_Cluster_Es
from model.config_llama3 import llama_config
from model.moe_layer_llama import MoeBlock_RS
from peft import LoraConfig, TaskType, get_peft_model
from utils.prompt_process import prompt_template,Prompt_Process
from utils.utils import save_gate_index, postprocessing
from pepler_utils.utils import bleu_score, rouge_score
from distinct_n import distinct_n_sentence_level, distinct_n_corpus_level, distinct_n_corpus_whole

import json
import concurrent.futures
from openai import OpenAI  # pip install openai>=1.0

#会计算 P(ref | pred) 和 P(pred | ref) 的概率，如果两句话语义接近且都流畅，分数就会高。
from pepler_utils.bart_score import BARTScorer
# ---------------------------- GPT评分开关与配置 ----------------------------
client = OpenAI(api_key="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")  # 直接写死测试用
system_prompt = ("Score the given explanation against the ground truth on a scale from 0 to 100, focusing on the alignment of meanings rather than the formatting. Provide your score as a number and do not provide any other text.")
def get_gpt_response(prompt):
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        model="gpt-3.5-turbo",
    )
    response = completion.choices[0].message.content
    return float(response)

def get_gpt_score(predictions, references):
    prompts = []
    for i in range(len(predictions)):
        prompt = {
            "prediction": predictions[i],
            "reference": references[i],
        }
        prompts.append(json.dumps(prompt))

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        results = list(executor.map(get_gpt_response, prompts))

    return np.mean(results), np.std(results)

# ---------------------------- 参数封装类 ----------------------------
# 用于保存一些必要的参数（维度/聚类数等）
class Args:
    def __init__(self, embedding_size,latent_dim,num_cluster):
        self.embedding_size = embedding_size   # 用户/物品嵌入维度
        self.latent_dim = latent_dim           # VAE 隐变量维度
        self.num_cluster = num_cluster         # 聚类数
        
'''
    yelp: user: 27147 item: 20266
    tripadvisor: user: 9765 item: 6280
    amazon: user: 7506 item: 7360
'''
# ---------------------------- 数据 & 模型路径设置 ----------------------------
### Inference Setting
n_user         = 7506# amazon 数据集的用户数
n_item         = 7360 # amazon 数据集的物品数
latent_dim     = 16
num_cluster    = 5
embedding_size = 768
vae_model_path = r'D:\hxuser\u430201701\82MegablocksGavaMOE-vmf3\output\Amazon\Amazon\Amazon__cluster_5_epoch_50.pth'     # 预训练的 VAE 模型路径
tokenizer_path = r'D:\hxuser\u430201701\82MegablocksGavaMOE-vmf3\meta-llama\Llama-3-8B-Instruct'# LLaMA 模型路径
llm_model_path = r'D:\hxuser\u430201701\82MegablocksGavaMOE-vmf3\output\explain\checkpoint-11043' # LLaMA 模型的 LoRA checkpoint 路径
data_path      = r'D:\hxuser\u430201701\82MegablocksGavaMOE-vmf3\datasets\cached_datasets\Amazon\test' # HuggingFace 格式的数据集路径
excel_path     = r'D:\hxuser\u430201701\82MegablocksGavaMOE-vmf3\llama_amazon_cluster5.xlsx' # Excel 输出路径
txt_path       = r'D:\hxuser\u430201701\82MegablocksGavaMOE-vmf3\llama_amazon_cluster5.txt' # 评估指标保存路径
# ----------------------------
# 1) 模型加载
# ----------------------------
args = Args(embedding_size = embedding_size, latent_dim = latent_dim, num_cluster = num_cluster)
config = llama_config  # LLaMA 的 config 已在外部定义好（模型结构信息）
#moe_config = Moe_config()
# —— 加载 VAE‑Cluster（只需要 Encoder 的 user/item Embedding）
vae_clu = Vae_Cluster_Es(n_user = n_user, n_item = n_item, args = args)
vae_clu.load_state_dict(torch.load(vae_model_path)) # 加载 VAE 预训练权重
user_embeds = vae_clu.encoder.user_embeddings
item_embeds = vae_clu.encoder.item_embeddings
#加载分词器 & 数据
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

data = load_from_disk(data_path).select(range(10000))# 只取前 10000 条数据加速测试

#获取聚类索引 
test_cluster_index = save_gate_index(data, vae_clu.cuda())# 对每条样本生成聚类 ID（用于路由）
#  构建解释生成模型（含 MoE 门控） --
vmoe_llama3 = Vmoe_llama3(config = config,tokenizer = tokenizer,gate_index_list = test_cluster_index, user_embed = user_embeds, item_embed = item_embeds, use_lora = False)
lora_checkpoint = llm_model_path
# 加载 LoRA 微调模型 
model = PeftModel.from_pretrained(vmoe_llama3, model_id = lora_checkpoint)

model = model.cuda()
# ----------------------------
# 2) 批量推理 & 计时
# ----------------------------
#   torch.cuda.Event 是 GPU 原生计时器，测得的是 **纯推理时间**（同步后更精准）
import pandas as pd
from tqdm import tqdm
df = pd.DataFrame(columns=['userID','itemID','feature','original_text','generate_text', 'inference_time'])
count = 0
# init
# 初始化 GPU 计时器
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
total_inference_time = 0.0
# ---------------------------- 遍历数据逐条生成 ----------------------------
for d in data:
    # — 1) 整理输入
    user = torch.tensor(d['user']).unsqueeze(0)
    item = torch.tensor(d['item']).unsqueeze(0)
    # 只取 62 tokens ⇒ 保持 batch1 单显存可控；超长输入可换 sliding-window
    input_ids = torch.tensor(d['input_ids'][:62]).unsqueeze(0).cuda()
    attention_mask = torch.tensor(d['attention_mask'][:62]).unsqueeze(0).cuda()
    text = d['text']
    # — 2) 推理 + 计时
    start_event.record()
    
    #out = model.generate(user, item, input_ids)
    out = model.generate(user, item, input_ids, attention_mask=attention_mask)


    end_event.record()

    torch.cuda.synchronize() # ***必须同步***，否则计时会提前结束

    inference_time = start_event.elapsed_time(end_event)   # 单位: 毫秒


    total_inference_time += inference_time
    
    inference_time = start_event.elapsed_time(end_event) 
    # — 3) 结果落盘 
    #skip_special_tokens=True 会自动去掉 <s>, </s>, <pad> 等特殊符号，得到干净文本。
    generate_text = tokenizer.decode(out[0], skip_special_tokens=True)#(batch_size, seq_len),batch_size=1，所以 out[0] 取到第 1 条生成序列的 id 列表。
    # index=[0]← 只建立 1 行 DataFrame;ignore_index=True ← 重新分配行号，保持递增 0,1,2…
    df = pd.concat([df,pd.DataFrame({'userID':d['user'],'itemID':d['item'],'feature':d['feature'],'original_text':text,'generate_text':generate_text, 'inference_time': inference_time},index=[0])], ignore_index=True)
    count = count + 1
    print(f'process {count}|{len(data["user"])}')#方便在终端查看当前已处理 / 总样本。
#  输出平均推理时间 
average_inference_time = total_inference_time / len(data)
print(f'average_inference_time: {average_inference_time:.2f} ms')
# —— 保存 Excel；`to_excel` 返回 None，需先赋值再写盘
df = df.to_excel(excel_path)
print('Excel already is saved {}'.format(excel_path))
    
# ----------------------------
# 3) 生成文本质量评估
# ----------------------------
dfm = pd.read_excel(excel_path)
#  防御性过滤：去除 NaN / 过短句子，避免评测脚本崩溃
dfm = dfm.dropna() 
dfm = dfm[dfm['generate_text'].str.len() >= 2]
print(dfm.columns)

# —— 准备 4 个列表
#    y_true / y_pred → 输入 BLEU (分词形式)
#    y_true_txt / y_pred_txt → 输入 ROUGE (整句形式)
y_true = []
y_pred = []
y_pred_distinct = []
print(len(dfm))
for index, row in dfm.iterrows():
     # 1) 原句 & 生成句做统一预处理（去除多余空格/大小写等）
    original_text = [postprocessing(row['original_text'])]
    generate_text = [postprocessing(row['generate_text'])]
    
    # original_text = [row['original_text']]
    # generate_text = [row['generate_text']]
    # 2) BLEU 需要“分词列表” → 这里直接 `.split()`
    original_text = original_text[0].split()
    generate_text = generate_text[0].split()
    # 3) ROUGE 用“整句字符串”
    y_true.append(original_text)
    y_pred.append(generate_text)
    # 4) distinct‑n 多样性统计
    y_pred_distinct.append(row['original_text'].split())
    y_pred_distinct.append(row['generate_text'].split())
    
# — BLEU
BLEU1 = bleu_score(y_true, y_pred, n_gram=1, smooth=True)
print('BLEU-1 {:7.4f}'.format(BLEU1))
BLEU4 = bleu_score(y_true, y_pred, n_gram=4, smooth=True)
print('BLEU-4 {:7.4f}'.format(BLEU4))
# — ROUGE（内部已计算 L / 1 / 2）
text_test = [' '.join(tokens) for tokens in y_true]
text_predict = [' '.join(tokens) for tokens in y_pred]
print("Prediction[0]:", text_predict[0])
print("Reference [0]:", text_test[0])
ROUGE = rouge_score(text_test, text_predict)  # a dictionary
print('ROUGE-1-F {:7.4f}'.format(ROUGE['rouge_1/f_score']))
print('ROUGE-L-F {:7.4f}'.format(ROUGE['rouge_l/f_score']))

# —— 多样性示例（去重后 token 覆盖率）
distinct1 = distinct_n_corpus_level(y_pred_distinct, n=1)
distinct2 = distinct_n_corpus_level(y_pred_distinct, n=2)
print(f"Distinct‑1句子级 {distinct1:.4f}")
print(f"Distinct‑2句子级 {distinct2:.4f}")
distinct1 = distinct_n_corpus_whole(y_pred_distinct, n=1)
distinct2 = distinct_n_corpus_whole(y_pred_distinct, n=2)
print(f"Distinct‑1语料级 {distinct1:.4f}")
print(f"Distinct‑2语料级 {distinct2:.4f}")
# —— BERTScore（使用 RoBERTa-large，英文示例；中文可改 lang="zh"）
# 初始化 BERTScore 评分器
#scorer = BERTScorer(model_type="/home/mail/2023t3/u430201701/hxproject/GavaMOE-vmf/roberta-large", lang="en")
model_path = r"D:\hxuser\u430201701\82MegablocksGavaMOE-vmf3\roberta-large"

# 手动加载本地模型
# 1. 加载 tokenizer 的时候，手动把最大长度限制成 512
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
tokenizer.model_max_length = 512   # ← 关键修复

# 有些版本还要同步改一下 init_kwargs，避免再读旧值
if hasattr(tokenizer, "init_kwargs"):
    tokenizer.init_kwargs["model_max_length"] = 512

model = AutoModel.from_pretrained(model_path)
scorer = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True, device="cuda")#batch_size=64默认
# 手动加载了本地的模型和分词器，然后把它们强制替换到 BERTScorer 对象里
scorer._tokenizer = tokenizer
scorer._model = model.cuda()
# 进行评分
P, R, F1 = scorer.score(text_predict, text_test)


bert_p = P.mean().item()
bert_r = R.mean().item()
bert_f = F1.mean().item()
print(f"BERTScore P {bert_p:.4f} | R {bert_r:.4f} | F1 {bert_f:.4f}")

# 保存评估结果\with open(txt_path, 'w') as f:
with open(txt_path, 'w') as f:
    f.write('BLEU-1 {:7.4f}\n'.format(BLEU1))
    f.write('BLEU-4 {:7.4f}\n'.format(BLEU4))
    for (k, v) in ROUGE.items():
        f.write('{} {:7.4f}\n'.format(k, v))
    f.write(f"Distinct-1 {distinct1:.4f}\nDistinct-2 {distinct2:.4f}\n")
    f.write(f"BERTScore_P {bert_p:.4f}\nBERTScore_R {bert_r:.4f}\nBERTScore_F1 {bert_f:.4f}\n")

# ——（可选）GPT评分：使用GPT对生成解释进行主观评分（1.0~5.0）
#gpt_mean = None
#gpt_std  = None
#gpt_mean, gpt_std = get_gpt_score(predictions=text_predict,references=text_test,)
#print(f"GPTScore Mean {gpt_mean:.4f} | Std {gpt_std:.4f}")

#BART
bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
# P: pred -> ref（模型用“生成”去解释“参考”）
bart_scores_p = bart_scorer.score(text_predict, text_test, batch_size=4)
bart_p_mean = float(np.mean(bart_scores_p))
bart_p_std = float(np.std(bart_scores_p))
print(f"BARTScore-P Mean: {bart_p_mean:.4f}")
print(f"BARTScore-P Std : {bart_p_std:.4f}")


from bleurt.bleurt import score
checkpoint = r"D:\hxuser\u430201701\82MegablocksGavaMOE-vmf3\BLEURT-20"#bleurt/test_checkpoint
scorer = score.BleurtScorer(checkpoint)
scores = scorer.score(references=text_test, candidates=text_predict)
bleurt_mean = np.mean(scores)
bleurt_std  = np.std(scores)
print(f"BLEURT Mean: {bleurt_mean:.4f}")
print(f"BLEURT Std : {bleurt_std:.4f}")