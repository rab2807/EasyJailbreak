from easyjailbreak import models
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.attacker.AutoDAN_Liu_2023 import *
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
sys.path.append(os.getcwd())

# model_path = 'lmsys/vicuna-13b-v1.5'
# model_name = 'vicuna_v1.1'
model_path = 'meta-llama/Llama-2-7b-chat-hf'
model_name = 'llama-2'
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizers = AutoTokenizer.from_pretrained(model_path)
target_model = models.HuggingfaceModel(model, tokenizers, model_name)

model_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizers = AutoTokenizer.from_pretrained(model_path)
attack_model = models.HuggingfaceModel(model, tokenizers, 'llama-3')

# 加载数据
dataset_name = '../data_harmful-behaviors-for-easyjailbreak.json'
dataset = JailbreakDataset(dataset_name)

# attacker初始化
attacker = AutoDAN(
    attack_model=attack_model,
    target_model=target_model,
    model_name="llama-2",
    jailbreak_datasets=dataset,
    eval_model=None,
    max_query=100,
    max_jailbreak=100,
    max_reject=100,
    max_iteration=100,
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
    num_steps=100,
    sentence_level_steps=5,
    word_dict_size=30,
    batch_size=64,
    num_elites=0.1,
    crossover_rate=0.5,
    mutation_rate=0.01,
    num_points=5,
    low_memory=1,
    pattern_dict=None
)

attacker.attack()

attacker.attack_results.save_to_jsonl("AdvBench_AutoDAN.jsonl")
