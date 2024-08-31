import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.attacker.Cipher_Yuan_2023 import *
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.huggingface_model import HuggingfaceModel, from_pretrained

sys.path.append(os.getcwd())

generation_config = {'max_new_tokens': 100}
# llama_model_path = 'meta-llama/Llama-2-7b-chat-hf'
# model_name = 'llama-2'
llama_model_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
model_name = 'llama-3'
model = AutoModelForCausalLM.from_pretrained(llama_model_path)
tokenizers = AutoTokenizer.from_pretrained(llama_model_path)
hf_model = HuggingfaceModel(
    model=model, tokenizer=tokenizers, model_name=model_name, generation_config=generation_config)

# chat_name = 'GPT-4'
# api_key = 'your key'
# GPT4 = OpenaiModel(model_name=chat_name, api_keys=api_key)

dataset_name = '../data_harmful-behaviors-for-easyjailbreak.json'
num_attack = 1
dataset = JailbreakDataset(dataset_name)
dataset._dataset = dataset._dataset[:num_attack]

attacker = Cipher(attack_model=None,
                  target_model=hf_model,
                  eval_model=hf_model,
                  jailbreak_datasets=dataset)

attacker.attack()
attacker.log()
attacker.attack_results.save_to_jsonl('AdvBench_cipher.jsonl')
