#!/usr/bin/env python3

from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from time import sleep
import torch

model = AutoModelForCausalLM.from_pretrained("baichuan-inc/baichuan-7B",
                                                 quantization_config=BitsAndBytesConfig(
                                                    load_in_4bit=True,
                                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                                    bnb_4bit_use_double_quant=True,
                                                    bnb_4bit_quant_type='nf4'
                                                 ),
                                                 device_map="cuda",
                                                 trust_remote_code=True) 
print(model.hf_device_map)
sleep(6)
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)

