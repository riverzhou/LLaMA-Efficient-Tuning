#!/usr/bin/env python3

from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from time import sleep
import torch

model = AutoModelForCausalLM.from_pretrained("baichuan-inc/baichuan-7B",
                                                load_in_8bit=True,
                                                device_map="cuda",
                                                trust_remote_code=True)
print(model.hf_device_map)
sleep(6)
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)

