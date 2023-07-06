#!/usr/bin/env python3

from transformers import AutoModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from time import sleep
import torch

model = AutoModel.from_pretrained("openlm-research/open_llama_3b", 
                                    torch_dtype=torch.float16, 
                                    device_map="cuda") 
print(model.hf_device_map)
sleep(6)
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)

