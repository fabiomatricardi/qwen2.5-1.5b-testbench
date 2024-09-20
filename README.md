# qwen2.5-1.5b-testbench
run GGUF qwen2.5-1.5b with llamaCPP-python


#### LlamaCPP 

This venv contains:
- gradio
- tiktoken 
- llama-cpp-python

It is intended to test LlamaCPP python for qwen2.5 series


### Prompt catalog
The scope is to create an automate verification of the main downstream NLP tasks used the most
- short summarization
- summarization
- topic extraction
- table of Contents
- essay generation
- chit-chat
- content creation
- reflect prompt

### MODELS
- GGUF version of Qwen/Qwen2.5-1.5B-Instruct-GGUF

#### model files:
- qwen2.5-1.5b-instruct-q5_k_m.gguf

#### Model Card from GGUF
```
llm_load_print_meta: model ftype      = Q5_K - Medium
llm_load_print_meta: model params     = 1.78 B
llm_load_print_meta: model size       = 1.19 GiB (5.76 BPW)
llm_load_print_meta: general.name     = qwen2.5-1.5b-instruct
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_tensors: ggml ctx size =    0.15 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/29 layers to GPU
```


