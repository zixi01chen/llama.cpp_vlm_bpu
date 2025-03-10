### 1. 如何导出语言模型？

介绍如何导出vlm中的语言模型

tip：此过程可在pc(x86)上处理，更高效

1. 拉取代码
```shell
git clone https://github.com/ggerganov/llama.cpp -b b4749
```

2. 下载模型文件
```shell
git clone https://huggingface.co/OpenGVLab/InternVL2_5-1B
# 或 git clone https://hf-mirror.com/OpenGVLab/InternVL2_5-1B
```

3. 更新配置文件到模型文件夹中
```shell
cp utils/InternVL2_5-1B/config.json OpenGVLab/InternVL2_5-1B
```

4. 转换
```shell
cp utils/convert_hf_to_gguf_intern2.py llama.cpp
cd llama.cpp
python3 convert_hf_to_gguf.py OpenGVLab/InternVL2_5-1B
```

```bash
INFO:hf-to-gguf:blk.9.attn_v.weight,       torch.bfloat16 --> F16, shape = {896, 128}
language_model.model.norm.weight
output_norm.weight
INFO:hf-to-gguf:output_norm.weight,        torch.bfloat16 --> F32, shape = {896}
INFO:hf-to-gguf:Set meta model
INFO:hf-to-gguf:Set model parameters
INFO:hf-to-gguf:gguf: context length = 32768
INFO:hf-to-gguf:gguf: embedding length = 896
INFO:hf-to-gguf:gguf: feed forward length = 4864
INFO:hf-to-gguf:gguf: head count = 14
INFO:hf-to-gguf:gguf: key-value head count = 2
INFO:hf-to-gguf:gguf: rope theta = 1000000.0
INFO:hf-to-gguf:gguf: rms norm epsilon = 1e-06
INFO:hf-to-gguf:gguf: layer norm epsilon = 24
INFO:hf-to-gguf:gguf: file type = 1
INFO:hf-to-gguf:Set model tokenizer
INFO:gguf.vocab:Adding 151387 merge(s).
INFO:gguf.vocab:Setting special token type bos to 151643
INFO:gguf.vocab:Setting special token type eos to 151645
INFO:gguf.vocab:Setting add_bos_token to False
INFO:gguf.vocab:Setting add_eos_token to False
INFO:gguf.vocab:Setting chat_template to {%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}

INFO:hf-to-gguf:Set model quantization version
INFO:gguf.gguf_writer:Writing the following files:
INFO:gguf.gguf_writer:/data/huggingface/OpenGVLab/InternVL2_5-1B/Qwen2.5-0.5B-Instruct-F16.gguf: n_tensors = 291, total_size = 1.3G
Writing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.26G/1.26G [00:06<00:00, 192Mbyte/s]
INFO:hf-to-gguf:Model successfully exported to /data/huggingface/OpenGVLab/InternVL2_5-1B/Qwen2.5-0.5B-Instruct-F16.gguf
```
