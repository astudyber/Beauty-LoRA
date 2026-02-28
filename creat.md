> 此文件记录该项目最新的工作进展（2026-2-26  ~  Now）



# 1. 环境

## 创建.gitignore文件

作用：github仓库上传时，忽略模型文件、数据集文件、缓存文件



## 下载数据集 和 基底模型

**数据集**：dataset/

**基底模型**：Model/Qwen3-VL-2B/



## 配置 LoRA 环境

```python
# 当前环境以及包含 torch-2.5.1+cu124
conda activate pytorch
cd LoRA
pip install -e .
pip install -r requirements/metrics.txt
cd ..

# 适用于 CUDA==12.4 的 BitsAndBytes 库下载轮子文件
# https://github.com/jllllll/bitsandbytes-windows-webui/releases/tag/wheels
pip install E:\VLP\第一届人工智能创新大赛（校赛）\Ours(参赛)\Beauty-LoRA\Model\bitsandbytes\bitsandbytes-0.41.1-py3-none-win_amd64.whl
    
pip install -U bitsandbytes>=0.46.1
    
# 安装 Flash Attention-2
# https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main
pip install ./Model/Flash-Attention/flash_attn-2.7.4%2Bcu124torch2.5.1cxx11abiFALSE-cp310-cp310-win_amd64.whl

# 安装正确的 transformers 版本 ==5.0.0
# 注意：请尽量采用较高的python版本（python > 3.11）
# 还有一些其他版本冲突问题，已解决，这边强烈推荐采用最新的Llama_Factory以及python和pytorch版本
```



## 初步运行  +  本地微调 Qwen3-VL-2B

```
set CUDA_VISIBLE_DEVICES=0 && set GRADIO_SHARE=1 && set GRADIO_SERVER_PORT=6678

llamafactory-cli webui
```



## 进入训练微调

```python
# 训练参数保存路径
llamaboard_config/2026-02-26-16-38-15.yaml

# 模型微调的AB矩阵保存路径
saves/Qwen3-VL-2B-Instruct/lora/输出目录名

# 模型合并命令（windows）
llamafactory-cli export ^
  --model_name_or_path E:/VLP/Ours/Ours/Beauty-LoRA/Model/Qwen3-VL-2B ^
  --adapter_name_or_path E:/VLP/Ours/Ours/Beauty-LoRA/saves/Qwen3-VL-2B-Instruct/lora/LoRA_Origin/checkpoint-40 ^
  --template qwen ^
  --finetuning_type lora ^
  --export_dir ./Model/LoRA_Origin ^
  --export_size 5 ^
  --export_device cpu

# 模型合并命令（windows）
llamafactory-cli export \
  --model_name_or_path E:/VLP/Ours/Ours/Beauty-LoRA/Model/Qwen3-VL-2B \
  --adapter_name_or_path E:/VLP/Ours/Ours/Beauty-LoRA/saves/Qwen3-VL-2B-Instruct/lora/LoRA_Origin/checkpoint-40 \
  --template qwen \
  --finetuning_type lora \
  --export_dir /Model/LoRA_Origin \
  --export_size 5 \
  --export_device cpu

```





# 2. 数据集增强

注：实例效果对比，由于每一次文本增强的结果具有随机性，此处采用典型案例进行说明

数据量增长220%：原数据集包含   8400  个QA对；数据增强后为  26880  个QA对

### 原句

```python
'The content of the picture is vivid and aesthetically pleasing.'
图片的内容生动形象，富有美感。
```

### A. BERT 随机插入（100%）

```python
'The original content form of all the picture frames is vivid and aesthetically pleasing.'
所有图像的原始内容形式都生动美观。
```

* **分析**：保留核心谓语和宾语， 对主语进行了修饰和扩充（添加了 original, form, frames），让模型学会即使句子变长、变复杂，核心含义依然不变。 
* **泛化能力**：非常适合训练多模态模型处理不同风格的描述。

### B. 同义词替换（100%）

```python
'The content of the picture is vivid and esthetically please.'
图片的内容生动美观，请欣赏。
```

* **分析**：将 "aesthetically" 换成了美式拼写 "esthetically"，但将 "pleasing" 误写成了 "please"。
* **模拟真实噪声**：在实际应用中，用户的输入往往是不完美的，这一项模拟了同义替换和轻微语法错误。

###  C. BERT 上下文替换（20%）

```python
'The content and every picture were physically and aesthetically pleasing.'
内容和每张照片令人身心愉悦。
```

* **分析**：把一张图的内容”改成了“内容和每一张图”，并加入了“physically”（物理上的）。 在多模态任务中，可能增加泛化性，但也可能带来幻觉。 

### D. 其他方式（0%）

```python
# 词向量替换
'The combining of the portrait is vivid from unappealing pleasing.'
这幅图像描绘的内容既生动又令人乏味的愉快。

# 键盘输入错误模拟
'The f*n%ent of the pictKdF is vivid and sestYetisal>y pleaWiJn.'
图片KdF的f*n%ent是生动的，并且是完整的。

# 随机词删除
'The content of vivid and aesthetically.'
内容生动美观。
```

* **分析**：逻辑混乱。“unappealing pleasing”属于矛盾修饰 

* **分析**：过度噪声。 这种数据毫无意义，只会降低模型对正常语言的理解。 
* **分析**：信息严重丢失（丢失主语信息“图像”，缺乏定语修饰）



# 3. 模型结构优化

### 3.1 确定 Qwen3-VL 的网络结构

```python
# 总结如下
Qwen3-VL-2B:{
    visual:{
        'patch_embed': Conv3d [1024, 2, 3, 16, 16]
        'pos_embed': Embedding [2304, 1024]
        'rotary_pos_emb': Qwen3VLVisionRotaryEmbedding
        'blocks': 23 × {
            norm1: LayerNorm  [1024]
            norm2: LayerNorm  [1024]
            attn: {
                qkv: Linear [3072, 1024]
                proj: Linear [1024, 1024]
            }
            mlp: {
                linear_fc1: Linear [4096, 1024]
                linear_fc2: Linear [1024, 4096]
                act_fn: GELUTanh
            }
        }
        'merger': {
            norm: LayerNorm  [1024]
            linear_fc1:  Linear [4096, 4096]
            act_fn: GELU
            linear_fc2:  Linear [2048, 4096]
        }
        'deepstack_merger_list': 3 × {
            norm: LayerNorm  [4096]
            linear_fc1:  Linear [4096, 4096]
            act_fn: GELU
            linear_fc2:  Linear [2048, 4096]
        }
    }
    
    language_model.layers.*.self_attn.q_proj
    language_model:{
        'embed_tokens': Embedding [151936, 2048]
        'layers': 27 × {
            self_attn: {
                q_proj:  Linear [2048, 2048]
                k_proj:  Linear [1024, 2048]
                v_proj:  Linear [1024, 2048]
                o_proj:  Linear [2048, 2048]
                q_norm:  Qwen3VLTextRMSNorm [128]
                k_norm:  Qwen3VLTextRMSNorm [128]
            }
            mlp: {
                gate_proj:  Linear [6144, 2048]
                up_proj  :  Linear [6144, 2048]
                down_proj:  Linear [2048, 6144]
                act_fn   :  SiLUActivation
            }
            input_layernorm: Qwen3VLTextRMSNorm [2048]
            post_attention_layernorm: Qwen3VLTextRMSNorm [2048]
        }
        'norm': Qwen3VLTextRMSNorm [2048]
        'rotary_emb': Qwen3VLTextRotaryEmbedding
    }
}
```

* 默认的 LoRA 微调的低秩矩阵具体添加在：
  * 语言模块 language_model中的所有线性层，包括下面的线性层
  * `q_proj`、`k_proj`、`v_proj`、`o_proj`
  * `gate_proj`、`up_proj`、`down_proj`

### 3.2 改进 LoRA （视觉 visual）

* 解冻视觉编码器
* 为如下网络层添加低秩矩阵（主要是基础视觉模块）：
  * `visual.blocks.*.attn.qkv`
  * `visual.blocks.*.attn.proj`
  * `visual.blocks.*.mlp.linear_fc1`
  * `visual.blocks.*.mlp.linear_fc2`

* 简洁写法：
  * `qkv`、`proj`、`linear_fc1`、`linear_fc2`



### 3.3 采用 MOE 多型复合矩阵（语言 language）













# 4. 模型推理优化









# 5. 模型测评













