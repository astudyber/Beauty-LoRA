> 此文件记录该项目最新的工作进展（2026-2-26  ~  Now）



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
    
# 安装 Flash Attention-2
# https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main
pip install ./Model/Flash-Attention/flash_attn-2.7.4%2Bcu124torch2.5.1cxx11abiFALSE-cp310-cp310-win_amd64.whl

# 安装正确的 transformers 版本 ==5.0.0
# 注意：请尽量采用较高的python版本（python > 3.11）
# 还有一些其他版本冲突问题，已解决，这边强烈推荐采用最新的Llama_Factory以及python和pytorch版本
```



## 初步运行

```
set CUDA_VISIBLE_DEVICES=0 && set GRADIO_SHARE=1 && set GRADIO_SERVER_PORT=6678

llamafactory-cli webui
```







