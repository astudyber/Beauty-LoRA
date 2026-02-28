# 1. 修改引入，使用最通用的 AutoModel
from transformers import AutoModel 

# 模型路径
model_id = "Model/Qwen3-VL-2B" 

# 2. 加载模型
# 使用 AutoModel 可以自动处理 remote_code 定义的特殊结构
model = AutoModel.from_pretrained(model_id, trust_remote_code=True, device_map="cpu")

# 3. 打印结构
output_path = "Model/Qwen3-VL-model-structure.json"

import json
import torch.nn as nn

def build_clean_tree(model):
    """
    构建一个没有 'children' 字段的纯净树状结构。
    元数据使用下划线前缀（如 _type）以区分于子模块名称。
    """
    # 初始化根字典
    tree = {}

    for name, module in model.named_modules():
        # 跳过最外层的空名节点（整个模型），或者你可以手动处理它
        if name == "":
            continue
            
        parts = name.split('.')
        current_level = tree
        
        # 1. 逐层深入（如果路径上的节点不存在，则创建字典）
        for part in parts:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
        
        # 2. 到达当前模块的字典层级后，直接写入元数据
        # 使用 "_" 前缀是为了在视觉上和逻辑上与子模块区分开
        current_level["_type"] = module.__class__.__name__
        
        # 获取权重形状（针对 Linear, Conv 等层）
        if hasattr(module, 'weight') and module.weight is not None:
            current_level["_shape"] = list(module.weight.shape)
            
    return tree

# --- 使用示例 ---
# 假设 model 已经加载
# model = AutoModel.from_pretrained(...)

print("正在生成精简版层级结构...")
clean_structure = build_clean_tree(model)

# 保存为 JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(clean_structure, f, indent=4)

print(f"已保存到 {output_path}")