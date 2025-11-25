# -*- coding: utf-8 -*-
# Time: 2025-11-20 ~ now
# Create by: 程惠泽
# Email: hzcheng@chd.edu.cn
# Created by: Visual Studio Code 1.104.0
# 使用1张 V100 显卡测试， 基准模型：QWen3-VL-8B

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from tqdm import tqdm
import json

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/chz/Qwen3/Qwen3-VL-8B", dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/chz/Qwen3/Qwen3-VL-8B")


# 推理
def AesBench(image_path="dataset/images/0a0fec73_0503_4182_bb35_7dba44756913.jpg", Question="", Options=""):
    image = Image.open(image_path).resize((1024, 1024))  #  ！！！ 不调整大小就会爆显存 ！！！

    prompt = f"""
    You are good at conducting aesthetic evaluations of images.
    Select the most suitable answer based on image information and problem description.
    Example input:What aspect of aesthetic perception is most accurately depicted in the image?A) Depth of field variation\nB) Uniform sharpness throughout the image\nC) Vibrant and contrasting color palette\nD)xxxxx
    Example output:D)xxxxx
    Note that apart from selection, you cannot have any other content output.

    The formal input is as follows:
    Question: {Question}
    Options: {Options}
    
    """

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    device = next(model.parameters()).device  # 获取模型所在的设备，比如 cuda:0
    inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])
    return output_text[0]

if __name__ == "__main__":
    # 300 个测试图片 id
    input_id = []
    with open('dataset/eval.txt', 'r') as f:
        for line in f:
            input_id.append(line.strip())
    
    # QAs
    input_data = {}
    with open("dataset/AesBench_evaluation.json", "r") as f:
        input_data = json.load(f)

    # 推理结果
    results = {}
    type = ['AesP', 'AesE', 'AesA1']
    for img in tqdm(input_id):
        results[img] = {}
        for t in type:
            Question = input_data[img][t]["Question"]
            Options = input_data[img][t]["Options"]
            try:
                result = AesBench('./dataset/images/'+img, Question, Options)
            except RuntimeError as e:
                print(f"处理 {img} 时出错")
                continue  # 出错时跳过，继续下一张/下一个问题
            if t == 'AesA1':
                results[img][t] = result[3:].lower()
            else:
                results[img][t] = result
    
    # 保存结果
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)



