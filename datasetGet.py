# -*- coding: utf-8 -*-
# Time: 2025-11-10 ~ now
# Create by: 程惠泽
# Email: hzcheng@chd.edu.cn
# Created by: Visual Studio Code 1.104.0
# 将官方提供的训练数据集，调整为可以LoRA可微调格式
import json


if __name__ == "__main__":

    # 原格式数据集
    QAs = {}
    with open("dataset/AesBench_evaluation.json", "r") as f:
        QAs = json.load(f)
    data = {}
    with open('dataset/200GT.json', 'r') as f:
        data = json.load(f)

    # 微调格式数据集
    dataset = []
    type = ['AesP', 'AesE', 'AesA1']
    for k,v in data.items():
        for t in type:
            Que = QAs[k][t]["Question"]  # 问题
            Opt = QAs[k][t]["Options"]  # 选项
            onedata = {
                "conversations": [
                    {
                        "value": f"<image>Question: {Que} Options: {Opt}",
                        "from": "human"
                    },
                    {
                        "value": v[t],
                        "from": "gpt"
                    }
                ],
                "images": [
                    "dataset/images/" + k
                ]
            }
            dataset.append(onedata)


    # 写入文件中
    with open('data/gt.json', 'w') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


