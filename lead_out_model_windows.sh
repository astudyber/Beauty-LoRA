llamafactory-cli export ^
  --model_name_or_path E:/VLP/Ours/Ours/Beauty-LoRA/Model/Qwen3-VL-2B ^
  --adapter_name_or_path E:/VLP/Ours/Ours/Beauty-LoRA/saves/Qwen3-VL-2B-Instruct/lora/LoRA_Origin/checkpoint-40 ^
  --template qwen ^
  --finetuning_type lora ^
  --export_dir ./Model/LoRA_Origin ^
  --export_size 5 ^
  --export_device cpu
