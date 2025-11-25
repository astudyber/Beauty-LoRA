llamafactory-cli export \
  --model_name_or_path /mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/chz/Qwen3/Qwen3-VL-8B \
  --adapter_name_or_path /mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/chz/Qwen3/saves/Qwen3-VL-8B-Instruct/lora/train_model/checkpoint-30 \
  --template qwen \
  --finetuning_type lora \
  --export_dir ./Beauty-LoRA-30 \
  --export_size 5 \
  --export_device cpu
