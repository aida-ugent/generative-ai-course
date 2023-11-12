# accelerate launch --mixed_precision="fp16"  --num_processes=1 train_text_to_image_lora.py \
#   --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#   --train_data_dir="./tmp/lora/rorschach_inkblot_512" \
#   --dataloader_num_workers=8 \
#   --resolution=512 --center_crop --random_flip \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=15000 \
#   --learning_rate=1e-04 \
#   --max_grad_norm=1 \
#   --lr_scheduler="cosine" --lr_warmup_steps=0 \
#   --output_dir="./tmp/lora/rorschach_inkblot_model" \
#   --checkpointing_steps=500 \
#   --validation_prompt="UGENTAIGC3 butterfly" \
#   --validation_epochs=10 \
#   --seed=0


accelerate launch --mixed_precision="fp16"  --num_processes=1 train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="./tmp/lora/flemish_tapestry_512" \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir="./tmp/lora/flemish_tapestry_model" \
  --checkpointing_steps=500 \
  --validation_prompt="UGENTAIGC3 butterfly" \
  --validation_epochs=10 \
  --seed=0