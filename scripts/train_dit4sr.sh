cd /mnt/media01/dataset/media_algo_share/lanjinghong/lanjinghong/project/diffsynth-studio
CUDA_VISIBLE_DEVICES="6,7" python examples/sd35_medium/train.py \
  --deg_file_path "./examples/sd35_medium/deg.yaml" \
  --dataset_txt_paths "/mnt/media01/dataset/media_algo_share/xiangfeng/png_list.txt" \
  --accumulate_grad_batches 1 \
  --learning_rate 1e-5 \
  --null_text_ratio 0.25 \
  --dataloader_num_workers 3 \
  --max_epochs 10 \
  --use_gradient_checkpointing \
  --output_path "./experiments/sd35_dit4sr" \
  --batchsize 8