cd /mnt/media01/dataset/media_algo_share/lanjinghong/lanjinghong/project/diffsynth-studio
pip install -e . -i https://pypi.mirrors.ustc.edu.cn/simple/
CUDA_VISIBLE_DEVICES="0,1,2,3" python examples/omini_removal/remover.py \
  --deg_file_path "./examples/sd35_medium/deg.yaml" \
  --dataset_txt_paths "/mnt/media01/dataset/media_algo_share/xiangfeng/png_list.txt" \
  --accumulate_grad_batches 1 \
  --learning_rate 5e-5 \
  --null_text_ratio 0.25 \
  --dataloader_num_workers 3 \
  --max_epochs 100 \
  --use_gradient_checkpointing \
  --output_path "./experiments/flux_removal_no_mask_condition" \
  --batchsize 4 \
  --json_txt_list /mnt/media01/dataset/media_algo_share/lanjinghong/datasets/syn4removal.txt \
  --fill \
  --ram_path /mnt/media01/dataset/media_algo_share/lanjinghong/lanjinghong/Pretrain_ckpt/ram_swin_large_14m.pth