export FLUX_MINI=/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/flux-mini/flux-mini.safetensors
export FLUX_DEV=/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors
# export FLUX_DEV=/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/flux.1-dev/flux1-dev.safetensors
export AE=/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/flux.1-dev/ae.safetensors
export HF_HOME=/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained
cd /mnt/media01/dataset/media_algo_share/lanjinghong/lanjinghong/project/diffsynth-studio
accelerate launch --num_processes=2 --gpu_ids="6,7" --main_process_port 29300 examples/flux_gan/train_fabric_gan.py \
  --mmaigc_dataset_yml "./examples/flux_gan/config.yaml" \
  --deg_file_path "./examples/sd35_medium/deg.yaml" \
  --dataset_txt_paths "/mnt/media01/dataset/media_algo_share/xiangfeng/png_list.txt" \
  --null_text_ratio 0.01 \
  --task train 