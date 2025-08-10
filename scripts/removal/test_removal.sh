cd /mnt/media01/dataset/media_algo_share/lanjinghong/lanjinghong/project/diffsynth-studio
CUDA_VISIBLE_DEVICES="1" python examples/omini_removal/test_kontext_removal.py \
    --csv_path /mnt/media01/dataset/media_algo_share/lanjinghong/lanjinghong/project/inpainting_pipeline/demo_test/removal_testset.csv \
    --trained_ckpt /mnt/media01/dataset/media_algo_share/lanjinghong/lanjinghong/project/diffsynth-studio/experiments/kontext_removal_mask_condition/lightning_logs/version_3/checkpoints/epoch=2-step=10000.ckpt \
    --output_path expreiments/removal_testset_kontext_maskcondition_6k
CUDA_VISIBLE_DEVICES="0" python examples/omini_removal/test_remover.py \
    --csv_path /mnt/media01/dataset/media_algo_share/lanjinghong/lanjinghong/project/inpainting_pipeline/demo_test/removal_testset.csv \
    --trained_ckpt /mnt/media01/dataset/media_algo_share/lanjinghong/lanjinghong/project/diffsynth-studio/experiments/flux_removal_no_mask_condition/lightning_logs/version_0/epoch=11-step=27000.ckpt \
    --output_path expreiments/removal_testset_kontext_no_mask_condition