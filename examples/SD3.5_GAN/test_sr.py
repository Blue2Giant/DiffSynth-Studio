"""
    单步模型 任意分辨率超分 推理代码
    注意分块大小要和训练时一致
"""
import os
import argparse
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
import glob
from PIL import Image
from generator import Generator


def adaptive_pad(lq_tensor, tilesize, stride):
    """
    将输入Tensor填充至符合 tilesize + n*stride 的尺寸
    规则：
    1. 若 h/w < tilesize → 直接填充到 tilesize
    2. 若 h/w ≥ tilesize → 计算最小n使得 tilesize + n*stride ≥ h/w，填充差值部分
    3. 仅填充右侧和下侧，使用0填充
    """
    _, _, h, w = lq_tensor.shape
    pad_h = (tilesize - h)  if h <= tilesize else ((h - tilesize + stride - 1) // stride) * stride + tilesize - h
    pad_w = (tilesize - w)  if w <= tilesize else ((w - tilesize + stride - 1) // stride) * stride + tilesize - w
    
    # 右/下侧填充 (PyTorch的pad顺序为左、右、上、下)
    lq_padded = F.pad(lq_tensor, (0, pad_w, 0, pad_h), mode='constant', value=-1)
    return lq_padded, (pad_h, pad_w)


@torch.no_grad()
def test(args):
    tilesize = 100
    tile_stride = tilesize - 25
    
    weight_dtype = torch.bfloat16

    use_sd35_large = True
    if use_sd35_large:
        sd_safe_tensor_path = "/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-large/sd3.5_large.safetensors"
    else:
        sd_safe_tensor_path = "/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-medium/sd3.5_medium.safetensors"

    model = Generator(
        torch_dtype = weight_dtype,
        pretrained_weights=["/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-medium/text_encoders/clip_g.safetensors",
                            "/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-medium/text_encoders/clip_l.safetensors",
                            "/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-medium/text_encoders/t5xxl_fp16.safetensors",
                            sd_safe_tensor_path
                            ],
        learning_rate=0,
        use_gradient_checkpointing=False,
        pretrained_ckpt_path_gen = args.trained_ckpt
    )
    
    model.pipe.requires_grad_(False)
    model = model.to(dtype=weight_dtype, device='cuda')
    model.device = next(model.parameters()).device
    model.pipe.device = model.device
    
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_path, '**', ext), recursive=True))
    
    image_paths = sorted(image_paths)
    print("total len:", len(image_paths))
    
    range_parts = args.start_end.split(',')
    start_idx = int(range_parts[0]) if range_parts[0] else 0
    end_idx = int(range_parts[1]) if len(range_parts) > 1 and range_parts[1] else None

    if end_idx is not None:
        image_paths = image_paths[start_idx:end_idx]
    else:
        image_paths = image_paths[start_idx:]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    os.makedirs(args.output_path, exist_ok=True)
    
    image_paths.insert(0, image_paths[0])

    for idx,img_path in enumerate(image_paths):
        if idx == 1:
            start = time.perf_counter()

        # 读取并转换图像
        print(f'deal {img_path}')
        filename_without_extension = os.path.splitext(os.path.basename(img_path))[0]
        res_path = os.path.join(args.output_path, f"{filename_without_extension}.png")
        print(f"will save to {res_path} \n")
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        # get caption
        caption = "High Contrast, highly detailed, hyper detailed photo - realistic maximum detail, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations."
        
        print(caption)
        
        _, _, h, w = img_tensor.shape
        h_adj = round(h * args.scale)
        w_adj = round(w * args.scale)
        
        lq_tensor = F.interpolate(
            img_tensor,
            size=(h_adj, w_adj),
            mode='bicubic',
            align_corners=False
        )
        
        lq_tensor, _ = adaptive_pad(lq_tensor, tilesize=tilesize * 8, stride=tile_stride * 8)
        _, _, paded_h, paded_w = lq_tensor.shape
        lq_tensor = lq_tensor.to(dtype=weight_dtype, device='cuda')
        
        # encode hr_tensor to latent space
        lq_latent = model.pipe.encode_image(lq_tensor, tiled=True, tile_size=tilesize * 8, tile_stride=tile_stride * 8)
        
        print("origin shape: H*W", h, w)
        print("desti shape: H*W", h_adj, w_adj)
        print("paded shape: H*W", paded_h, paded_w)
        print(f"{paded_h} = {tilesize * 8} + {(paded_h - tilesize * 8) / (tile_stride * 8)} * {tile_stride * 8}")
        print(f"{paded_w} = {tilesize * 8} + {(paded_w - tilesize * 8) / (tile_stride * 8)} * {tile_stride * 8}")
        print("lq_latent shape: ", lq_latent.shape)
        
        image = model.infer(
            prompt=caption,
            lq_latent=lq_latent,
            tiled=True,
            tile_size=tilesize,
            tile_stride=tile_stride
        )
        
        cropped_image = image.crop((0, 0, w_adj, h_adj))
        cropped_image.save(res_path)
            
    end = time.perf_counter()
    print(f"耗时: {end - start:.6f} 秒")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a test script.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input images.",
    )
    parser.add_argument(
        "--trained_ckpt",
        type=str,
        required=True,
        help="Path to trained_ckpt.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./test_outputs",
        help="Path to save the results.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="sr scale",
    )
    parser.add_argument(
        "--start_end",
        type=str,
        default="0,",
        help="index range",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    test(args)