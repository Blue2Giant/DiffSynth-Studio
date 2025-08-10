import os
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
import glob
from PIL import Image
from diffsynth import ModelManager, SD3ImagePipeline
from diffsynth.models.sd3_dit_3stream import SD3DiT_3_stream

# for qwenVL
from qwen import get_prompt
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


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


class WarpedTest(torch.nn.Module):
    def __init__(
        self,
        trained_ckpt_path,
        pretrained_weights=[],
        shift = 1.0,
        cfg = 4.5
    ):
        super().__init__()
        self.shift = shift
        self.trained_ckpt_path = trained_ckpt_path
        self.tilesize = 128
        self.tile_stride = 128 - 48
        self.cfg = cfg
        
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(pretrained_weights)
        
        self.pipe = SD3ImagePipeline.from_model_manager(model_manager)
        
        qwen_path = '/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/Qwen2.5-VL-7B-Instruct'
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_path, use_fast=True)
            
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_path, torch_dtype=torch.bfloat16, 
                                                                            attn_implementation="flash_attention_2",
                                                                            device_map="cpu")
        
        self.operate_parameters()
        
        self.neg_caption = "worst quality, low quality, blurring, dirty, messy, jpeg artifacts, zombie, interlocked fingers."
        self.neg_prompt_emb = None
        # for k,v in self.neg_prompt_emb.items():
        #     print(k, v.shape) 
        """
            prompt_emb         torch.Size([1, 154, 4096])
            pooled_prompt_emb  torch.Size([1, 2048])
        """
        
    def operate_parameters(self):
        new_model = SD3DiT_3_stream(embed_dim = 1536,
                            num_layers=24,
                            use_rms_norm=True,
                            num_dual_blocks=13,
                            pos_embed_max_size=384)
        # note: very important , bfloat16
        new_model = new_model.to(dtype=torch.bfloat16, device="cpu")
        
        self.pipe.requires_grad_(False) # vae,clip,t5 to requires_grad_ False 
        
        checkpoint = torch.load(self.trained_ckpt_path)
        new_model.init_from_basemodel(self.pipe.dit, trained_state_dict=checkpoint)
        
        self.pipe.dit = new_model
        
        self.pipe.eval()
        
        self.qwen_model.requires_grad_(False)
        self.qwen_model.eval()
    

    @torch.no_grad()
    def TI2I(self, prompt,
             negative_prompt,
             cfg_scale,
             lq_latent,
             denoising_strength,
             num_inference_steps,
             lq_init_noise = True,
             shift=3.0,
             t5_sequence_length=256,
             tiled=False,
             tile_size=128,
             tile_stride=64,
             seed=None
            ):
        """
            return: PIL image
        """
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
        self.pipe.scheduler.set_timesteps(num_inference_steps,
                                           denoising_strength,
                                           training=False,
                                           shift = shift)
        
        b,c,h,w = lq_latent.shape
        assert b == 1
        
        # prepare input
        if lq_init_noise:
            noise = self.pipe.generate_noise((1, 16, h, w), seed=seed, device=self.device, dtype=self.pipe.torch_dtype)
            latents = self.pipe.scheduler.add_noise(lq_latent, noise, timestep=self.pipe.scheduler.timesteps[0])
        else:
            latents = self.pipe.generate_noise((1, 16, h, w), seed = seed, device=self.device, dtype=self.pipe.torch_dtype)
        
        # Encode prompts
        prompt_emb_posi = self.pipe.encode_prompt(prompt, positive=True, t5_sequence_length=t5_sequence_length)
        prompt_emb_nega = self.pipe.encode_prompt(negative_prompt, positive=False, t5_sequence_length=t5_sequence_length)

        # Denoise
        for progress_id, timestep in enumerate(tqdm(self.pipe.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)
            
            noise_pred_posi = self.pipe.dit(
                latents, lq_latent, timestep=timestep, **prompt_emb_posi, **tiler_kwargs,
            )
            noise_pred_nega = self.pipe.dit(
                latents, lq_latent, timestep=timestep, **prompt_emb_nega, **tiler_kwargs,
            )
            
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            
            latents = self.pipe.scheduler.step(noise_pred, self.pipe.scheduler.timesteps[progress_id], latents)
            
        # Decode image
        image = self.pipe.decode_image(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        
        return image


@torch.no_grad()
def test(args):
    model = WarpedTest(
        trained_ckpt_path=args.trained_ckpt,
        pretrained_weights=["/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-medium/text_encoders/clip_g.safetensors",
                                                "/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-medium/text_encoders/clip_l.safetensors",
                                                "/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-medium/text_encoders/t5xxl_fp16.safetensors",
                                                "/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-medium/sd3.5_medium.safetensors"],
        shift=1.0,
        cfg=args.cfg
    )
    
    model = model.to(dtype=torch.bfloat16, device='cuda')
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
    
    for img_path in image_paths:
        # 读取并转换图像
        print(f'deal {img_path}')
        filename_without_extension = os.path.splitext(os.path.basename(img_path))[0]
        res_path = os.path.join(args.output_path, f"{filename_without_extension}_cfg-{model.cfg}.png")
        print(f"will save to {res_path} \n")
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        # get caption
        caption = get_prompt(model = model.qwen_model, processor=model.qwen_processor, image = img_tensor*0.5+0.5)
        
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
        
        lq_tensor, _ = adaptive_pad(lq_tensor, tilesize=model.tilesize * 8, stride=model.tile_stride * 8)
        _, _, paded_h, paded_w = lq_tensor.shape
        lq_tensor = lq_tensor.to(dtype=torch.bfloat16, device='cuda')
        
        # encode hr_tensor to latent space
        lq_latent = model.pipe.encode_image(lq_tensor, tiled=True, tile_size=model.tilesize * 8, tile_stride=model.tile_stride * 8)
        
        print("origin shape: H*W", h, w)
        print("desti shape: H*W", h_adj, w_adj)
        print("paded shape: H*W", paded_h, paded_w)
        print(f"{paded_h} = {model.tilesize * 8} + {(paded_h - model.tilesize * 8) / (model.tile_stride * 8)} * {model.tile_stride * 8}")
        print(f"{paded_w} = {model.tilesize * 8} + {(paded_w - model.tilesize * 8) / (model.tile_stride * 8)} * {model.tile_stride * 8}")
        print("lq_latent shape: ", lq_latent.shape)
        
        image = model.TI2I(
            prompt=caption,
            negative_prompt=model.neg_caption,
            cfg_scale=model.cfg,
            lq_latent=lq_latent,
            denoising_strength=1.0,
            num_inference_steps=28,
            lq_init_noise=True,
            shift=model.shift,
            tiled=True,
            tile_size=model.tilesize,
            tile_stride=model.tile_stride
        )
        
        cropped_image = image.crop((0, 0, w_adj, h_adj))
        cropped_image.save(res_path)
            
        
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
        "--cfg",
        type=float,
        default=4.5,
        help="cfg",
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