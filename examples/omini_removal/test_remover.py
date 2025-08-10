"""
测试函数主要的几个修改点
operate parameter要加载lora参数
修改kontext的初始化，替换卷积层为
"""
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
import glob
from PIL import Image
from diffsynth import ModelManager, FluxImagePipeline
from diffsynth.schedulers import FlowMatchScheduler
from diffsynth.models.sd3_dit_3stream import SD3DiT_3_stream
from peft import LoraConfig, inject_adapter_in_model
from base import new_forward,new_forward_mask_condition
import torch.nn as nn
import pandas as pd
# for qwenVL
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
        cfg = 4.5,
        device ="cuda",
        dtype = 'bf16',
        use_mask = False
    ):
        super().__init__()
        self.shift = shift
        self.torch_dtype = torch.bfloat16 if dtype =='bf16' else torch.float16
        self.trained_ckpt_path = trained_ckpt_path
        self.tilesize = 128
        self.tile_stride = 128 - 48
        self.cfg = cfg
        self.device = device
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(pretrained_weights)
        
        self.pipe = FluxImagePipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=self.shift)
        
        self.infer_scheduler = FlowMatchScheduler(num_inference_steps=28)
        self.use_mask = use_mask
        # qwen_path = '/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/Qwen2.5-VL-7B-Instruct'
        # self.qwen_processor = AutoProcessor.from_pretrained(qwen_path, use_fast=True)
            
        # self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_path, torch_dtype=torch.bfloat16, 
        #                                                                     attn_implementation="flash_attention_2",
        #                                                                     device_map="cpu")
        
        self.operate_parameters(lora_ckpt=trained_ckpt_path)
        
        self.neg_caption = "worst quality, low quality, blurring, dirty, messy, jpeg artifacts, zombie, interlocked fingers."
        self.neg_prompt_emb = None


    # def add_conv_in(self):
    #     original_in_channels = self.pipe.dit.x_embedder.in_features
    #     original_out_channels = self.pipe.dit.x_embedder.out_features
        
    #     # 创建新层并确保设备和dtype匹配
    #     new_x_embedder = nn.Linear(original_in_channels * 2, original_out_channels)
    #     new_x_embedder = new_x_embedder.to(device=self.device, dtype=self.torch_dtype)
        
    #     # 复制原始参数
    #     with torch.no_grad():
    #         original_weight = self.pipe.dit.x_embedder.weight.data
    #         original_bias = self.pipe.dit.x_embedder.bias.data
            
    #         new_weight = torch.zeros_like(new_x_embedder.weight.data)
    #         new_bias = torch.zeros_like(new_x_embedder.bias.data)
            
    #         new_weight[:, :original_in_channels] = original_weight.clone()
    #         new_bias[:] = original_bias.clone()
            
    #         new_x_embedder.weight.data = new_weight
    #         new_x_embedder.bias.data = new_bias
        
    #     # 确保新增的embedder存在
    #     self.pipe.dit.LR_x_embedder = new_x_embedder
    #     print(f"Replaced x_embedder. New layer: {self.pipe.dit.x_embedder}")  # 调试输出


    def operate_parameters(self,lora_ckpt=None):
        self.pipe.requires_grad_(False) # vae,clip,t5 to requires_grad_ False 
        self.pipe.eval()
        self.pipe.dit.train()
        print("replacing forward--------------->")
        self.pipe.dit.forward = new_forward.__get__(self.pipe.dit)
        
        #把输入的通道数扩展一下
        #self.add_conv_in()

        # 设置final_norm_out，final_proj_out，time_embedder，guidance_embedder全部训练
        trainable_modules = [
            self.pipe.dit.final_norm_out,
            self.pipe.dit.final_proj_out,
            self.pipe.dit.time_embedder,
            self.pipe.dit.guidance_embedder,
            self.pipe.dit.x_embedder,
        ]

        # 统一设置 requires_grad=True
        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True
        
        original_trainable_params = {
            name for name, param in self.pipe.dit.named_parameters() if param.requires_grad
        }
        
        lora_target_modules = "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp"
        
        # # add lora to new_model
        # # 只对某些不训练的模块加lora 需要训练的模块因为peft库 也不训练了 所以需要置回训练
        lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            init_lora_weights=True,
            target_modules=lora_target_modules.split(",")
        )
        inject_adapter_in_model(lora_config, self.pipe.dit)
        
        for name, param in self.pipe.dit.named_parameters():
            if name in original_trainable_params:
                param.requires_grad = True
        
        for param in self.pipe.dit.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

        # 加载lora模型参数
        # -------- 1. 先加载预训练权重（如果有） --------
        if lora_ckpt is not None:
            state_dict = torch.load(lora_ckpt, map_location="cpu")
            model_state = self.pipe.dit.state_dict()

            # 只保留键名完全匹配的部分
            filtered = {k: v for k, v in state_dict.items() if k in model_state}
            print(f"[INFO] Loading {len(filtered)}/{len(state_dict)} pretrained keys")
            model_state.update(filtered)
            self.pipe.dit.load_state_dict(model_state, strict=False)
        
        #模型不训练
        self.pipe.dit.requires_grad_(False)
        self.pipe.eval()

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
             seed=None,
             use_mask=False,extra_inputs=None
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
        for progress_id, timestep in enumerate(tqdm(self.infer_scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)
            if extra_inputs is None:
                noise_pred_posi = self.pipe.dit(
                    latents, lq_latent, timestep=timestep,  **prompt_emb_posi, **tiler_kwargs,
                )
                noise_pred_nega = self.pipe.dit(
                    latents, lq_latent, timestep=timestep,  **prompt_emb_posi, **tiler_kwargs,
                )
            else:
                noise_pred_posi = self.pipe.dit(
                    latents, lq_latent, timestep=timestep,  **prompt_emb_posi, **tiler_kwargs, **extra_inputs,
                )
                noise_pred_nega = self.pipe.dit(
                    latents, lq_latent, timestep=timestep,  **prompt_emb_posi, **tiler_kwargs, **extra_inputs,
                )
            
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            latents = self.infer_scheduler.step(noise_pred, self.infer_scheduler.timesteps[progress_id], latents)
        # Decode image
        image = self.pipe.decode_image(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        
        return image


@torch.no_grad()
def test(args):
    model = WarpedTest(
        trained_ckpt_path=args.trained_ckpt,
        pretrained_weights=['/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors', 
                            '/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/flux.1-dev/text_encoder/model.safetensors', 
                            '/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/flux.1-dev/text_encoder_2',
                            '/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/flux.1-dev/ae.safetensors'],
        shift=1.0
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    model = model.to(dtype=torch.bfloat16, device='cuda')
    model.device = next(model.parameters()).device
    model.pipe.device = model.device
    
    df = pd.read_csv(args.csv_path)
    os.makedirs(args.output_path, exist_ok=True)
    caption = ""
    for index, row in df.iterrows():
        # 创建输出子目录
        # 因为一张图有多个mask，需要根据mask的index创建名称
        mask_id = row['id']
        img_name = os.path.splitext(os.path.basename(row["image_path"]))[0]
        mask_path = row["mask_path"]
        image_path = row["image_path"]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        #把mask乘上去
        image_tensor = transform(image).unsqueeze(0)
        mask_tensor = transform(mask).unsqueeze(0)
        image_tensor = image_tensor * (1-mask_tensor)
        image_tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image_tensor)

        # obj_label = row.get("object_label", f"obj_{index}")
        output_subdir = os.path.join(args.output_path, f"{img_name}_{mask_id}")
        os.makedirs(output_subdir, exist_ok=True) 
        
        # 保存拼接图像组件
        mask_path = os.path.join(output_subdir, f"mask.jpg")
        img_path = os.path.join(output_subdir, f"img.jpg")
        image.save(img_path)
        mask.save(mask_path)
    

        #prepare latent
        lq_latents = model.pipe.vae_encoder(image_tensor.to(dtype=model.pipe.torch_dtype, device=model.device))
        extra_inputs = model.pipe.prepare_extra_input(latents=lq_latents, guidance=3.5)

        # 使用模型生成结果
        predict_image = model.TI2I(
                prompt=[caption], # same to text[viz_batch_idx]
                negative_prompt=model.neg_caption,
                cfg_scale=1,
                lq_latent=lq_latents,
                denoising_strength=1.0,
                num_inference_steps=40,
                lq_init_noise=False,#完全从噪声出发
                shift=model.shift,
                use_mask = model.use_mask,
                extra_inputs=extra_inputs
            )
        predict_save_path = os.path.join(output_subdir, f"predict.png")
        predict_image.save(predict_save_path)
        
                    
    # image_extensions = ['*.png', '*.jpg', '*.jpeg']
    # image_paths = []
    # for ext in image_extensions:
    #     image_paths.extend(glob.glob(os.path.join(args.input_path, '**', ext), recursive=True))
    
    # image_paths = sorted(image_paths)
    # print("total len:", len(image_paths))
    
    # range_parts = args.start_end.split(',')
    # start_idx = int(range_parts[0]) if range_parts[0] else 0
    # end_idx = int(range_parts[1]) if len(range_parts) > 1 and range_parts[1] else None

    # if end_idx is not None:
    #     image_paths = image_paths[start_idx:end_idx]
    # else:
    #     image_paths = image_paths[start_idx:]

    
    # for img_path in image_paths:
    #     # 读取并转换图像
    #     print(f'deal {img_path}')
    #     filename_without_extension = os.path.splitext(os.path.basename(img_path))[0]
    #     res_path = os.path.join(args.output_path, f"{filename_without_extension}_cfg-{model.cfg}.png")
    #     print(f"will save to {res_path} \n")
    #     img = Image.open(img_path).convert('RGB')
    #     img_tensor = transform(img).unsqueeze(0)

    #     # get caption
    #     # caption = get_prompt(model = model.qwen_model, processor=model.qwen_processor, image = img_tensor*0.5+0.5)
    #     caption = "Remove the object covered by the mask area from this image. Replace the masked region with a seamless and context-appropriate background that matches the surrounding environment. Maintain consistent lighting, textures, and perspective. Ensure no traces of the object remain, and avoid leaving any unnatural blurring, artifacts, or repetitive patterns in the inpainted area. Prioritize photorealism and visual coherence with the original image."        
    #     _, _, h, w = img_tensor.shape
    #     h_adj = round(h * args.scale)
    #     w_adj = round(w * args.scale)
        
    #     lq_tensor = F.interpolate(
    #         img_tensor,
    #         size=(h_adj, w_adj),
    #         mode='bicubic',
    #         align_corners=False
    #     )
        
    #     lq_tensor, _ = adaptive_pad(lq_tensor, tilesize=model.tilesize * 8, stride=model.tile_stride * 8)
    #     _, _, paded_h, paded_w = lq_tensor.shape
    #     lq_tensor = lq_tensor.to(dtype=torch.bfloat16, device='cuda')

    #     # encode hr_tensor to latent space
    #     lq_latent = model.pipe.encode_image(lq_tensor, tiled=True, tile_size=model.tilesize * 8, tile_stride=model.tile_stride * 8)
        
    #     print("origin shape: H*W", h, w)
    #     print("desti shape: H*W", h_adj, w_adj)
    #     print("paded shape: H*W", paded_h, paded_w)
    #     print(f"{paded_h} = {model.tilesize * 8} + {(paded_h - model.tilesize * 8) / (model.tile_stride * 8)} * {model.tile_stride * 8}")
    #     print(f"{paded_w} = {model.tilesize * 8} + {(paded_w - model.tilesize * 8) / (model.tile_stride * 8)} * {model.tile_stride * 8}")
    #     print("lq_latent shape: ", lq_latent.shape)
        
        
            
        
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a test script.")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        default="/mnt/media01/dataset/media_algo_share/lanjinghong/lanjinghong/project/inpainting_pipeline/demo_test/removal_testset.csv",
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
        default=1.0,
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
    #print saved keys:
    # ckpt_path = "/mnt/media01/dataset/media_algo_share/lanjinghong/lanjinghong/project/diffsynth-studio/experiments/kontext_removal_mask_condition/lightning_logs/version_3/checkpoints/epoch=1-step=3000.ckpt"
    # #加载ckpt参数，然后打印所有的key
    # print('\n'.join(torch.load(ckpt_path).keys()))