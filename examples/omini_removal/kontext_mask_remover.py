import os
import argparse, random
from tqdm import tqdm
import torch
import lightning as pl
from diffsynth import ModelManager, FluxImagePipeline
from diffsynth.schedulers import FlowMatchScheduler
from diffsynth.models.sd3_dit_3stream import SD3DiT_3_stream
from diffsynth.extensions.realesrgan.dataset import PairedSROnlineTxtDataset
from diffsynth.extensions.Removal.dataset_removal import Syn4Removal
# for qwenVL
from diffsynth.extensions.qwen.qwen import get_prompt
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from diffsynth.extensions.ram.models.ram_lora import ram
from diffsynth.extensions.ram import inference_ram as inference
from torchvision import transforms
import pdb
from peft import LoraConfig, inject_adapter_in_model
from base import new_forward,new_forward_mask_condition
import torch.nn as nn
def save_model_info_to_txt(model, filename="model_info.txt"):
    with open(filename, "w") as f:
        for name, param in model.named_parameters():
            # 参数名称
            f.write(f"Parameter: {name}\n")
            # 参数的 shape
            f.write(f"Shape: {tuple(param.shape)}\n")
            # 是否需要梯度（是否可训练）
            f.write(f"Requires grad: {param.requires_grad}\n")
            f.write("-" * 50 + "\n")  # 分隔线


def save_timesteps_sigmas_to_txt(timesteps, sigmas, filename):
    with open(filename, "w") as f:
        f.write(f"timesteps: {timesteps} \n")
        f.write(f"sigmas: {sigmas} \n")


def slice_for_prompt_dict(prompt_dict, slice_start, slice_end):
    for k, v in prompt_dict.items():
        prompt_dict[k] = v[slice_start: slice_end, ...]
    return prompt_dict
    

class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        pretrained_weights=[],
        learning_rate=1e-5,
        use_gradient_checkpointing=True,
        use_qwen = False,
        use_ram = False,
        null_text_ratio = 1/3,
        shift = 1.0,
        dtype = 'bf16',
        device = 'cuda',
        inference_interval = 100,
        use_mask = False
    ):
        super().__init__()
        self.use_qwen = use_qwen
        self.use_ram = use_ram
        self.shift = shift
        self.use_mask = use_mask
        # self.dtype = torch.bfloat16
        # self.device = device
        self.torch_dtype = dtype
        self.torch_device = device
        torch_dtype = torch.bfloat16 if dtype =='bf16' else torch.float16

        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(pretrained_weights)
        
        self.pipe = FluxImagePipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=self.shift)
        
        self.infer_scheduler = FlowMatchScheduler(num_inference_steps=50)
        self.inference_interval = inference_interval
        
        if use_qwen:   
            qwen_path = '/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/Qwen2.5-VL-7B-Instruct'
            self.qwen_processor = AutoProcessor.from_pretrained(qwen_path, use_fast=True)
            
            self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_path, torch_dtype=torch_dtype, 
                                                                            attn_implementation="flash_attention_2",
                                                                            device_map="cpu")
    
        self.operate_parameters()
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.neg_caption = "" # "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw"
        self.neg_prompt_emb = None
        # for k,v in self.neg_prompt_emb.items():
        #     print(k, v.shape) 
        """
            prompt_emb         torch.Size([1, 154, 4096])
            pooled_prompt_emb  torch.Size([1, 2048])
        """    
        self.null_text_ratio = null_text_ratio
        
    def add_conv_in(self):
        original_in_channels = self.pipe.dit.x_embedder.in_features
        original_out_channels = self.pipe.dit.x_embedder.out_features
        
        # 创建新层并确保设备和dtype匹配
        new_x_embedder = nn.Linear(original_in_channels * 2, original_out_channels)
        new_x_embedder = new_x_embedder.to(device=self.device, dtype=self.dtype)
        
        # 复制原始参数
        with torch.no_grad():
            original_weight = self.pipe.dit.x_embedder.weight.data
            original_bias = self.pipe.dit.x_embedder.bias.data
            
            new_weight = torch.zeros_like(new_x_embedder.weight.data)
            new_bias = torch.zeros_like(new_x_embedder.bias.data)
            
            new_weight[:, :original_in_channels] = original_weight.clone()
            new_bias[:] = original_bias.clone()
            
            new_x_embedder.weight.data = new_weight
            new_x_embedder.bias.data = new_bias
        
        # 确保新增的embedder存在
        self.pipe.dit.LR_x_embedder = new_x_embedder
        print(f"Replaced x_embedder. New layer: {self.pipe.dit.x_embedder}")  # 调试输出


    def operate_parameters(self):
        self.pipe.requires_grad_(False) # vae,clip,t5 to requires_grad_ False 
        self.pipe.eval()
        self.pipe.dit.train()
        print("replacing forward--------------->")
        self.pipe.dit.forward = new_forward_mask_condition.__get__(self.pipe.dit)
        
        #把输入的通道数扩展一下
        self.add_conv_in()

        # 设置final_norm_out，final_proj_out，time_embedder，guidance_embedder全部训练
        trainable_modules = [
            self.pipe.dit.final_norm_out,
            self.pipe.dit.final_proj_out,
            self.pipe.dit.time_embedder,
            self.pipe.dit.guidance_embedder,
            self.pipe.dit.x_embedder,
            self.pipe.dit.LR_x_embedder
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
            r=4,
            lora_alpha=4,
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

        if self.use_qwen:
            # qwen model to eval
            self.qwen_model.requires_grad_(False)
            self.qwen_model.eval()


    def training_step(self, batch, batch_idx):
        self.pipe.device = self.torch_device
        
        # if self.neg_prompt_emb is None:
        #     self.neg_prompt_emb = self.pipe.encode_prompt([self.neg_caption], positive=False)
        
        gt, lq = batch['gt'], batch['lq']
        mask = batch['mask']
        bs = gt.shape[0]
        
        ###############################################################################################################
        prompt = "Remove the object covered by the mask area from this image. Replace the masked region with a seamless and context-appropriate background that matches the surrounding environment. Maintain consistent lighting, textures, and perspective. Ensure no traces of the object remain, and avoid leaving any unnatural blurring, artifacts, or repetitive patterns in the inpainted area. Prioritize photorealism and visual coherence with the original image."
        if 'text' in batch.keys():
            text = batch['text']
        else:
            if self.use_qwen:
                text = []
                with torch.no_grad():
                    for i in range(bs):
                        random_value = random.random()
                        if random_value <= self.null_text_ratio:
                            caption = ""
                        else:
                                #gt使用qwen
                            x_gt = gt[i:i+1, :, :, :]*0.5+0.5 # 0~1
                            caption = get_prompt(model = self.qwen_model, processor=self.qwen_processor, image = x_gt)

                        text.append(caption)

            else:
                #使用固定的prompt
                text = [prompt]*bs
                    
        ###############################################################################################################
        prompt_emb = self.pipe.encode_prompt(text, positive=True, t5_sequence_length=256)
        
        ###############################################################################################################
        
        gt_latents = self.pipe.vae_encoder(gt.to(dtype=self.pipe.torch_dtype, device=self.torch_device))
        lq_latents = self.pipe.vae_encoder(lq.to(dtype=self.pipe.torch_dtype, device=self.torch_device))
        mask_latents = self.pipe.vae_encoder(mask.to(dtype=self.pipe.torch_dtype, device=self.torch_device))
        lq_latents = torch.cat([lq_latents, mask_latents], dim=1) #if self.use_mask else lq_latents#将mask和lq在latent上拼接
        ###############################################################################################################
        
        noise = torch.randn_like(gt_latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (bs,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.torch_device)
        # print('timestep',timestep,timestep.dtype)
        noisy_latents = self.pipe.scheduler.add_noise(gt_latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(gt_latents, noise, timestep)
        
        # Compute loss
        extra_inputs = self.pipe.prepare_extra_input(latents=noisy_latents, guidance=3.5)
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, lq_latents, timestep=timestep, **prompt_emb, **extra_inputs,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        # loss = loss * self.pipe.scheduler.training_weight(timestep)
        
        # Record log
        self.log("train_loss", loss, prog_bar=True)
        
        ###############################################################################################################
        # self.trainer.strategy.barrier()
        if self.global_step % self.inference_interval == 1:
            global_rank = self.global_rank
            workdir = self.trainer.logger.log_dir
            viz_batch_idx = 0
            assert viz_batch_idx < bs
            # print(f"\n rank: {global_rank}  batch_idx: {batch_idx} global_step: {self.global_step} \n")
            if global_rank == 0:
                print(text[viz_batch_idx])
                if self.global_step == 0:
                    # 第一次 打印所有参数的梯度是否需要训练信息 以及参数的shape 保存到txt
                    save_model_info_to_txt(self.pipe.dit, os.path.join(workdir, "a_model_params_need_grad_info.txt"))
                    save_timesteps_sigmas_to_txt(self.pipe.scheduler.timesteps, self.pipe.scheduler.sigmas, 
                                                 os.path.join(workdir, "a_timestep_sigma.txt"))
            
            #cfg_scale = random.randint(6, 10)
            cfg_scale =1
            name = f"iter_{self.global_step}_batchidx_{batch_idx}_rank_{global_rank}_cfg_{cfg_scale}.png"
            save_path = os.path.join(workdir, name)
            

            extra_inputs['image_ids'] = extra_inputs['image_ids'][viz_batch_idx : viz_batch_idx + 1, ...]
            extra_inputs['guidance'] = extra_inputs['guidance'][viz_batch_idx : viz_batch_idx + 1, ...]
            # print('extra_inputs ......:',extra_inputs['guidance'],extra_inputs['guidance'].shape)
            image = self.TI2I(
                prompt=slice_for_prompt_dict(prompt_emb, slice_start=viz_batch_idx, slice_end=viz_batch_idx + 1), # same to text[viz_batch_idx]
                negative_prompt=self.neg_caption,
                cfg_scale=cfg_scale,
                lq_latent=lq_latents[viz_batch_idx : viz_batch_idx + 1, ...],
                denoising_strength=1.0,
                num_inference_steps=40,
                lq_init_noise=False,#完全从噪声出发
                shift=self.shift,
                use_mask = self.use_mask,
                extra_inputs=extra_inputs
            )
            image.save(save_path)
            
            name = f"iter_{self.global_step}_batchidx_{batch_idx}_rank_{global_rank}_GT.png"
            save_path = os.path.join(workdir, name)
            image = self.pipe.vae_output_to_image(gt[viz_batch_idx : viz_batch_idx + 1, ...])
            image.save(save_path)
            
            name = f"iter_{self.global_step}_batchidx_{batch_idx}_rank_{global_rank}_LQ.png"
            save_path = os.path.join(workdir, name)
            image = self.pipe.vae_output_to_image(lq[viz_batch_idx : viz_batch_idx + 1, ...])
            image.save(save_path)

            #保存mask
            name = f"iter_{self.global_step}_batchidx_{batch_idx}_rank_{global_rank}_Mask.png"
            save_path = os.path.join(workdir, name)
            image = self.pipe.vae_output_to_image(mask[viz_batch_idx : viz_batch_idx + 1, ...])
            image.save(save_path)
            
            txt_name = f'iter_{self.global_step}_batchidx_{batch_idx}_rank_{global_rank}.txt'
            with open(os.path.join(workdir,txt_name), 'w') as f:
                f.write(batch['to_remove_path'][viz_batch_idx]+'\n')
                f.write(batch['gt_path'][viz_batch_idx]+'\n')
                f.write(text[viz_batch_idx])
        return loss

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
        
        self.infer_scheduler.set_timesteps(num_inference_steps,
                                           denoising_strength,
                                           training=False,
                                           shift = shift)
        
        b,c,h,w = lq_latent.shape
        assert b == 1
        assert extra_inputs['image_ids'].shape[0]==1
        
        # prepare input
        if lq_init_noise:
            noise = self.pipe.generate_noise((1, 16, h, w), seed=seed, device=self.torch_device, dtype=self.pipe.torch_dtype)
            latents = self.infer_scheduler.add_noise(lq_latent, noise, timestep=self.infer_scheduler.timesteps[0])
        else:
            latents = self.pipe.generate_noise((1, 16, h, w), seed=seed, device=self.torch_device, dtype=self.pipe.torch_dtype)

        # Encode prompts
        if isinstance(prompt, dict):
            prompt_emb_posi = prompt
        else:
            prompt_emb_posi = self.pipe.encode_prompt(prompt, positive=True, t5_sequence_length=t5_sequence_length)
        prompt_emb_nega = self.pipe.encode_prompt(negative_prompt, positive=False, t5_sequence_length=t5_sequence_length)
        # Denoise
        for progress_id, timestep in enumerate(tqdm(self.infer_scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.torch_device)
            # print('latents.shape -----------> \n',latents.shape)
            # print('lq_latents.shape------->\n',lq_latent.shape)
            # print('timestep.shape----------->\n',timestep.shape)
            if extra_inputs is None:
                noise_pred_posi = self.pipe.dit(
                    latents, lq_latent, timestep=timestep, **prompt_emb_posi, **tiler_kwargs,
                )
                noise_pred_nega = self.pipe.dit(
                    latents, lq_latent, timestep=timestep, **prompt_emb_posi, **tiler_kwargs,
                )
            else:
                noise_pred_posi = self.pipe.dit(
                    latents, lq_latent, timestep=timestep, **prompt_emb_posi, **tiler_kwargs, **extra_inputs,
                )
                noise_pred_nega = self.pipe.dit(
                    latents, lq_latent, timestep=timestep, **prompt_emb_posi, **tiler_kwargs, **extra_inputs,
                )
            
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            latents = self.infer_scheduler.step(noise_pred, self.infer_scheduler.timesteps[progress_id], latents)
            
        # Decode image
        image = self.pipe.decode_image(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        
        return image
        
    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate, betas=(0.9, 0.95), weight_decay=1e-4)
        return optimizer    
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)

def train(args):
    # dataset = PairedSROnlineTxtDataset(
    #     split="train",
    #     args = args
    # )
    dataset = Syn4Removal(
        split='train',
        json_txt_list= args.json_txt_list,
        use_mask = args.use_mask
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batchsize,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForTrain(pretrained_weights=['/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors', 
                            '/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/flux.1-dev/text_encoder/model.safetensors', 
                            '/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/flux.1-dev/text_encoder_2',
                            '/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/flux.1-dev/ae.safetensors'],
                                learning_rate=args.learning_rate,
                                use_gradient_checkpointing=args.use_gradient_checkpointing,
                                use_qwen = args.use_qwen,
                                use_ram = args.use_ram,
                                null_text_ratio=args.null_text_ratio
                                )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1, every_n_train_steps=1000)],
        logger=None,
    )
    trainer.fit(model, dataloader)
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--deg_file_path",
        type=str,
        default=None,
        required=True,
        help="The path of the deg yaml."
    )
    parser.add_argument(
        "--dataset_txt_paths",
        type=str,
        default=None,
        required=True,
        help="The path of the images."
    )
    parser.add_argument('--highquality_dataset_txt_paths', 
                        type=str, 
                        nargs='?', 
                        default=None, 
                        help='Paths to high quality dataset txt files'
    )
    parser.add_argument(
        "--null_text_ratio",
        type=float,
        default=0,
        help="null_text_ratio",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./experiments",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1,
        help="batchsize",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_qwen",
        default=False,
        action="store_true",
        help="Whether to use qwen to get prompt",
    )
    parser.add_argument(
        "--use_ram",
        default=False,
        action="store_true",
        help="Whether to use ram model to get prompt",
    )
    parser.add_argument(
        '--json_txt_list',
        default='/mnt/media01/dataset/media_algo_share/lanjinghong/datasets/syn4removal.txt',
        help='json txt list'
    )
    parser.add_argument(
        '--ram_path',
        default='/mnt/media01/dataset/media_algo_share/lanjinghong/datasets/ram',
        help='ram path'
    )
    parser.add_argument(
        '--ram_ft_path',
        type=str,
        default='',
        help='ram path finetune'
    )
    parser.add_argument(
        '--use_mask',
        default=False,
        action='store_true',
        help='use mask'
    )
    parser.add_argument(
        '--fill',
        default=False,
        action='store_true',
        help='whether to multiply the mask to image'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)