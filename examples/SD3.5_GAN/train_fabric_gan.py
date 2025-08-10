"""
生成器不用prompt
判别器概率使用prompt
prompt来自qwen
单步 gan训练
"""
import os
import shutil
import torch
from datetime import datetime
import random
from contextlib import redirect_stdout, nullcontext
from torchvision import transforms
from utils import yaml_load, parse_args

from ganloss import GANLoss
from generator import Generator
from discriminator import Discriminator

from diffsynth.extensions.realesrgan.dataset import PairedSROnlineTxtDataset
from copy import deepcopy
from accelerate import Accelerator
from accelerate.utils import set_seed
from lightning.fabric.loggers import TensorBoardLogger

import lpips
from elatentlpips.elatentlpips import ELatentLPIPS
import matplotlib.pyplot as plt
import numpy as np
from srcnn import SRCNN
from core import imresize

def adaptive_relaxed_mean(tensor, output_size=10):
    if output_size >= 100:
        return tensor
    else:
        # 使用自适应平均池化然后上采样
        orig_size = tensor.shape[2:]
        pooled = torch.nn.functional.adaptive_avg_pool2d(tensor, (output_size, output_size))
        return torch.nn.functional.interpolate(pooled, size=orig_size, mode='bilinear')

def add_gaussian_noise(input_tensor, variance=0.01):
    # 计算标准差
    std = variance ** 0.5  # sqrt(0.01) = 0.1
    
    # 生成相同形状的噪声
    noise = torch.randn_like(input_tensor) * std
    
    # 添加到原始输入
    return input_tensor + noise

@torch.no_grad()
def pre_process_batch(batch, do_crop):
    assert batch['lq'].shape[0] == 1
    
    lq = batch['lq']
    gt = batch['gt']

    assert lq.shape[2] == gt.shape[2] and lq.shape[3] == gt.shape[3] 
    
    if do_crop:
        H, W = lq.shape[2], lq.shape[3]
        
        crop_H = 1024 # random.randint(32, 50) * 16 # 512 ~ 800
        crop_W = 1024 # random.randint(32, 50) * 16
    
        start_H = torch.randint(0, H - crop_H + 1, (1,)).item()
        start_W = torch.randint(0, W - crop_W + 1, (1,)).item()
        end_H = start_H + crop_H
        end_W = start_W + crop_W

        lq = lq[:, :, start_H:end_H, start_W:end_W]
        gt = gt[:, :, start_H:end_H, start_W:end_W]
    
    return lq, gt


def crop_for_bchw(model_pred, cropsize, gt_rgb):
    batch_size, channels, height, width = model_pred.shape
    assert cropsize < height and cropsize < width
    
    start_h = np.random.randint(0, height - cropsize + 1)
    start_w = np.random.randint(0, width - cropsize + 1)
    
    cropped_latent = model_pred[:, :, start_h:start_h + cropsize, start_w:start_w + cropsize]
    
    gt_start_h = start_h * 8
    gt_start_w = start_w * 8
    gt_cropsize = cropsize * 8
    
    cropped_gt = gt_rgb[:, :, gt_start_h:gt_start_h + gt_cropsize, gt_start_w:gt_start_w + gt_cropsize]
    
    return cropped_latent, cropped_gt

def get_start_timestep_given_iteration(iteration, step_thr = 10000):
    if iteration > step_thr:
        return 800
    else:
        return int(iteration / step_thr * 800)

def train(args):
    dataset_yaml = yaml_load(args.mmaigc_dataset_yml)
    gradient_accumulation_steps = dataset_yaml['accumulate_grad_batches']
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='bf16',
        log_with='tensorboard',
        project_dir = os.path.join("./experiments", dataset_yaml['exp_tag'])
    )
    
    set_seed(42)

    dataset = PairedSROnlineTxtDataset(
        split="train",
        args=args
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=2
    )
    
    use_sd35_large = 'large' in dataset_yaml['exp_tag']
    if use_sd35_large:
        sd_safe_tensor_path = "/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-large/sd3.5_large.safetensors"
    else:
        sd_safe_tensor_path = "/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-medium/sd3.5_medium.safetensors"

    generator = Generator(
        torch_dtype = torch.bfloat16,
        pretrained_weights=["/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-medium/text_encoders/clip_g.safetensors",
                            "/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-medium/text_encoders/clip_l.safetensors",
                            "/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-medium/text_encoders/t5xxl_fp16.safetensors",
                            sd_safe_tensor_path
                            ],
        learning_rate=dataset_yaml['learning_rate'],
        use_gradient_checkpointing=dataset_yaml['use_gradient_checkpointing'],
        pretrained_ckpt_path_gen = dataset_yaml['pretrained_ckpt_path_gen']
    )
    
    discriminator = Discriminator(
        torch_dtype = torch.bfloat16,
        pretrained_weights=["/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-medium/sd3.5_medium.safetensors"],
        learning_rate=dataset_yaml['learning_rate_dis'],
        use_gradient_checkpointing=dataset_yaml['use_gradient_checkpointing'],
        pretrained_ckpt_path_dis = dataset_yaml['pretrained_ckpt_path_dis']
    )
    
    cri_gan = GANLoss(gan_type = dataset_yaml['gan_type'], loss_weight = dataset_yaml['gan_loss_weight'], 
                      real_label_val=dataset_yaml['real_label_val'], fake_label_val = dataset_yaml['fake_label_val'])
    
    optimizer_g = generator.configure_optimizers()
    optimizer_d = discriminator.configure_optimizers()
  
    generator.pipe.device = accelerator.device
    discriminator.pipe.device = accelerator.device
    
    net_lpips = lpips.LPIPS(net='vgg')
    # net_lpips = ELatentLPIPS(pretrained=True, net='vgg16', encoder='flux', augment=None)
    net_lpips.requires_grad_(False)
    net_lpips.eval()
    
    ddv2 = SRCNN(3, 3, nf = 32, nb=20, downscale=1, scale=2)
    pth_path = '/mnt/media01/dataset/media_algo_share/xiangfeng/repos/diffsynth-studio/ckpt/net_g_185050.pth'
    ddv2_model_params = torch.load(pth_path)['params']
    ddv2.load_state_dict(ddv2_model_params, strict=True)
    ddv2.requires_grad_(False)
    ddv2.eval()
    
    generator, discriminator, net_lpips, optimizer_g, optimizer_d, dataloader = accelerator.prepare(
        generator, discriminator, net_lpips, optimizer_g, optimizer_d, dataloader
    )
    
    ddv2.to(device = accelerator.device)
    
    # initial ram model
    # ram_path = args.ram_path
    
    # ram_transforms = transforms.Compose([
    #     transforms.Resize((384, 384)),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    
    # model_vlm = ram(pretrained=ram_path,
    #         pretrained_condition=None,
    #         image_size=384,
    #         vit='swin_l')
    # model_vlm.eval()
    # model_vlm.to(device = fabric.device, dtype=torch.bfloat16)
    
    # initial qwen 7b model
    # qwen_path = dataset_yaml['qwen_path']
    # qwen_processor = AutoProcessor.from_pretrained(qwen_path, use_fast=True)
    # qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_path, torch_dtype=torch.bfloat16, 
    #                                                                 attn_implementation="flash_attention_2",
    #                                                                 device_map=fabric.device)
    
    
    if accelerator.is_main_process:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        workdir = os.path.join("./experiments", dataset_yaml['exp_tag'], current_time)
        os.makedirs(workdir, exist_ok=False)
        trt_logger = TensorBoardLogger(workdir, name="tensorboard")

        # backup config file
        yaml_file_name = os.path.basename(args.mmaigc_dataset_yml)
        target_path = os.path.join(workdir, yaml_file_name)
        shutil.copy2(args.mmaigc_dataset_yml, target_path)
        
        # plot model structure
        filename = os.path.join(workdir, 'generator_model_structure.txt')
        with open(filename, "a") as f:
            with redirect_stdout(f):
                print(generator.module.pipe.denoising_model())
                
        filename = os.path.join(workdir, 'discriminator_model_structure.txt')
        with open(filename, "a") as f:
            with redirect_stdout(f):
                print(discriminator.module.pipe.denoising_model())
        
        # 获取生成器可训练参数名字并写入文件
        filename = os.path.join(workdir, 'generator_trainable_parameters.txt')
        with open(filename, "a") as f:
            with redirect_stdout(f):
                for name, param in generator.named_parameters():
                    if param.requires_grad:
                        print(name)

        # 获取判别器可训练参数名字并写入文件
        filename = os.path.join(workdir, 'discriminator_trainable_parameters.txt')
        with open(filename, "a") as f:
            with redirect_stdout(f):
                print("Trainable parameters in discriminator:")
                for name, param in discriminator.named_parameters():
                    if param.requires_grad:
                        print(name)
        
        # plot generator.pipe_G.scheduler
        data_np = generator.module.pipe.scheduler.sigmas.numpy()
        plt.plot(data_np, marker='o')
        plt.title('Line Plot of sigmas')
        plt.xlabel('timesteps_id')
        plt.ylabel('sigma')
        # 显示网格（可选）
        plt.grid(True)
        # 保存图像到硬盘
        plt.savefig(os.path.join(workdir, 'line_plot_G.png')) 
        
        plt.clf()
        
        data_np = discriminator.module.pipe.scheduler.sigmas.numpy()
        plt.plot(data_np, marker='o')
        plt.title('Line Plot of sigmas')
        plt.xlabel('timesteps_id')
        plt.ylabel('sigma')
        # 显示网格（可选）
        plt.grid(True)
        # 保存图像到硬盘
        plt.savefig(os.path.join(workdir, 'line_plot_D.png')) 
    
    neg_caption = "" # "Bright tones, overexposed, blurred details, subtitles, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, walking backwards"
    # neg_prompt_emb = generator.pipe_G.encode_prompt(neg_caption, positive=False)
    t5_sequence_length = 77
    prompt_emb_G = generator.module.pipe.encode_prompt("High Contrast, highly detailed, hyper detailed photo - realistic maximum detail, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.", t5_sequence_length=t5_sequence_length)
    # null_text_ratio = dataset_yaml['null_text_ratio']
    crop_latent_for_rgb_size = 60
    # assert crop_latent_for_rgb_size <= 44
    rgb_w = dataset_yaml['rgb_w']
    lpips_w = dataset_yaml['lpips_w']

    iteration = 0
    log_iteration = 0
    
    # 记录判别器需要训练的参数的名字
    dis_trainable_param_names = [
        name for name, param in discriminator.named_parameters() 
        if param.requires_grad
    ]

    for epoch in range(dataset_yaml['max_epochs']):
        for batch_idx, data in enumerate(dataloader, 0):
            accelerator.print(batch_idx, iteration)
            
            """
                一些预处理
            """
            with torch.no_grad():          
                # gt_usm = data['gt'] * 0.5 + 0.5
                # gt_usm = imresize(gt_usm, scale = 2.0).clamp(min = 0, max = 1)
                # gt_usm = ddv2(gt_usm)
                # gt_usm = torch.nn.functional.interpolate(gt_usm, scale_factor=0.5, mode='bicubic', antialias=True, align_corners=False).clamp(min = 0, max = 1)
                # data['gt'] = (gt_usm - 0.5) * 2.0
                
                data['lq'] = data['lq'].to(dtype = torch.bfloat16)
                data['gt'] = data['gt'].to(dtype = torch.bfloat16)
        
                # precess data
                lq_rgb, gt_rgb = pre_process_batch(data, do_crop = True)
            
                lq_latents = generator.module.pipe.encode_image(lq_rgb)
                gt_latents = generator.module.pipe.encode_image(gt_rgb)
            
                caption = data['text'][0]
                # print(caption)
                # get prompt embding using caption
                prompt_emb = generator.module.pipe.encode_prompt(caption, t5_sequence_length=t5_sequence_length)
                # print(prompt_emb['context'].shape, prompt_emb['context'].dtype, prompt_emb['context'].device)
                # torch.Size([1, 512, 4096]) torch.bfloat16 cuda:0
                
            """
                train G
            """
            # D训练的参数 不要产生梯度 否则影响D的梯度累积                
            for name, param in discriminator.named_parameters():
                if name in dis_trainable_param_names:
                    param.requires_grad = False

            sync_gradients = ((batch_idx + 1) % gradient_accumulation_steps == 0) or (batch_idx == len(dataloader) - 1)
            ctx_g = accelerator.no_sync(generator) if not sync_gradients else nullcontext()

            with ctx_g:
                start_point = 900
                random_timestep_id_for_dis = torch.randint(start_point, discriminator.module.pipe.scheduler.num_train_timesteps, (1,))
                random_timestep = discriminator.module.pipe.scheduler.timesteps[random_timestep_id_for_dis].to(device=accelerator.device)
                
                fixed_timestep_id = torch.randint(900, 901, (1,))
                fixed_timestep = generator.module.pipe.scheduler.timesteps[fixed_timestep_id].to(device=accelerator.device)
                one_step_sigma = generator.module.pipe.scheduler.sigmas[fixed_timestep_id].to(dtype=torch.bfloat16, device=accelerator.device)
                noise = torch.randn_like(lq_latents)
                noisy_latents = generator.module.pipe.scheduler.add_noise(lq_latents, noise, fixed_timestep)

                noise_pred = generator(noisy_latents, lq_latents, fixed_timestep, prompt_emb_G).to(dtype=torch.bfloat16)
                training_pred = noisy_latents + (0 - one_step_sigma) * noise_pred

                # select part of model_pred
                # target_training_pred, target_gt_rgb = crop_for_bchw(training_pred, crop_latent_for_rgb_size, gt_rgb)
                training_pred_rgb = generator.module.pipe.vae_decoder(training_pred)
                loss_rgb_mse = torch.nn.functional.mse_loss(training_pred_rgb.float(), gt_rgb.detach().float())
                loss_rgb_mse = rgb_w * loss_rgb_mse
                
                loss_lpips  = net_lpips(training_pred_rgb.float(), gt_rgb.detach().float()).mean()
                loss_lpips = lpips_w * loss_lpips
                
                # noise = torch.randn_like(lq_latents)
                # noisy_latents_fake = discriminator.module.pipe.scheduler.add_noise(training_pred, noise, random_timestep)
                # noisy_latents_fake = training_pred

                with torch.no_grad():
                    real_d_pred = discriminator(gt_latents, random_timestep, prompt_emb).detach().clone()
                fake_g_pred = discriminator(training_pred, random_timestep, prompt_emb)
                tmp1 = cri_gan(real_d_pred - adaptive_relaxed_mean(fake_g_pred, output_size = dataset_yaml['relaxed_mean_size']), False, is_disc=False)
                tmp2 = cri_gan(fake_g_pred - adaptive_relaxed_mean(real_d_pred, output_size = dataset_yaml['relaxed_mean_size']), True, is_disc=False)
                loss_g_gan = (tmp1 + tmp2) / 2
                
                total_loss = (loss_rgb_mse + loss_lpips + loss_g_gan)
                total_loss = total_loss / gradient_accumulation_steps
                accelerator.backward(total_loss)
            
            if sync_gradients:
                optimizer_g.step()
                optimizer_g.zero_grad()
            
            ################################################################################################################################################################
            """
                train D
            """
            for name, param in discriminator.named_parameters():
                if name in dis_trainable_param_names:
                    param.requires_grad = True

            # for name, param in discriminator.named_parameters():
            #     if param.requires_grad and param.grad is not None:
            #         accelerator.print(f"[Warning] after G training: {name} has non-None grad")

            ctx_d = accelerator.no_sync(discriminator) if not sync_gradients else nullcontext()
                
            with ctx_d:
                fake_d_pred = fake_g_pred.detach().clone()
                real_d_pred = discriminator(gt_latents, random_timestep, prompt_emb)
                tmp_3 = cri_gan(real_d_pred - adaptive_relaxed_mean(fake_d_pred, output_size = dataset_yaml['relaxed_mean_size']), True, is_disc=True)
                # cal R1 loss 
                real_d_pred_R1 = discriminator(add_gaussian_noise(gt_latents, variance=dataset_yaml['variance']), random_timestep, prompt_emb)
                tmp_4 = torch.nn.functional.mse_loss(real_d_pred.detach().float(), real_d_pred_R1.float())
                tmp_4 = dataset_yaml['r1_regularization'] * tmp_4
            
                # accelerator.backward(tmp_3 + tmp_4)

                fake_d_pred = discriminator(training_pred.detach(), random_timestep, prompt_emb)
                tmp_5 = cri_gan(fake_d_pred - adaptive_relaxed_mean(real_d_pred.detach(), output_size = dataset_yaml['relaxed_mean_size']), False, is_disc=True)
                # cal R2 loss
                fake_d_pred_R2 = discriminator(add_gaussian_noise(training_pred.detach(), variance=dataset_yaml['variance']), random_timestep, prompt_emb)
                tmp_6 = torch.nn.functional.mse_loss(fake_d_pred.detach().float(), fake_d_pred_R2.float())
                tmp_6 = dataset_yaml['r2_regularization'] * tmp_6
                
                accelerator.backward((tmp_3*0.5 + tmp_4 + tmp_5*0.5 + tmp_6) / gradient_accumulation_steps)
            
            if accelerator.is_main_process:
                with torch.no_grad():
                    print(real_d_pred - adaptive_relaxed_mean(fake_d_pred, output_size = dataset_yaml['relaxed_mean_size']))

            if sync_gradients:
                optimizer_d.step()
                optimizer_d.zero_grad()

            if accelerator.is_main_process:
                trt_logger.log_metrics({
                    "loss_rgb_mse": loss_rgb_mse.item(), 
                    "loss_lpips": loss_lpips.item(),
                    "loss_g_gan_tmp1": tmp1.item(),
                    "loss_g_gan_tmp2": tmp2.item(),
                    "loss_d_real": tmp_3.item(),
                    "loss_d_real_r1": tmp_4.item(),
                    "loss_d_fake": tmp_5.item(),
                    "loss_d_fake_r2": tmp_6.item()
                    }, step = log_iteration)
                log_iteration += 1
            
            if sync_gradients:
                iteration += 1

                if accelerator.is_main_process:
                    if iteration % dataset_yaml['viz_iters'] == 1:
                        print(caption)
                        
                        generator.module.save_to_disk(iteration, lq = lq_rgb, gt = gt_rgb, pred_latent=training_pred,
                                            savedir = workdir)
                
                    if iteration % dataset_yaml['save_ckpt_iters'] == 1:
                        generator.module.save_ckpt(
                            os.path.join(workdir, 'checkpoints'), iter = iteration, tag = "gen"
                        )
                        discriminator.module.save_ckpt(
                            os.path.join(workdir, 'checkpoints'), iter = iteration, tag = "dis"
                        )
            
        

if __name__ == '__main__': 
    args = parse_args()
    if args.task == "data_process":
        raise NotImplementedError("")
    elif args.task == "train":
        train(args)