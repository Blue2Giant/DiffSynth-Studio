"""
生成器不用prompt
判别器概率使用prompt
prompt来自qwen

单步 gan训练
"""
import os
import torch
from datetime import datetime
import random
from contextlib import redirect_stdout
from torchvision import transforms
from utils import yaml_load, parse_args

from ganloss import GANLoss
from generator import Generator
from discriminator import Discriminator
from discriminator2 import UNetDiscriminatorSN

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
from PIL import Image

def save_heatmap(tensor1, output_path1, tensor2, output_path2):
    heatmap1 = tensor1.squeeze().detach().cpu().numpy()
    heatmap2 = tensor2.squeeze().detach().cpu().numpy()
    
    global_min = min(heatmap1.min(), heatmap2.min())
    global_max = max(heatmap1.max(), heatmap2.max())
    
    def normalize(x):
        return (x - global_min) / (global_max - global_min + 1e-8)
    
    heatmap1_norm = normalize(heatmap1)
    heatmap2_norm = normalize(heatmap2)
    heatmap1_uint8 = (heatmap1_norm * 255).astype(np.uint8)
    heatmap2_uint8 = (heatmap2_norm * 255).astype(np.uint8)
    Image.fromarray(heatmap1_uint8, mode='L').save(output_path1)
    Image.fromarray(heatmap2_uint8, mode='L').save(output_path2)
    
def adaptive_relaxed_mean(tensor, output_size=20):
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

def get_dis_feature_loss(pred_features, gt_features):
    loss = 0.0
    num_features = 0
    for pred, gt in zip(pred_features, gt_features):
        gt = gt.detach()
        loss += torch.nn.functional.l1_loss(pred, gt)
        num_features += 1
    if num_features > 0:
        loss = loss / num_features
    return loss

@torch.no_grad()
def pre_process_batch(batch, do_crop):
    assert batch['lq'].shape[0] == 1
    
    lq = batch['lq']
    gt = batch['gt']

    assert lq.shape[2] == gt.shape[2] and lq.shape[3] == gt.shape[3] 
    
    if do_crop:
        H, W = lq.shape[2], lq.shape[3]
        
        crop_H = 800 # random.randint(32, 50) * 16 # 512 ~ 800
        crop_W = 800 # random.randint(32, 50) * 16
    
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

    accelerator = Accelerator(
        gradient_accumulation_steps=dataset_yaml['accumulate_grad_batches'],
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

    generator = Generator(
        torch_dtype = torch.bfloat16,
        learning_rate=dataset_yaml['learning_rate'],
        use_gradient_checkpointing=dataset_yaml['use_gradient_checkpointing'],
        pretrained_ckpt_path_gen = dataset_yaml['pretrained_ckpt_path_gen']
    )
    
    discriminator = Discriminator(
        torch_dtype = torch.bfloat16,
        learning_rate=dataset_yaml['learning_rate_dis'],
        use_gradient_checkpointing=dataset_yaml['use_gradient_checkpointing']
    )
    
    discriminator2 = UNetDiscriminatorSN(
        num_in_ch = 3
    )
    
    cri_gan = GANLoss(gan_type = dataset_yaml['gan_type'], loss_weight = dataset_yaml['gan_loss_weight'], 
                      real_label_val=dataset_yaml['real_label_val'], fake_label_val = dataset_yaml['fake_label_val'])
    cri_gan2 = GANLoss(gan_type = 'vanilla', loss_weight = dataset_yaml['gan_loss_weight2'])
    
    optimizer_g = generator.configure_optimizers()
    optimizer_d = discriminator.configure_optimizers()
    optimizer_d2 = discriminator2.configure_optimizers(learning_rate = 1e-4)
    
    generator.device = accelerator.device
    discriminator.device = accelerator.device
    
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
    
    generator, discriminator, discriminator2, net_lpips, optimizer_g, optimizer_d, optimizer_d2, dataloader = accelerator.prepare(
        generator, discriminator, discriminator2, net_lpips, optimizer_g, optimizer_d, optimizer_d2, dataloader
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

        # plot model structure
        filename = os.path.join(workdir, 'generator_model_structure.txt')
        with open(filename, "a") as f:
            with redirect_stdout(f):
                print(generator.module.dit)
                
        filename = os.path.join(workdir, 'discriminator_model_structure.txt')
        with open(filename, "a") as f:
            with redirect_stdout(f):
                print(discriminator.module.dit)
        
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
        
        
    
    neg_caption = "" # "Bright tones, overexposed, blurred details, subtitles, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, walking backwards"
    # neg_prompt_emb = generator.pipe_G.encode_prompt(neg_caption, positive=False)
    prompt_G = """Enhance the image while preserving as much of the original details as possible, including common textures such as faces, hands, trees, flowers, buildings, water surfaces, fabrics, stone, metal, and foliage.
                                                       realistic maximum detail, ultra HD, skin pore detailing, perfect without deformations."""
    
    prompt_emb_G = {}
    with torch.no_grad():
        prompt_emb_G['t5'] = generator.module.t5(prompt_G)
        prompt_emb_G['clip'] = generator.module.clip(prompt_G)
    # null_text_ratio = dataset_yaml['null_text_ratio']
    crop_latent_for_rgb_size = 60
    # assert crop_latent_for_rgb_size <= 44
    rgb_w = dataset_yaml['rgb_w']
    lpips_w = dataset_yaml['lpips_w']

    iteration = 0
    
    for epoch in range(dataset_yaml['max_epochs']):
        for batch_idx, data in enumerate(dataloader, 0):
            accelerator.print(batch_idx, iteration)
            l_acc = [generator, discriminator, discriminator2]
            
            with torch.no_grad():          
                gt_usm = data['gt'] * 0.5 + 0.5
                gt_usm = imresize(gt_usm, scale = 2.0).clamp(min = 0, max = 1)
                gt_usm = ddv2(gt_usm)
                gt_usm = torch.nn.functional.interpolate(gt_usm, scale_factor=0.5, mode='bicubic', antialias=True, align_corners=False).clamp(min = 0, max = 1)
                data['gt'] = (gt_usm - 0.5) * 2.0
                
                data['lq'] = data['lq'].to(dtype = torch.bfloat16)
                data['gt'] = data['gt'].to(dtype = torch.bfloat16)
        
                # precess data
                lq_rgb, gt_rgb = pre_process_batch(data, do_crop = True)
            
                lq_latents = generator.module.vae.encode(lq_rgb)
                gt_latents = generator.module.vae.encode(gt_rgb)
        
                caption = data['text'][0]
                # print(caption)
                # get prompt embding using caption
                prompt_emb = {}
                prompt_emb['t5'] = generator.module.t5(caption)
                prompt_emb['clip'] = generator.module.clip(caption)


            with accelerator.accumulate(*l_acc):
                for p in discriminator.parameters():
                    p.requires_grad = False
                for p in discriminator2.parameters():
                    p.requires_grad = False
                
                # start_point = get_start_timestep_given_iteration(iteration)
                random_timestep = 100 * torch.rand(1).to(device=accelerator.device, dtype = torch.bfloat16)
                fixed_timestep = 0.2 * torch.ones(1).to(device=accelerator.device, dtype = torch.bfloat16)
                
                noise = torch.randn_like(lq_latents)
                noisy_latents = (1 - fixed_timestep) * lq_latents + fixed_timestep * noise

                noise_pred = generator(noisy_latents, lq_latents, (fixed_timestep+0.4) * 1000, prompt_emb_G).to(dtype=torch.bfloat16)
                training_pred = noisy_latents  - (fixed_timestep+0.4) * noise_pred

                # select part of model_pred
                # target_training_pred, target_gt_rgb = crop_for_bchw(training_pred, crop_latent_for_rgb_size, gt_rgb)
                training_pred_rgb = generator.module.vae.decode(training_pred)
                loss_rgb_mse = torch.nn.functional.mse_loss(training_pred_rgb.float(), gt_rgb.detach().float())
                loss_rgb_mse = rgb_w * loss_rgb_mse
                
                # noise = torch.randn_like(lq_latents)
                # noisy_latents_fake = discriminator.module.pipe.scheduler.add_noise(training_pred, noise, random_timestep)
                # noisy_latents_fake = training_pred
                real_d_pred, list_of_real_feature = discriminator(gt_latents, random_timestep, prompt_emb)
                real_d_pred = real_d_pred.detach()
                fake_g_pred, list_of_fake_feature = discriminator(training_pred, random_timestep, prompt_emb)
                
                # loss_lpips = get_dis_feature_loss(list_of_fake_feature, list_of_real_feature) # order: pred, gt. can't swap!
                loss_lpips  = net_lpips(training_pred_rgb.float(), gt_rgb.detach().float()).mean()
                loss_lpips = lpips_w * loss_lpips
                
                tmp1 = cri_gan(real_d_pred - adaptive_relaxed_mean(fake_g_pred), False, is_disc=False)
                tmp2 = cri_gan(fake_g_pred - adaptive_relaxed_mean(real_d_pred), True, is_disc=False)
                loss_g_gan = (tmp1 + tmp2) / 2
                
                fake_g_pred_2 = discriminator2(training_pred_rgb.float())
                loss_g_gan2 = cri_gan2(fake_g_pred_2, True, is_disc=False)
                
                total_loss = (loss_rgb_mse + loss_lpips + loss_g_gan + loss_g_gan2)
                
                accelerator.backward(total_loss, retain_graph=False)
                optimizer_g.step()
                optimizer_g.zero_grad()

                ################################################################################################################################################################
                # optimizer_d.zero_grad() 前面禁止了D的梯度 所以训练G不产生梯度 可以注释（梯度累积必须注释，否则有计算错误)

                # set d required grad to true
                for p in discriminator.parameters():
                    p.requires_grad = True

                # for name, param in discriminator.named_parameters():
                #     if param.requires_grad and param.grad is not None:
                #         accelerator.print(f"[Warning] after G training: {name} has non-None grad")

                # noisy_latents_real = discriminator.module.pipe.scheduler.add_noise(gt_latents, noise, random_timestep)
                # noisy_latents_real = gt_latents
                for _ in range(dataset_yaml['d_iters']):
                    fake_d_pred = fake_g_pred.detach().clone()
                    real_d_pred, _ = discriminator(gt_latents, random_timestep, prompt_emb)
                    tmp_3 = cri_gan(real_d_pred - adaptive_relaxed_mean(fake_d_pred), True, is_disc=True) 
                    # cal R1 loss 
                    real_d_pred_R1, _ = discriminator(add_gaussian_noise(gt_latents, variance=dataset_yaml['variance']), random_timestep, prompt_emb)
                    tmp_4 = torch.nn.functional.mse_loss(real_d_pred.float(), real_d_pred_R1.float())
                    tmp_4 = dataset_yaml['r1_regularization'] * tmp_4
                
                    # accelerator.backward(tmp_3 + tmp_4)

                    fake_d_pred, _ = discriminator(training_pred.detach(), random_timestep, prompt_emb)
                    tmp_5 = cri_gan(fake_d_pred - adaptive_relaxed_mean(real_d_pred.detach()), False, is_disc=True)
                    # cal R2 loss
                    # fake_d_pred_R2 = discriminator(lq_latents, add_gaussian_noise(training_pred.detach(), variance=dataset_yaml['variance']), random_timestep, prompt_emb)
                    # tmp_6 = torch.nn.functional.mse_loss(fake_d_pred.float(), fake_d_pred_R2.float())
                    # tmp_6 = dataset_yaml['r2_regularization'] * tmp_6
                    
                    accelerator.backward(tmp_3 + tmp_4 + tmp_5)
                    optimizer_d.step()
                    optimizer_d.zero_grad()


                # set d2 required grad to true
                for p in discriminator2.parameters():
                    p.requires_grad = True
                
                for _ in range(dataset_yaml['d_iters']):
                    # real
                    real_d_pred_2 = discriminator2(gt_rgb.detach().float())
                    tmp_7 = cri_gan2(real_d_pred_2, True, is_disc=True)
                    
                    # fake
                    fake_d_pred_2 = discriminator2(training_pred_rgb.detach().float().clone())  # clone for pt1.9
                    tmp_8 = cri_gan2(fake_d_pred_2, False, is_disc=True)
                    accelerator.backward(tmp_7 + tmp_8)
                    optimizer_d2.step()
                    optimizer_d2.zero_grad()
                    
            if accelerator.sync_gradients:
                iteration += 1

                if accelerator.is_main_process:
                    trt_logger.log_metrics({
                                        "loss_rgb_mse": loss_rgb_mse.item(), 
                                        "loss_lpips": loss_lpips.item(),
                                        "loss_g_gan": loss_g_gan.item(),
                                        "loss_g_gan2": loss_g_gan2.item(),
                                        "loss_d_real": tmp_3.item(),
                                        "loss_d_real_r1": tmp_4.item(),
                                        "loss_d_fake": tmp_5.item(),
                                        "loss_d2_real": tmp_7.item(),
                                        "loss_d2_fake": tmp_8.item(),
                                        # "loss_d_fake_r2": tmp_6.item()
                                        }, step = iteration)
         
                    if iteration % 25 == 1:
                        print(caption)
                        save_heatmap(real_d_pred, os.path.join(workdir, 'iter_{}_real_d_score.png'.format(iteration)),
                                     fake_d_pred, os.path.join(workdir, 'iter_{}_fake_d_score.png'.format(iteration)))
                        generator.module.save_to_disk(iteration, lq = lq_rgb, gt = gt_rgb, pred_latent=training_pred,
                                            savedir = workdir)
                
                    if iteration % 3000 == 1:
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