import os
import torch
from base import BaseModelForT2ILoRA
from einops import rearrange, repeat
from tqdm import tqdm
import itertools
from diffsynth.extensions.flux.util import load_flow_model2
from diffsynth.extensions.flux.sampling import unpack
from omegaconf import OmegaConf



class Discriminator(BaseModelForT2ILoRA):
    def __init__(
        self,
        torch_dtype=torch.float16,
        learning_rate=1e-4, use_gradient_checkpointing=True
    ):
        super().__init__(learning_rate=learning_rate, use_gradient_checkpointing=use_gradient_checkpointing)
        
        args = OmegaConf.load('/mnt/media01/dataset/media_algo_share/lanjinghong/lanjinghong/project/diffsynth-studio/examples/flux_gan/fluxmini.yaml')
        
        self.dit = load_flow_model2(args.model_name, args, device='cpu')
        self.dit = self.dit.to(dtype=torch_dtype)
        
        self.D_last_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0),
            torch.nn.PReLU(num_parameters=32),
            torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        self.dit.train()
        self.dit.requires_grad_(True)
        self.D_last_conv.requires_grad_(True)
        
        
    def configure_optimizers(self):
        trainable_modules = [
            p for model in [
                self.dit,
                self.D_last_conv,
            ]
            for p in model.parameters() if p.requires_grad
        ]

        # opt = torch.optim.Adam(trainable_modules, lr=self.learning_rate, betas=(0.0, 0.9))
        opt = torch.optim.RMSprop(
            trainable_modules,
            lr=self.learning_rate,  # 保持相同学习率
            alpha=0.9,              # beta2=0.9 → alpha=0.9
            eps=1e-8,               # Adam 默认 eps=1e-8，RMSprop 也需一致
            momentum=0,             # beta1=0.0 → 无动量
            weight_decay=0,         # 默认无 L2 正则化
            centered=False          # 默认不中心化
        )
        return opt
    
    def forward(self, noisy_latents, timestep, prompt_emb, return_score = True):
        batch,_,h,w = noisy_latents.shape
        assert batch == 1
        guidance = 3.5
        guidance_vec = torch.full((batch,), guidance, device=noisy_latents.device, dtype=noisy_latents.dtype)
        t_vec = timestep / 1000.0
        t_vec = t_vec.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        assert len(timestep.shape) == 1
        
        noisy_latents = rearrange(noisy_latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch)
        img_ids = img_ids.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        
        txt_ids = torch.zeros(batch, prompt_emb['t5'].shape[1], 3)
        txt_ids = txt_ids.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        
        out, intermediate_double, intermediate_single  = self.dit(
            img       = noisy_latents,
            img_ids   = img_ids,
            txt       = prompt_emb['t5'],
            txt_ids   = txt_ids,
            y         = prompt_emb['clip'],
            timesteps = t_vec,
            guidance  = guidance_vec,
            return_intermediate = True
        )        
        out = unpack(out, h *8, w * 8)
        
        intermediates = []
        selected_double_index = (1,4)
        selected_single_index = (4,9)
        
        assert len(intermediate_double) == 5
        for i in range(5):
            if i in selected_double_index:
                intermediates = intermediates + intermediate_double[i]
                
        assert len(intermediate_single) == 10
        for i in range(10):
            if i in selected_single_index:
                intermediates = intermediates + intermediate_single[i]
        
        if return_score:
            return self.D_last_conv(out), intermediates
        else:
            return out, intermediates