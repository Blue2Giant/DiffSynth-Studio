import os
import torch
import torchvision
from torchvision.transforms import v2
from torch.nn import functional as F
from copy import deepcopy
from diffsynth import ModelManager, SD3ImagePipeline
from base import BaseModelForT2ILoRA
from einops import rearrange
from tqdm import tqdm
import random


class Generator(BaseModelForT2ILoRA):
    def __init__(
        self,
        torch_dtype=torch.float16, pretrained_weights=[],
        learning_rate=1e-4, use_gradient_checkpointing=True,
        pretrained_ckpt_path_gen = None
    ):
        super().__init__(learning_rate=learning_rate, use_gradient_checkpointing=use_gradient_checkpointing)
        
        model_manager = ModelManager(torch_dtype=torch_dtype, device='cpu')
        model_manager.load_models(pretrained_weights)
        
        self.pipe = SD3ImagePipeline.from_model_manager(model_manager)
        
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=3.0)

        self.operate_parameters()
        
        if pretrained_ckpt_path_gen:
            state_dict = torch.load(pretrained_ckpt_path_gen, map_location='cpu')
            self.load_state_dict(state_dict, strict = False)
        
    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        # opt = torch.optim.AdamW(trainable_modules, lr=self.learning_rate, betas=(0.9, 0.95), weight_decay=0)
        opt = torch.optim.Adam(trainable_modules, lr=self.learning_rate, betas=(0.0, 0.9))
        return opt
    
    def forward(self, noisy_latents, condition_latent, timestep, prompt_emb):
        out = self.pipe.denoising_model()(
            noisy_latents, condition_latent, timestep=timestep, **prompt_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        return out

    @torch.no_grad()
    def save_to_disk(self, iter, lq, gt, pred_latent, savedir):
        pred_latent = pred_latent.to(dtype = self.pipe.torch_dtype)
        
        lq = lq.cpu().float() # -1~1
        gt = gt.cpu().float()
        pred_rgb = self.pipe.vae_decoder(pred_latent)
        pred_rgb = pred_rgb.cpu().float()
        
        grid = torchvision.utils.make_grid(lq, nrow=4, normalize=True, value_range=(-1, 1))
        grid_image = v2.ToPILImage()(grid)
        grid_image.save(os.path.join(savedir, 'iter_{}_lq.png'.format(iter)))
    
        grid = torchvision.utils.make_grid(gt, nrow=4, normalize=True, value_range=(-1, 1))
        grid_image = v2.ToPILImage()(grid)
        grid_image.save(os.path.join(savedir, 'iter_{}_gt.png'.format(iter)))
    
        grid = torchvision.utils.make_grid(pred_rgb, nrow=4, normalize=True, value_range=(-1, 1))
        grid_image = v2.ToPILImage()(grid)
        grid_image.save(os.path.join(savedir, 'iter_{}_pred.png'.format(iter)))
    
    @torch.no_grad()
    def infer(self, prompt,
              lq_latent,
              tiled,
              tile_size,
              tile_stride):
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
        prompt_emb_posi = self.pipe.encode_prompt(prompt, positive=True, t5_sequence_length=77)
        
        fixed_timestep_id = torch.randint(900, 901, (1,))
        fixed_timestep = self.pipe.scheduler.timesteps[fixed_timestep_id].to(device=self.device)
        one_step_sigma = self.pipe.scheduler.sigmas[fixed_timestep_id].to(dtype=torch.bfloat16, device=self.device)
        noise = torch.randn_like(lq_latent)
        noisy_latents = self.pipe.scheduler.add_noise(lq_latent, noise, fixed_timestep)
        
        pred = self.pipe.dit(noisy_latents, lq_latent, timestep=fixed_timestep, **prompt_emb_posi, **tiler_kwargs)
        
        pred = noisy_latents + (0 - one_step_sigma) * pred

        image = self.pipe.decode_image(pred, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        
        return image