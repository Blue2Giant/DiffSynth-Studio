import os
import torch
import torchvision
from torchvision.transforms import v2
from base import BaseModelForT2ILoRA
from einops import rearrange, repeat
from diffsynth.extensions.flux.util import (load_ae, load_clip,
                       load_flow_model, load_t5)
from diffsynth.extensions.flux.sampling import unpack
from peft import LoraConfig, inject_adapter_in_model


def get_models(name: str, device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae, t5, clip

class Generator(BaseModelForT2ILoRA):
    def __init__(
        self,
        torch_dtype=torch.float16,
        learning_rate=1e-4, use_gradient_checkpointing=True,
        pretrained_ckpt_path_gen = None
    ):
        super().__init__(learning_rate=learning_rate, use_gradient_checkpointing=use_gradient_checkpointing)
        
        # load t5 clip vae and dit
        """
            t5 bf16
            clip bf16
            vae bf16
            dit bf16
        """
        self.dit, self.vae, self.t5, self.clip = get_models(name='flux-dev', device='cpu', offload=False, is_schnell=False)
        self.dit = self.dit.to(dtype=torch_dtype)
        self.vae = self.vae.to(dtype=torch_dtype)
        
        self.operate_parameters()
        
        if pretrained_ckpt_path_gen:
            state_dict = torch.load(pretrained_ckpt_path_gen)
            self.load_state_dict(state_dict, strict = False)
        
    def operate_parameters(self):
        self.requires_grad_(False) # all requires_grad False 
        self.eval()
        self.dit.train()
        
        trainable_modules = [
            self.dit.final_layer,
            self.dit.time_in,
            self.dit.guidance_in,
            self.dit.img_in
        ]

        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True
        
        original_trainable_params = {
            name for name, param in self.dit.named_parameters() if param.requires_grad
        }
        
        lora_target_modules = "img_mod.lin,qkv,proj,img_mlp.0,img_mlp.2,txt_mod.lin,txt_mlp.0,txt_mlp.2,linear1,linear2,modulation.lin"
        
        lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=lora_target_modules.split(",")
        )
        inject_adapter_in_model(lora_config, self.dit)
        
        for name, param in self.dit.named_parameters():
            if name in original_trainable_params:
                param.requires_grad = True
        
        # for param in self.dit.parameters():
        #     # Upcast LoRA parameters into fp32
        #     if param.requires_grad:
        #         param.data = param.to(torch.float32)
                
    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.dit.parameters())
        # opt = torch.optim.AdamW(trainable_modules, lr=self.learning_rate, betas=(0.9, 0.95), weight_decay=0)
        opt = torch.optim.Adam(trainable_modules, lr=self.learning_rate, betas=(0.0, 0.9))
        return opt
    
    def forward(self, noisy_latents, condition_latent, timestep, prompt_emb):
        """
            noisy_latents: 1,16,h,w
            condition_latent: 1,16,h,w
            prompt_emb: include t5 res and clip res
        """
        batch,_,h,w = noisy_latents.shape
        assert batch == 1
        guidance = 3.5
        guidance_vec = torch.full((batch,), guidance, device=noisy_latents.device, dtype=noisy_latents.dtype)
        t_vec = timestep / 1000.0
        t_vec = t_vec.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        assert len(timestep.shape) == 1
        
        noisy_latents = rearrange(noisy_latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if condition_latent is not None:
            condition_latent = rearrange(condition_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            noisy_latents = torch.concat([noisy_latents, condition_latent], dim = 1) # b 2l c
        
        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch)
        
        _,res_len,_ = img_ids.shape
        # print(res_len)
        
        if condition_latent is not None:
            condition_ids = img_ids.clone()
            condition_ids[..., 0] = 1.0
            img_ids = torch.concat([img_ids, condition_ids], dim = 1) # b 2l c
        img_ids = img_ids.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        
        txt_ids = torch.zeros(batch, prompt_emb['t5'].shape[1], 3)
        txt_ids = txt_ids.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        
        out = self.dit(
            img       = noisy_latents,
            img_ids   = img_ids,
            txt       = prompt_emb['t5'],
            txt_ids   = txt_ids,
            y         = prompt_emb['clip'],
            timesteps = t_vec,
            guidance  = guidance_vec,
        )        
        # only use the pre half for the res
        out = out[:, 0:res_len, :]
        out = unpack(out, h *8, w * 8)
        return out

    @torch.no_grad()
    def save_to_disk(self, iter, lq, gt, pred_latent, savedir):
        
        lq = lq.cpu().float() # -1~1
        gt = gt.cpu().float()
        pred_rgb = self.vae.decode(pred_latent)
        pred_rgb = pred_rgb.clamp(-1, 1)
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