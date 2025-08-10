import os
import torch
from diffsynth import ModelManager, SD3ImagePipeline
from diffsynth.models.sd3_dit import RMSNorm
from base import BaseModelForT2ILoRA
from einops import rearrange
from tqdm import tqdm
import itertools


class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, dim = 1536, num_heads =24, head_dim = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.query_length = 1
        self.learnable_query = torch.nn.Parameter(torch.randn(1, self.query_length, dim))

        self.to_q = torch.nn.Linear(dim, dim)
        self.to_k = torch.nn.Linear(dim, dim)
        self.to_v = torch.nn.Linear(dim, dim)
        self.out_proj = torch.nn.Linear(dim, dim)
        
        self.pre_norm = RMSNorm(dim, eps=1e-6)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)
        
        self.layernorm = torch.nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        self.ff_a = torch.nn.Sequential(
            torch.nn.Linear(dim, dim*4),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(dim*4, dim)
        )
        
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        bakinput = self.learnable_query.expand(batch_size, -1, -1)
        q = self.to_q(bakinput)
        
        hidden_states = self.pre_norm(hidden_states)
        k = self.to_k(hidden_states) 
        v = self.to_v(hidden_states)
        
        q = q.view(batch_size, self.query_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, self.query_length, self.num_heads * self.head_dim)
        attn_output = attn_output.to(q.dtype)
        attn_output = self.out_proj(attn_output)
        
        res = bakinput + attn_output
        
        res = self.ff_a(self.layernorm(res)) + res
                
        return res


class FinalMLP(torch.nn.Module):
    def __init__(self, dim = 1536):
        super().__init__()
        
        self.layernorm = torch.nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        self.ff_a = torch.nn.Sequential(
            torch.nn.Linear(4*dim, 100),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(100, 1)
        )

    def forward(self, x):
        """
            x: B,4,C
        """
        out = self.layernorm(x).view(x.size(0), -1)
        out = self.ff_a(out)
        return out


class Discriminator(BaseModelForT2ILoRA):
    def __init__(
        self,
        torch_dtype=torch.float16, pretrained_weights=[],
        learning_rate=1e-4, use_gradient_checkpointing=True,
        pretrained_ckpt_path_dis = None
    ):
        super().__init__(learning_rate=learning_rate, use_gradient_checkpointing=use_gradient_checkpointing)
        
        model_manager = ModelManager(torch_dtype=torch_dtype, device='cpu')
        model_manager.load_models(pretrained_weights)
        
        self.pipe = SD3ImagePipeline.from_model_manager(model_manager)
        
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=1.0)

        self.D_last_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0),
            torch.nn.PReLU(num_parameters=32),
            torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        )


        self.pipe.requires_grad_(False) # vae,clip,t5 to requires_grad_ False 
        self.pipe.eval()
        self.pipe.denoising_model().train()
        
        
        # self.cross_attention_blocks = torch.nn.ModuleList(
        #     [CrossAttentionBlock() for i in range(4)]
        # )
        # self.final_mlp = FinalMLP()
        
        self.pipe.denoising_model().requires_grad_(True)
        self.pipe.denoising_model().pos_embedder.requires_grad_(False)
        # self.cross_attention_blocks.requires_grad_(True)
        # self.final_mlp.requires_grad_(True)
        self.D_last_conv.requires_grad_(True)

        if pretrained_ckpt_path_dis:
            state_dict = torch.load(pretrained_ckpt_path_dis, map_location='cpu')
            self.load_state_dict(state_dict, strict = False)
        
    def configure_optimizers(self):
        param_groups = []

        denoising_params = [
            p for p in self.pipe.denoising_model().parameters() 
            if p.requires_grad
        ]
        param_groups.append({
            'params': denoising_params,
            'lr': self.learning_rate
        })
        
        d_last_conv_params = [
            p for p in self.D_last_conv.parameters() 
            if p.requires_grad
        ]
        param_groups.append({
            'params': d_last_conv_params,
            'lr': self.learning_rate * 1.0
        })
        
        opt = torch.optim.Adam(
            param_groups,
            lr=self.learning_rate,  # 注意：这里的基础学习率将被参数组覆盖
            betas=(0.0, 0.9)
        )
        return opt
    
    def forward(self, noisy_latents, timestep, prompt_emb):
        out, list_of_b1c, conditioning = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        assert len(out.shape) == 4 and out.shape[0] == 1
        
        # res = []
        # for idx, item in enumerate(list_of_b1c):
        #     res.append(self.cross_attention_blocks[idx](item))
        # out = torch.cat(res, dim=1) + 0 * out[:, 0, 0:1, 0:1] # b4c
        # out = self.final_mlp(out)
        
        out = self.D_last_conv(out)
        return out