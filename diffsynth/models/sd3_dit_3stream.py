"""
    重新实现SD3DiT

    基于DiT4SR: Taming Diffusion Transformer for Real-World Image Super-Resolution
    训练多步 sd3.5
    
    降质： 使用basicsr 2x超分， 推理512->1024
    
    主要两方面改动：
    1.LR Integration in Attention
        新增一个lr流，新增的参数有：
        ▪ AdaLayerNorm (dual 同noise分支)                (copy from stream_a) 可训练

        ▪ JointAttention中新增qkv                       （zero初始化）可训练

        ▪ JointAttention中新增linear                     (copy from stream_a) 可训练

        ▪ MLP                                            copy from stream_a  局部训练

    2.LR Injection between MLP
        ▪ 新增一个3*3  分组卷积                             (zero初始化)  可训练

    
    旧的参数不训练
"""
import torch
from einops import rearrange
from .tiler import TileWorker
from .sd3_dit import RMSNorm, PatchEmbed, TimestepEmbeddings, AdaLayerNorm, SingleAttention
from functools import partial




class JointAttention_3_stream(torch.nn.Module):
    def __init__(self, dim_a, dim_b, num_heads, head_dim, only_out_a=False, use_rms_norm=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.only_out_a = only_out_a

        self.a_to_qkv = torch.nn.Linear(dim_a, dim_a * 3)
        self.b_to_qkv = torch.nn.Linear(dim_b, dim_b * 3)
        # self.c_to_qkv = torch.nn.Linear(dim_a, dim_a * 3)
        self.c_to_q = torch.nn.Linear(dim_a, dim_a)
        self.c_to_k = torch.nn.Linear(dim_a, dim_a)
        self.c_to_v = torch.nn.Linear(dim_a, dim_a)

        self.a_to_out = torch.nn.Linear(dim_a, dim_a)
        if not only_out_a:
            self.b_to_out = torch.nn.Linear(dim_b, dim_b)
            self.c_to_out = torch.nn.Linear(dim_a, dim_a)
            self.use_LR_Residual = True

        if use_rms_norm:
            self.norm_q_a = RMSNorm(head_dim, eps=1e-6)
            self.norm_k_a = RMSNorm(head_dim, eps=1e-6)
            self.norm_q_b = RMSNorm(head_dim, eps=1e-6)
            self.norm_k_b = RMSNorm(head_dim, eps=1e-6)
            self.norm_q_c = RMSNorm(head_dim, eps=1e-6)
            self.norm_k_c = RMSNorm(head_dim, eps=1e-6)
        else:
            self.norm_q_a = None
            self.norm_k_a = None
            self.norm_q_b = None
            self.norm_k_b = None
            self.norm_q_c = None
            self.norm_k_c = None

        

    def process_qkv(self, hidden_states, to_qkv, norm_q, norm_k, is_control = False):
        batch_size = hidden_states.shape[0]
        qkv = to_qkv(hidden_states)
        
        if is_control:
            q, k, v = qkv.chunk(3, dim=-1)
            q = self.c_to_q(q)
            k = self.c_to_k(k)
            v = self.c_to_v(v)
            q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            qkv = qkv.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
            q, k, v = qkv.chunk(3, dim=1)
        
        if norm_q is not None:
            q = norm_q(q)
        if norm_k is not None:
            k = norm_k(k)
        return q, k, v

    def generate_mask(self, l1, l2, l3, device, dtype):
        # mask = torch.zeros((1, 1, l1 + l2 + l3, l1 + l2 + l3), device=device, dtype=dtype)
        # seq1_start, seq1_end = 0, l1
        # seq2_start, seq2_end = l1, l1 + l2
        # seq3_start, seq3_end = l1 + l2, l1 + l2 + l3

        # mask[:, :, seq1_start:seq1_end, seq3_start:seq3_end] = -100
        # mask[:, :, seq2_start:seq2_end, seq3_start:seq3_end] = -100
        # mask[:, :, seq3_start:seq3_end, seq1_start:seq1_end] = -100
        
        # return mask
        return None

    def forward(self, hidden_states_a, hidden_states_b, hidden_states_c):
        batch_size = hidden_states_a.shape[0]
        la = hidden_states_a.shape[1]
        lb = hidden_states_b.shape[1]
        lc = hidden_states_c.shape[1]
        assert la == lc

        backup_hidden_states_c = hidden_states_c

        qa, ka, va = self.process_qkv(hidden_states_a, self.a_to_qkv, self.norm_q_a, self.norm_k_a)
        qb, kb, vb = self.process_qkv(hidden_states_b, self.b_to_qkv, self.norm_q_b, self.norm_k_b)
        qc, kc, vc = self.process_qkv(hidden_states_c, self.a_to_qkv, self.norm_q_c, self.norm_k_c, is_control=True)
        
        q = torch.concat([qa, qb, qc], dim=2) # B, num_heads, L, head_dim
        k = torch.concat([ka, kb, kc], dim=2)
        v = torch.concat([va, vb, vc], dim=2)

        attention_mask = self.generate_mask(la, lb, lc, q.device, q.dtype)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)


        hidden_states_a, hidden_states_b, hidden_states_c = hidden_states[:, :la], hidden_states[:, la:la+lb], hidden_states[:, la+lb:]
        hidden_states_a = self.a_to_out(hidden_states_a)
        if self.only_out_a:
            return hidden_states_a
        else:
            hidden_states_b = self.b_to_out(hidden_states_b)
            if self.use_LR_Residual:
                hidden_states_c = self.c_to_out(hidden_states_c + backup_hidden_states_c)
            else:
                hidden_states_c = self.c_to_out(hidden_states_c)
            return hidden_states_a, hidden_states_b, hidden_states_c
        









class JointTransformerBlock_3_stream(torch.nn.Module):
    def __init__(self, dim, num_attention_heads, use_rms_norm=False, dual=False):
        super().__init__()
        self.norm1_a = AdaLayerNorm(dim, dual=dual)
        self.norm1_b = AdaLayerNorm(dim)
        self.norm1_c = AdaLayerNorm(dim, dual=dual)

        self.attn = JointAttention_3_stream(dim, dim, num_attention_heads, dim // num_attention_heads, use_rms_norm=use_rms_norm)
        if dual:
            self.attn2 = SingleAttention(dim, num_attention_heads, dim // num_attention_heads, use_rms_norm=use_rms_norm)

        self.norm2_a = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_a = torch.nn.Sequential(
            torch.nn.Linear(dim, dim*4),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(dim*4, dim)
        )

        self.norm2_b = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_b = torch.nn.Sequential(
            torch.nn.Linear(dim, dim*4),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(dim*4, dim)
        )
        
        self.norm2_c = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_c = torch.nn.Sequential(
            torch.nn.Linear(dim, dim*4),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(dim*4, dim)
        )
        
        self.depthwise_conv = torch.nn.Conv2d(
            in_channels=dim * 4,
            out_channels=dim * 4,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=dim//2, # 3*3*8
            bias=True
        )

    def ff_c_middle_res(self, norm_hidden_states_c):
        """
            use self.height_, self.width_, self.depthwise_conv, self.ff_c
        """
        b, l, c = norm_hidden_states_c.shape
        assert l == self.height_ * self.width_, "输入张量的长度 l 必须等于 height_ 和 width_ 的乘积"

        # 获取第一个Linear层的输出
        linear_output = self.ff_c[0](norm_hidden_states_c)
        linear_output = self.ff_c[1](linear_output)
        conv_output = linear_output.reshape(b, self.height_, self.width_, linear_output.shape[2]) 
        conv_output = conv_output.permute(0, 3, 1, 2)
        conv_output = self.depthwise_conv(conv_output).permute(0, 2, 3, 1) # b,h,w,c
        conv_output = conv_output.reshape(b, self.height_ * self.width_, conv_output.shape[3])
        
        # 继续完成原始前向传播
        final_output = self.ff_c[2](linear_output)
        
        return conv_output, final_output

    def ff_a_inject(self, norm_hidden_states_a, mid_res):
        """
            use self.ff_a
        """
        final_output = self.ff_a[0](norm_hidden_states_a)
        final_output = self.ff_a[1](final_output)
        final_output = self.ff_a[2](final_output + mid_res)
        return final_output

    def forward(self, hidden_states_a, hidden_states_b, hidden_states_c, temb):
        if self.norm1_a.dual:
            norm_hidden_states_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a, norm_hidden_states_a_2, gate_msa_a_2 = self.norm1_a(hidden_states_a, emb=temb)
            norm_hidden_states_c, gate_msa_c, shift_mlp_c, scale_mlp_c, gate_mlp_c, norm_hidden_states_c_2, gate_msa_c_2 = self.norm1_c(hidden_states_c, emb=temb)
        else:
            norm_hidden_states_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a = self.norm1_a(hidden_states_a, emb=temb)
            norm_hidden_states_c, gate_msa_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = self.norm1_c(hidden_states_c, emb=temb)
        norm_hidden_states_b, gate_msa_b, shift_mlp_b, scale_mlp_b, gate_mlp_b = self.norm1_b(hidden_states_b, emb=temb)

        # Attention
        attn_output_a, attn_output_b, attn_output_c = self.attn(norm_hidden_states_a, norm_hidden_states_b, norm_hidden_states_c)

        # Part C
        hidden_states_c = hidden_states_c + gate_msa_c * attn_output_c
        norm_hidden_states_c = self.norm2_c(hidden_states_c) * (1 + scale_mlp_c) + shift_mlp_c
        mid_res, origin_res = self.ff_c_middle_res(norm_hidden_states_c)
        hidden_states_c = hidden_states_c + gate_mlp_c * origin_res

        # Part A
        hidden_states_a = hidden_states_a + gate_msa_a * attn_output_a
        if self.norm1_a.dual:
            hidden_states_a = hidden_states_a + gate_msa_a_2 * self.attn2(norm_hidden_states_a_2)
        norm_hidden_states_a = self.norm2_a(hidden_states_a) * (1 + scale_mlp_a) + shift_mlp_a
        hidden_states_a = hidden_states_a + gate_mlp_a * self.ff_a_inject(norm_hidden_states_a, mid_res)

        # Part B
        hidden_states_b = hidden_states_b + gate_msa_b * attn_output_b
        norm_hidden_states_b = self.norm2_b(hidden_states_b) * (1 + scale_mlp_b) + shift_mlp_b
        hidden_states_b = hidden_states_b + gate_mlp_b * self.ff_b(norm_hidden_states_b)

        return hidden_states_a, hidden_states_b, hidden_states_c



class JointTransformerFinalBlock_3_stream(torch.nn.Module):
    def __init__(self, dim, num_attention_heads, use_rms_norm=False):
        super().__init__()
        self.norm1_a = AdaLayerNorm(dim)
        self.norm1_b = AdaLayerNorm(dim, single=True)
        self.norm1_c = AdaLayerNorm(dim)

        self.attn = JointAttention_3_stream(dim, dim, num_attention_heads, dim // num_attention_heads, only_out_a=True, use_rms_norm=use_rms_norm)

        self.norm2_a = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_a = torch.nn.Sequential(
            torch.nn.Linear(dim, dim*4),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(dim*4, dim)
        )


    def forward(self, hidden_states_a, hidden_states_b, hidden_states_c, temb):
        norm_hidden_states_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a = self.norm1_a(hidden_states_a, emb=temb)
        norm_hidden_states_b = self.norm1_b(hidden_states_b, emb=temb)
        norm_hidden_states_c, gate_msa_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = self.norm1_c(hidden_states_c, emb=temb)

        # Attention
        attn_output_a = self.attn(norm_hidden_states_a, norm_hidden_states_b, norm_hidden_states_c)

        # Part A
        hidden_states_a = hidden_states_a + gate_msa_a * attn_output_a
        norm_hidden_states_a = self.norm2_a(hidden_states_a) * (1 + scale_mlp_a) + shift_mlp_a
        hidden_states_a = hidden_states_a + gate_mlp_a * self.ff_a(norm_hidden_states_a)

        return hidden_states_a, hidden_states_b, hidden_states_c



class SD3DiT_3_stream(torch.nn.Module):
    def __init__(self, embed_dim=1536, num_layers=24, use_rms_norm=False, num_dual_blocks=0, pos_embed_max_size=192):
        super().__init__()
        self.pos_embedder = PatchEmbed(patch_size=2, in_channels=16, embed_dim=embed_dim, pos_embed_max_size=pos_embed_max_size)
        self.pos_embedder_LR = PatchEmbed(patch_size=2, in_channels=16, embed_dim=embed_dim, pos_embed_max_size=pos_embed_max_size)
        self.time_embedder = TimestepEmbeddings(256, embed_dim)
        self.pooled_text_embedder = torch.nn.Sequential(torch.nn.Linear(2048, embed_dim), torch.nn.SiLU(), torch.nn.Linear(embed_dim, embed_dim))
        self.context_embedder = torch.nn.Linear(4096, embed_dim)
        self.blocks = torch.nn.ModuleList([JointTransformerBlock_3_stream(embed_dim, embed_dim//64, use_rms_norm=use_rms_norm, dual=True) for _ in range(num_dual_blocks)]
                                        + [JointTransformerBlock_3_stream(embed_dim, embed_dim//64, use_rms_norm=use_rms_norm) for _ in range(num_layers-1-num_dual_blocks)]
                                        + [JointTransformerFinalBlock_3_stream(embed_dim, embed_dim//64, use_rms_norm=use_rms_norm)])

        self.norm_out = AdaLayerNorm(embed_dim, single=True)
        self.proj_out = torch.nn.Linear(embed_dim, 64)
        # self._compiled_forward = torch.compile(partial(self.forward, tiled=False), dynamic=False, fullgraph=True, mode='max-autotune-no-cudagraphs')

    def tiled_forward(self, hidden_states, LR_states, timestep, prompt_emb, pooled_prompt_emb, tile_size=128, tile_stride=64):
        # Due to the global positional embedding, we cannot implement layer-wise tiled forward.
        use_compile = False
        if use_compile:
            def chunk_function(x):
                assert x.size(1) % 2 == 0
                split_size = x.size(1) // 2
                hidden_part, lr_part = torch.split(
                    x, 
                    split_size_or_sections=split_size, 
                    dim=1
                )
                return self._compiled_forward(
                    hidden_part,
                    lr_part,
                    timestep,
                    prompt_emb,
                    pooled_prompt_emb
                )
        else:
            def chunk_function(x):
                assert x.size(1) % 2 == 0
                split_size = x.size(1) // 2
                hidden_part, lr_part = torch.split(
                    x, 
                    split_size_or_sections=split_size, 
                    dim=1
                )
                return self.forward(
                    hidden_part,
                    lr_part,
                    timestep,
                    prompt_emb,
                    pooled_prompt_emb
                )
        
        hidden_states = TileWorker().tiled_forward(
            chunk_function,
            torch.concat([hidden_states, LR_states], dim=1),  # 由于有两个输入 都需要tile 所以在c维度 先concat
            tile_size,
            tile_stride,
            tile_device=hidden_states.device,
            tile_dtype=hidden_states.dtype
        )
        return hidden_states

    @torch.no_grad()
    def init_from_basemodel(self, sd3dit, trained_state_dict = None, single_step = False):
        """
            根据sd3dit的state_dict，初始化
            并设置好requires_grad_
            
            trained_state_dict: 已经训练的部分参数 当提供时 最后加载这部分参数
        """
        base_state_dict = sd3dit.state_dict()
        missing_keys, unexpected_keys = self.load_state_dict(base_state_dict, strict=False)
        # 将 missing_keys 写入 missing_keys.txt
        with open('missing_keys.txt', 'w') as f:
            for key in missing_keys:
                f.write(f"{key}\n")

        # step1. 默认所有参数都不训练
        for param in self.parameters():
            param.requires_grad = False
        
        # step2. copy或设置相关的参数，并设置梯度flag
        # 注意 pos_embedder_LR中的位置编码最好不要训练
        self.pos_embedder_LR.load_state_dict(self.pos_embedder.state_dict(), strict=True)
        self.pos_embedder_LR.requires_grad_(False)

        # 根据作者的补充材料，这部分也需要训练
        self.context_embedder.requires_grad_(True)

        # 训练单步 
        if single_step:
            self.time_embedder.requires_grad_(True)
            self.norm_out.requires_grad_(True)
            self.proj_out.requires_grad_(True)

        for idx, block in enumerate(self.blocks):
            number = 0
            for name, param in block.named_parameters():
                number += 1
                if name in ["attn.norm_q_c.weight", "attn.norm_k_c.weight"]:
                    param.requires_grad_(True)
                elif name in ['attn.c_to_q.weight', 'attn.c_to_q.bias', 'attn.c_to_k.weight', 'attn.c_to_k.bias', 'attn.c_to_v.weight', 'attn.c_to_v.bias', 'depthwise_conv.weight', 'depthwise_conv.bias']:
                    param.zero_()  # or param.fill_(value)
                    param.requires_grad_(True)
                elif name == "norm1_c.linear.weight":
                    param.copy_(block.norm1_a.linear.weight)
                    param.requires_grad_(True)
                elif name == "norm1_c.linear.bias":
                    param.copy_(block.norm1_a.linear.bias)
                    param.requires_grad_(True)
                elif name == "attn.c_to_out.weight":
                    param.copy_(block.attn.a_to_out.weight)
                    param.requires_grad_(True)
                elif name == "attn.c_to_out.bias":
                    param.copy_(block.attn.a_to_out.bias)
                    param.requires_grad_(True)
                elif name == "ff_c.0.weight":
                    param.copy_(block.ff_a[0].weight)
                    param.requires_grad_(False)
                elif name == "ff_c.0.bias":
                    param.copy_(block.ff_a[0].bias)
                    param.requires_grad_(False)
                elif name == "ff_c.2.weight":
                    param.copy_(block.ff_a[2].weight)
                    param.requires_grad_(False)
                elif name == "ff_c.2.bias":
                    param.copy_(block.ff_a[2].bias)
                    param.requires_grad_(False)
                else:
                    number -=1
            # print(idx, number)

        # load trained params
        if trained_state_dict:
            missing_keys, unexpected_keys = self.load_state_dict(trained_state_dict, strict=False)
            with open('trained_load_missing_keys.txt', 'w') as f:
                for key in missing_keys:
                    f.write(f"{key}\n")
                    
            with open('trained_load_unexpected_keys.txt', 'w') as f:
                for key in unexpected_keys:
                    f.write(f"{key}\n")

    def forward(self, hidden_states, LR_states, timestep, prompt_emb, pooled_prompt_emb, tiled=False, tile_size=128, tile_stride=64, use_gradient_checkpointing=False):
        if tiled:
            return self.tiled_forward(hidden_states, LR_states, timestep, prompt_emb, pooled_prompt_emb, tile_size, tile_stride)
        conditioning = self.time_embedder(timestep, hidden_states.dtype) + self.pooled_text_embedder(pooled_prompt_emb)
        prompt_emb = self.context_embedder(prompt_emb)

        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embedder(hidden_states)
        LR_states = self.pos_embedder_LR(LR_states)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block in self.blocks:
            block.height_ = height // 2 # 64
            block.width_ = width // 2 # 64
            if self.training and use_gradient_checkpointing:
                hidden_states, prompt_emb, LR_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states, prompt_emb, LR_states, conditioning,
                    use_reentrant=False,
                )
            else:
                hidden_states, prompt_emb, LR_states = block(hidden_states, prompt_emb, LR_states, conditioning)
        
        hidden_states = self.norm_out(hidden_states, conditioning)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = rearrange(hidden_states, "B (H W) (P Q C) -> B C (H P) (W Q)", P=2, Q=2, H=height//2, W=width//2)
        return hidden_states