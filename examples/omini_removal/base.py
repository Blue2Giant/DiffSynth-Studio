import torch
import os
# from diffsynth.models.flux_dit_3_stream import FluxDiT_3_stream
from diffsynth.models.flux_dit_3_stream_kontext import FluxDiT_3_stream_kontext
from peft import LoraConfig, inject_adapter_in_model

def new_forward(
        self,
        hidden_states,
        LR_states,
        timestep, prompt_emb, pooled_prompt_emb, guidance, text_ids, image_ids=None, LR_ids=None,
        tiled=False, tile_size=128, tile_stride=64, entity_prompt_emb=None, entity_masks=None,
        use_gradient_checkpointing=False,
        **kwargs
    ):
    if tiled:
        return self.tiled_forward(
            hidden_states,
            LR_states,
            timestep, prompt_emb, pooled_prompt_emb, guidance, text_ids,
            tile_size=tile_size, tile_stride=tile_stride,
            **kwargs
        )

    if image_ids is None:
        image_ids = self.prepare_image_ids(hidden_states)
    if LR_ids is None:
        LR_ids = image_ids.clone()     # Create a copy
        LR_ids[:, :, 0] += 1      # Increment the first channel by 1 现在只是FLUX都应该为0

    conditioning = self.time_embedder(timestep, hidden_states.dtype) + self.pooled_text_embedder(pooled_prompt_emb)
    if self.guidance_embedder is not None:
        guidance = guidance * 1000
        conditioning = conditioning + self.guidance_embedder(guidance, hidden_states.dtype)

    height, width = hidden_states.shape[-2:]
    hidden_states = self.patchify(hidden_states)
    hidden_states = self.x_embedder(hidden_states)

    LR_states = self.patchify(LR_states)
    LR_states = self.x_embedder(LR_states)
    
    assert hidden_states.shape[1] == LR_states.shape[1]
    hidden_states = torch.concat([hidden_states, LR_states], dim=1) # concat on seuqence length

    if entity_prompt_emb is not None and entity_masks is not None:
        prompt_emb, image_rotary_emb, attention_mask = self.process_entity_masks(hidden_states, prompt_emb, entity_prompt_emb, entity_masks, text_ids, image_ids)
    else:
        prompt_emb = self.context_embedder(prompt_emb)
        # print('ids shape and value !!! ')
        # print(text_ids.shape, image_ids.shape, LR_ids.shape)
        # print(text_ids[0,0,:], image_ids[0,0,:], LR_ids[0,0,:])
        assert text_ids.shape[0] == image_ids.shape[0] == LR_ids.shape[0], 'batch size must the same'
        image_rotary_emb = self.pos_embedder(torch.cat((text_ids, image_ids, LR_ids), dim=1))
        attention_mask = None

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block in self.blocks:
        if self.training and use_gradient_checkpointing:
            hidden_states, prompt_emb = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states, prompt_emb, conditioning, image_rotary_emb, attention_mask,
                use_reentrant=False,
            )
        else:
            hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb, attention_mask)

    hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
    for block in self.single_blocks:
        if self.training and use_gradient_checkpointing:
            hidden_states, prompt_emb = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states, prompt_emb, conditioning, image_rotary_emb, attention_mask,
                use_reentrant=False,
            )
        else:
            hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb, attention_mask)
    hidden_states = hidden_states[:, prompt_emb.shape[1]:]
    # select first half
    _,l,_ = hidden_states.shape
    hidden_states = hidden_states[:, :l//2, :]

    hidden_states = self.final_norm_out(hidden_states, conditioning)
    hidden_states = self.final_proj_out(hidden_states)
    hidden_states = self.unpatchify(hidden_states, height, width)

    return hidden_states



def new_forward_mask_condition(
        self,
        hidden_states,
        LR_states,
        timestep, prompt_emb, pooled_prompt_emb, guidance, text_ids, image_ids=None, LR_ids=None,
        tiled=False, tile_size=128, tile_stride=64, entity_prompt_emb=None, entity_masks=None,
        use_gradient_checkpointing=False,
        **kwargs
    ):
    if tiled:
        return self.tiled_forward(
            hidden_states,
            LR_states,
            timestep, prompt_emb, pooled_prompt_emb, guidance, text_ids,
            tile_size=tile_size, tile_stride=tile_stride,
            **kwargs
        )

    if image_ids is None:
        image_ids = self.prepare_image_ids(hidden_states)
    if LR_ids is None:
        LR_ids = image_ids.clone()     # Create a copy
        LR_ids[:, :, 0] += 1      # Increment the first channel by 1 现在只是FLUX都应该为0

    conditioning = self.time_embedder(timestep, hidden_states.dtype) + self.pooled_text_embedder(pooled_prompt_emb)
    if self.guidance_embedder is not None:
        guidance = guidance * 1000
        conditioning = conditioning + self.guidance_embedder(guidance, hidden_states.dtype)

    height, width = hidden_states.shape[-2:]
    hidden_states = self.patchify(hidden_states)
    hidden_states = self.x_embedder(hidden_states)

    LR_states = self.patchify(LR_states)
    LR_states = self.LR_x_embedder(LR_states)
    
    #assert hidden_states.shape[1] == LR_states.shape[1]
    hidden_states = torch.concat([hidden_states, LR_states], dim=1) # concat on seuqence length

    if entity_prompt_emb is not None and entity_masks is not None:
        prompt_emb, image_rotary_emb, attention_mask = self.process_entity_masks(hidden_states, prompt_emb, entity_prompt_emb, entity_masks, text_ids, image_ids)
    else:
        prompt_emb = self.context_embedder(prompt_emb)
        # print('ids shape and value !!! ')
        # print(text_ids.shape, image_ids.shape, LR_ids.shape)
        # print(text_ids[0,0,:], image_ids[0,0,:], LR_ids[0,0,:])
        assert text_ids.shape[0] == image_ids.shape[0] == LR_ids.shape[0], 'batch size must the same'
        image_rotary_emb = self.pos_embedder(torch.cat((text_ids, image_ids, LR_ids), dim=1))
        attention_mask = None

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block in self.blocks:
        if self.training and use_gradient_checkpointing:
            hidden_states, prompt_emb = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states, prompt_emb, conditioning, image_rotary_emb, attention_mask,
                use_reentrant=False,
            )
        else:
            hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb, attention_mask)

    hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
    for block in self.single_blocks:
        if self.training and use_gradient_checkpointing:
            hidden_states, prompt_emb = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states, prompt_emb, conditioning, image_rotary_emb, attention_mask,
                use_reentrant=False,
            )
        else:
            hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb, attention_mask)
    hidden_states = hidden_states[:, prompt_emb.shape[1]:]
    # select first half
    _,l,_ = hidden_states.shape
    hidden_states = hidden_states[:, :l//2, :]

    hidden_states = self.final_norm_out(hidden_states, conditioning)
    hidden_states = self.final_proj_out(hidden_states)
    hidden_states = self.unpatchify(hidden_states, height, width)

    return hidden_states


class BaseModelForT2ILoRA(torch.nn.Module):
    def __init__(
        self,
        learning_rate=1e-4,
        use_gradient_checkpointing=True,
    ):
        super().__init__()
        # Set parameters
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing

    # for G & D
    def operate_parameters(self):
        self.pipe.requires_grad_(False) # vae,clip,t5 to requires_grad_ False 
        self.pipe.eval()
        self.pipe.denoising_model().train()
        
        self.pipe.dit.forward = new_forward.__get__(self.pipe.dit)
        
        # 设置final_norm_out，final_proj_out，time_embedder，guidance_embedder全部训练
        trainable_modules = [
            self.pipe.dit.final_norm_out,
            self.pipe.dit.final_proj_out,
            self.pipe.dit.time_embedder,
            self.pipe.dit.guidance_embedder,
            self.pipe.dit.x_embedder
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
        
    @torch.no_grad()
    def save_ckpt(self, ckpt_workdir, iter, tag):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.state_dict()
        lora_state_dict = {}
        
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        
        if not os.path.exists(ckpt_workdir):
            os.makedirs(ckpt_workdir, exist_ok=True)
            
        torch.save(lora_state_dict, os.path.join(ckpt_workdir, f'net_{tag}_iter_{iter}.pth'))