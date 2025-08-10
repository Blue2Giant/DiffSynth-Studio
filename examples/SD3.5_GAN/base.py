import torch
import os
from diffsynth.models.sd3_dit_3stream import SD3DiT_3_stream
from peft import LoraConfig, inject_adapter_in_model


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

    def operate_parameters(self):
        new_model = SD3DiT_3_stream(embed_dim = 1536,
                            num_layers=24,
                            use_rms_norm=True,
                            num_dual_blocks=13,
                            pos_embed_max_size=384)
        # new_model = SD3DiT_3_stream(embed_dim = 2432,
        #             num_layers=38,
        #             use_rms_norm=True,
        #             num_dual_blocks=0,
        #             pos_embed_max_size=192)
        
        # note: very important , bfloat16
        new_model = new_model.to(dtype=torch.bfloat16, device='cpu')
        
        self.pipe.requires_grad_(False) # vae,clip,t5 to requires_grad_ False 
        
        new_model.init_from_basemodel(self.pipe.dit, single_step = True)
        
        # original_trainable_params = {
        #     name for name, param in new_model.named_parameters() if param.requires_grad
        # }
        
        # # add lora to new_model
        # # 只对某些不训练的模块加lora 需要训练的模块因为peft库 也不训练了 所以需要置回训练
        # lora_config = LoraConfig(
        #     r=32,
        #     lora_alpha=32,
        #     init_lora_weights=True,
        #     target_modules=["ff_a.0", 
        #                     "ff_a.2",
        #                     "ff_b.0",
        #                     "ff_b.2",
        #                     "ff_c.0",
        #                     "ff_c.2",
        #                     "attn2.a_to_qkv",
        #                     "attn2.a_to_out"
        #                     ]
        # )
        # new_model = inject_adapter_in_model(lora_config, new_model)
        
        # for name, param in new_model.named_parameters():
        #     if name in original_trainable_params:
        #         param.requires_grad = True
        
        self.pipe.dit = new_model
        
        self.pipe.eval()
        self.pipe.denoising_model().train()

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