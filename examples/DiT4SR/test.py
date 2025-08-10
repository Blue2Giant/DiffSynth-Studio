from diffsynth import ModelManager, SD3ImagePipeline
import torch
from diffsynth.models.sd3_dit_3stream import SD3DiT_3_stream

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=["/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-medium/text_encoders/clip_g.safetensors",
                                             "/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-medium/text_encoders/clip_l.safetensors",
                                             "/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-medium/text_encoders/t5xxl_fp16.safetensors",
                                             "/mnt/media01/dataset/media_algo_share/xiangfeng/aigc_pretrained/stable-diffusion-3.5-medium/sd3.5_medium.safetensors"])
pipe = SD3ImagePipeline.from_model_manager(model_manager)

# {'embed_dim': 1536, 'num_layers': 24, 'use_rms_norm': True, 'num_dual_blocks': 13, 'pos_embed_max_size': 384}

# new_model = SD3DiT_3_stream(embed_dim = 1536,
#                             num_layers=24,
#                             use_rms_norm=True,
#                             num_dual_blocks=13,
#                             pos_embed_max_size=384)

# new_model = new_model.to(dtype=torch.float16, device="cuda")

# new_model.init_from_basemodel(pipe.dit)
# pipe.dit = new_model

prompt = "a cinematic front view photo of a slim white male dryad emerging from a tree, his eyes closed with his head lowered, facing the viewer with his back on the tree. His arms and chest are made of green branches and white flowers, his hair made of brown vines and branches, his body fused with the tree trunk, his skin covered in moss and leaf, his shoulders and collar bone resembling pale human skin. The photo is taken with a 35mm lens capturing the essence of golden hour."
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

torch.manual_seed(7)
image = pipe(
    prompt=prompt, 
    negative_prompt=negative_prompt,
    cfg_scale=7.5,
    num_inference_steps=50, width=1024 + 256, height=1024 + 256,
)
image.save("image_1024.jpg")