"""
    测试flux/flux-kontext/flux-mini多步模型效果
"""
import os
import torch
from generator import Generator
from discriminator import Discriminator
from diffsynth.extensions.flux.sampling import get_noise, get_schedule
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def png_to_normalized_tensor(image_path):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    tensor = torch.from_numpy(img_array).float()  # 转换为float
    tensor = tensor.permute(2, 0, 1)  # 从(H, W, C)变为(C, H, W)
    tensor = tensor / 255.0  # [0, 255] -> [0, 1]
    tensor = tensor * 2 - 1  # [0, 1] -> [-1, 1]
    tensor = tensor.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    return tensor

def viz(latent, name):
    pred_rgb = genarator.vae.decode(latent)
    pred_rgb = pred_rgb.clamp(-1, 1)
    pred_rgb = pred_rgb.cpu().float()
    grid = torchvision.utils.make_grid(pred_rgb, nrow=4, normalize=True, value_range=(-1, 1))
    grid_image = v2.ToPILImage()(grid)
    grid_image.save(name)
    
    return grid_image

genarator = Generator(
    torch_dtype = torch.bfloat16
)

# discriminator = Discriminator(
#     torch_dtype = torch.bfloat16,
#     learning_rate=0,
#     use_gradient_checkpointing=False
# )

genarator.vae.to(device = 'cuda')
genarator.t5.to(device = 'cuda')
genarator.clip.to(device = 'cuda')
genarator.dit.to(device = 'cuda')
genarator.eval()
# discriminator.to(device = 'cuda')
# discriminator.requires_grad_(False)
# discriminator.eval()

######################################################################################################################################################################################################################################

prompt = """Enhance the image while preserving as much of the original details as possible, including common textures such as faces, hands, trees, flowers, buildings, water surfaces, fabrics, stone, metal, and foliage.
                                                       realistic maximum detail, ultra HD, skin pore detailing, perfect without deformations."""
condition_img_path = '/mnt/media01/dataset/media_algo_share/xiangfeng/repos/diffsynth-studio/experiments/flux_kontext_one_step_gan_image_restoration/20250704-101817/iter_1_lq.png'
neg_prompt = ''


is_schnell = False
device = 'cuda:0'
dtype = torch.bfloat16
seed = 123

# 1,16,128,128

condition_input = png_to_normalized_tensor(image_path=condition_img_path).to(device=device, dtype=dtype)
_,_,height,width = condition_input.shape
latent = get_noise(1, height, width, device, dtype, seed)
condition_input = genarator.vae.encode(condition_input)

num_steps = 28
timesteps = get_schedule(num_steps, height//16 * width//16, shift=(not is_schnell))
print(timesteps)

prompt_emb = {}
prompt_emb['t5'] = genarator.t5(prompt)
prompt_emb['clip'] = genarator.clip(prompt)
print(prompt_emb['t5'].shape)
# prompt_emb_neg = {}
# prompt_emb_neg['t5'] = genarator.t5(neg_prompt)
# prompt_emb_neg['clip'] = genarator.clip(neg_prompt)


input_sigma = 0.2
noise_latent = (1-input_sigma) * condition_input + input_sigma * latent
viz(noise_latent, name = "noise_latent.png")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'avc1' for H.264
video_out = cv2.VideoWriter('output.mp4', fourcc, 2.0, (width, height))  # fps=2
try:
    font = ImageFont.truetype("arial.ttf", 24)
except:
    font = ImageFont.load_default()

for out_sigma in tqdm(range(1,51)):
    out_sigma = out_sigma / 10
    print(out_sigma)
    with torch.no_grad():
        # 多步
        # for i, (t_curr, t_prev) in tqdm(enumerate(zip(timesteps[:-1], timesteps[1:]))):
        #     t_curr_tensor = torch.ones(1) * t_curr * 1000
        #     pred = genarator(latent, condition_input, t_curr_tensor, prompt_emb)
        #     latent = latent + (t_prev - t_curr) * pred
        
        # 单步，看初始结果
        pred = genarator(noise_latent, condition_input, torch.ones(1) * out_sigma * 1000, prompt_emb)
        latent = noise_latent + ( 0 - out_sigma) * pred    
        
    res = viz(latent, name = f"out_sigma={out_sigma}.png")

    # Convert PIL Image to OpenCV format and add text
    img = res.copy()

    # Convert to numpy array for OpenCV
    print(np.array(img).shape)
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    video_out.write(frame)
    
# Release the video writer
video_out.release()
print("Video saved as output.mp4")