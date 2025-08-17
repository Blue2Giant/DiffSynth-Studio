from diffsynth.models.wan_video_vae import VideoVAE_, WanVideoVAE
import torch
wan_vae = VideoVAE_()

#随机初始化一个输入张量
x = torch.randn(1, 3, 16, 64, 64)
mean = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
]
std = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
]
mean = torch.tensor(mean)
std = torch.tensor(std)
scale = [mean, 1.0 / std]
middle = wan_vae.encode(x,scale)
print(middle.shape)