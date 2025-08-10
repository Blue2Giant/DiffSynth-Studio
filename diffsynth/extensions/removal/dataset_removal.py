"""
copy from https://github.com/csslc/PiSA-SR/blob/main/src/datasets/dataset.py
"""
import os
import random
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from pathlib import Path
import base64
from pycocotools import mask as maskUtils
import numpy as np
import json
import pillow_heif
pillow_heif.options.DISABLE_SECURITY_LIMITS = True
import math 
import cv2
import albumentations as A
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def decode_rle_for_demo(rle_counts_str, image_size):
    '''
    解码 RLE counts 到二值掩码
    '''
    counts_binary = base64.b64decode(rle_counts_str.encode('utf-8'))  # 字符串->二进制
    rle_dict = {'counts': counts_binary, 'size': image_size}
    return maskUtils.decode(rle_dict)

class MaskAugmentor:
    def __init__(self, flip_prob=0.5):
        """
        掩码增强变换器
        
        参数:
            flip_prob (float): 水平翻转的概率，默认0.5
        """
        self.flip_prob = flip_prob
        
    def __call__(self, mask, operations=None):
        """
        对输入掩码应用增强操作
        
        参数:
            mask (np.ndarray): 输入的二值掩码
            operations (list): 要执行的操作列表，None表示随机选择1-3种操作
            
        返回:
            np.ndarray: 增强后的掩码
        """
        if mask.max()<2:#mask 必须是0-255
            mask = mask*255
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        # 随机选择0-1种操作
        if operations is None:
            all_ops = ['erode', 'dilate', 'convex_hull', 'ellipse', 'bbox', 'bezier']
            num_ops = random.randint(0,1)
            operations = random.sample(all_ops, num_ops)
        
        # 应用所有选中的操作
        for op in operations:
            if op == 'erode':
                kernel_size = random.choice([3, 5, 7])
                iterations = random.randint(1, 3)
                mask = self.erode_mask(mask, kernel_size, iterations)
                
            elif op == 'dilate':
                kernel_size = random.choice([3, 5, 7])
                iterations = random.randint(1, 3)
                mask = self.dilate_mask(mask, kernel_size, iterations)
                
            elif op == 'convex_hull':
                mask = self.convex_hull_mask(mask)
                
            elif op == 'ellipse':
                mask = self.ellipse_mask(mask)
                
            elif op == 'bbox':
                mask = self.bbox_mask(mask)
                
            elif op == 'bezier':
                n_points = random.randint(50, 200)
                smoothness = random.uniform(0.01, 1.0)
                mask = self.bezier_mask(mask, n_points, smoothness)
        
        return mask
    
    def erode_mask(self, mask, kernel_size=3, iterations=1):
        """腐蚀操作：减小掩码的白色区域"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)

    def dilate_mask(self, mask, kernel_size=3, iterations=1):
        """膨胀操作：扩大掩码的白色区域"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)

    def convex_hull_mask(self, mask):
        """凸包操作：计算掩码的凸包并填充"""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_mask = np.zeros_like(mask)
        
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            cv2.fillPoly(hull_mask, [hull], 255)
            
        return hull_mask

    def ellipse_mask(self, mask):
        """椭圆拟合：用最佳拟合椭圆替换原掩码"""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ellipse_mask = np.zeros_like(mask)
        
        for cnt in contours:
            if len(cnt) >= 5:  # 需要至少5个点拟合椭圆
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    # 解构椭圆参数
                    center, axes, angle = ellipse
                    
                    # 确保轴值为非负
                    corrected_axes = (abs(axes[0]), abs(axes[1]))
                    corrected_ellipse = (center, corrected_axes, angle)
                    
                    cv2.ellipse(ellipse_mask, corrected_ellipse, 255, -1)
                except cv2.error as e:
                    # 处理其他可能的椭圆绘制错误
                    print(f"椭圆绘制错误: {e}")
                    print(f'mask is too mall too draw a ellipse, return original mask')
                    # 回退方案：使用轮廓的凸包代替
                    # hull = cv2.convexHull(cnt)
                    # cv2.drawContours(ellipse_mask, [hull], 0, 255, -1)
                
        return mask

    def bbox_mask(self, mask):
        """矩形边界框：用矩形边界框替换原掩码"""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox_mask = np.zeros_like(mask)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(bbox_mask, (x, y), (x+w, y+h), 255, -1)  # -1 表示填充
                
        return bbox_mask

    def bezier_mask(self, mask, n_points=100, smoothness=0.01):
        """贝塞尔曲线平滑：使用贝塞尔曲线平滑掩码边界"""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bezier_mask = np.zeros_like(mask)
        
        for cnt in contours:
            cnt = cnt.squeeze()
            if len(cnt) > 3:
                # 闭合曲线处理（首尾相连）
                cnt = np.vstack([cnt, cnt[0]])
                
                # 参数化曲线
                tck, u = splprep(cnt.T, u=None, s=smoothness, per=1)
                u_new = np.linspace(0, 1, n_points)
                x_new, y_new = splev(u_new, tck, der=0)
                
                # 创建新轮廓并填充
                bezier_contour = np.array([x_new, y_new]).T.reshape(-1, 1, 2).astype(np.int32)
                cv2.fillPoly(bezier_mask, [bezier_contour], 255)
                
        return bezier_mask

    def apply_single(self, mask, operation, **kwargs):
        """应用单个指定操作"""
        op_map = {
            'flip': self.flip_mask,
            'erode': self.erode_mask,
            'dilate': self.dilate_mask,
            'convex_hull': self.convex_hull_mask,
            'ellipse': self.ellipse_mask,
            'bbox': self.bbox_mask,
            'bezier': self.bezier_mask
        }
        
        if operation not in op_map:
            raise ValueError(f"无效的操作: {operation}. 可用操作: {list(op_map.keys())}")
            
        return op_map[operation](mask, **kwargs) if kwargs else op_map[operation](mask)


class Syn4Removal(torch.utils.data.Dataset):
    """
        args should have:
        ▪ json_path: 所有的图片的标注的json

        json 中包含了原图和粘贴后的图以及mask字符串
    """
    def __init__(self, json_txt_list,split = 'train',use_mask=False):
        super().__init__()

        self.split = split
        self.augmentation= A.Compose([
            #刚体变换
            A.RandomCrop(1024,1024),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ],additional_targets = {'mask': 'image','gt':'image'})
        self.mask_augmentor = MaskAugmentor()
        self.use_mask = use_mask

        if json_txt_list is not None:#highquality现在可以是json文件路径
            with open(json_txt_list, 'r') as f:
                self.json_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, idx):
        annotations = json.load(open(self.json_list[idx], 'r'))
        to_remove_path = annotations['target_path']
        gt_path = annotations['source_path']
        to_remove = cv2.cvtColor(cv2.imread(to_remove_path), cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
        anns = annotations['annotations']
        for ann in anns:
            mask = decode_rle_for_demo(
                ann['segmentation']['counts'],
                ann['segmentation']['size']
            )        
        #统一增强
        transformed = self.augmentation(image=to_remove, mask=mask,gt=gt)
        to_remove=transformed['image']
        mask=transformed['mask']
        gt=transformed['gt']
        # print(mask.shape)
        #对mask增强,
        # augmented_mask = self.mask_augmentor(mask)
        augmented_mask = mask * 255
        if not self.use_mask: #不concat，直接乘上
            to_remove = to_remove * (1 - (augmented_mask[:,:,None] / 255.0))
        augmented_mask = (augmented_mask/127.5) - 1
        #把图像归一化到-1到+1
        to_remove = (to_remove / 127.5) - 1
        gt = (gt / 127.5) - 1
        #把图像和mask转换为tensor
        to_remove = torch.from_numpy(to_remove).permute(2, 0, 1)
        gt = torch.from_numpy(gt).permute(2, 0, 1)
        augmented_mask = torch.from_numpy(augmented_mask)
        augmented_mask = torch.repeat_interleave(augmented_mask.unsqueeze(0), 3, dim=0)#为了能够过VAE
        # to_remove = torch.concat([to_remove, augmented_mask], dim=0)
        example = {}
        example["gt"] = gt
        example["lq"] = to_remove
        example["mask"] = augmented_mask
        example['to_remove_path'] = to_remove_path
        example['gt_path'] = gt_path
        return example

# 创建保存图像的目录


# 可视化单个样本函数
def visualize_sample(gt_tensor, lq_tensor, mask_tensor, save_path):
    """
    将三个张量可视化并保存为图像
    gt_tensor: 原始图像张量 (C, H, W)
    lq_tensor: 低质量图像张量 (C, H, W)
    mask_tensor: 掩码张量 (C, H, W)
    save_path: 保存路径前缀
    """
    # 转换为numpy数组并调整通道顺序为HxWxC
    gt_np = gt_tensor.numpy().transpose(1, 2, 0)
    lq_np = lq_tensor.numpy().transpose(1, 2, 0)
    mask_np = mask_tensor.numpy().transpose(1, 2, 0)
    
    # 从归一化值恢复到0-255
    gt_img = ((gt_np + 1) * 127.5).astype(np.uint8)
    lq_img = ((lq_np + 1) * 127.5).astype(np.uint8)
    mask_img = ((mask_np+1) * 127.5).astype(np.uint8)
    
    # 创建叠加图像（在低质量图像上显示掩码区域）
    overlay = lq_img.copy()
    mask_area = mask_img.mean(axis=-1) > 0  # 创建掩码区域布尔图
    overlay[mask_area] = [255, 0, 0]  # 红色表示掩码区域
    
    # 创建4x1的图像布局
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    # 显示GT图像
    axes[0].imshow(gt_img)
    axes[0].set_title('GT Image')
    axes[0].axis('off')
    
    # 显示LQ图像
    axes[1].imshow(lq_img)
    axes[1].set_title('LQ Image')
    axes[1].axis('off')
    
    # 显示Mask
    axes[2].imshow(mask_img.mean(axis=-1), cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Mask')
    axes[2].axis('off')
    
    # 显示叠加图像
    axes[3].imshow(overlay)
    axes[3].set_title('LQ with Mask Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()
    
    # 单独保存每个图像
    Image.fromarray(gt_img).convert('RGB').save(f"{save_path}_gt.png")
    Image.fromarray(lq_img).convert('RGB').save(f"{save_path}_lq.png")
    Image.fromarray(mask_img.mean(axis=-1).astype(np.uint8)).save(f"{save_path}_mask.png")
    Image.fromarray(overlay).save(f"{save_path}_overlay.png")

if __name__ == "__main__":
    txt_list = '/mnt/media01/dataset/media_algo_share/lanjinghong/datasets/syn4removal.txt'
    syn4 = Syn4Removal(json_txt_list=txt_list)
    save_dir = "syn4removal_visualization"
    os.makedirs(save_dir, exist_ok=True)
    # 获取多个样本并可视化
    for i in range(5):  # 可视化前5个样本
        example = syn4.__getitem__(i)
        
        # 打印张量信息
        print(f"\nSample {i}:")
        print(f"GT - Min: {example['gt'].min().item():.4f}, Max: {example['gt'].max().item():.4f}, Shape: {example['gt'].shape}")
        print(f"LQ - Min: {example['lq'].min().item():.4f}, Max: {example['lq'].max().item():.4f}, Shape: {example['lq'].shape}")
        print(f"Mask - Min: {example['mask'].min().item()}, Max: {example['mask'].max().item()}, Shape: {example['mask'].shape}")
        print(f"Mask unique values: {torch.unique(example['mask'])}")
        
        # 可视化并保存
        save_path = os.path.join(save_dir, f"sample_{i}")
        visualize_sample(
            example['gt'], 
            example['lq'], 
            example['mask'],
            save_path
        )
        print(example['to_remove_path'])
        print(example['gt_path'])
        print(f"Images saved to {save_path}*.png")

    print(f"所有可视化图像已保存到 {save_dir} 目录")