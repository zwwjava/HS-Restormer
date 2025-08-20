'''
@Author  ：zww
@Date    ：2025/5/20 15:42
@Description : *
'''
import torch
import torch.nn.functional as F
from torchvision import models
import cv2
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # 注册钩子
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, lr_img):
        # 前向传播
        sr_img = self.model(lr_img)

        # 计算重建损失（MSE）
        loss = F.mse_loss(sr_img, lr_img)

        # 反向传播
        self.model.zero_grad()
        loss.backward()

        # 生成热力图
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, lr_img.shape[2:], mode='bilinear')
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.squeeze().cpu().numpy()