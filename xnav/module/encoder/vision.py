import cv2
import torch
from torch.nn import functional as F
import torch.nn as nn
import math

import numpy as np
from PIL import Image as PILImage

from xnav.module.encoder.depth_anything_v2.dinov2 import DINOv2
from xnav.module.projector.q_former import QFormerProjector

class VisionEncoder(nn.Module):
    """
    This is a abstract class that takes in images and returns image tokens
    """
    def __init__(self):
        super().__init__()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Alias to emb_images for Module API compatibility."""
        return self.emb_images(images)

    # NOTE: 此处不处理什么history images和current image，这个应该是模型搭建时，选择不同的VisionEncoder来分别处理history image和current image
    def emb_images(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        embed a batch of raw images.

        args:
        - images: A batch of images to be processed, with the shape of (..., C, H, W)

        returns:
        - The embedded image tokens with the shape of (..., N, D), where N is the number of tokens and D is the token dimension. N depends on the encoder architecture, often number of patches.
        """
        raise NotImplementedError("Infer_image method not implemented")

    
    
class DinoV2Encoder(VisionEncoder):
    """
    DINOv2 Vision Encoder
    """
    
    DIM_MAP = {
        "vits": 384,
        "vitb": 768,
        "vitl": 1024,
        "vitg": 1536,
    }
    mean: torch.Tensor = torch.tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1)
    std: torch.Tensor = torch.tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1)
    
    emb_dim: int

    def __init__(
        self,
        encoder: str = 'vits',
    ):
        super(DinoV2Encoder, self).__init__()
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        self.emb_dim = self.DIM_MAP[encoder]
        
    def emb_images(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        embed a batch of raw images.

        args:
        - images: A batch of images to be processed, with the shape of (..., C, H, W)

        returns:
        - The embedded image tokens with the shape of (..., N, D), where N is the number of tokens and D is the token dimension. N depends on the encoder architecture, often number of patches.
        """
        
        # NOTE: normalize images, this should be down as the code in depth_anything_v2
        images = images / 255.0 # scale to [0, 1]
        images = (images - self.mean.to(images.device)) / self.std.to(images.device)
        
        # reshape to (N, C, H, W)
        ori_shape = images.shape
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1]) 
        out = self.pretrained.get_intermediate_layers(images)[0]  # get the last layer features, shape (N, num_patches, D)
        out = out.reshape(ori_shape[:-3] + out.shape[1:])

        return out
    

class ImageGoalEncoder(VisionEncoder):
    """
    This encoder encodes goal images into vision tokens using a DINOv2 backbone and a Q-Former projector.
    """

    def __init__(self, encoder: str = 'vits', num_tokens: int = 32, emb_dim: int = 2048,
                q_former_layers: int = 2, q_former_heads: int = 32, dropout: float = 0.2):
        """
        Initialize the ImageGoalEncoder.

        Args:
        - encoder: The encoder type to use (e.g., 'vits').
        - num_tokens: The number of tokens to generate.
        - emb_dim: The embedding dimension.
        """
        super(ImageGoalEncoder, self).__init__()
        self.dinov2 = DinoV2Encoder(encoder=encoder)
        self.projector = QFormerProjector(
            d_model=emb_dim,
            num_queries=num_tokens,
            num_heads=q_former_heads,
            num_layers=q_former_layers,
            dropout=dropout,
            input_dim=self.dinov2.emb_dim,
            output_dim=emb_dim,
        )

    def emb_images(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed a batch of goal images.

        Args:
        - images: A batch of goal images to be processed, with the shape of (..., C, H, W)

        Returns:
        - The embedded goal image tokens with the shape of (..., N, D), where N is the number of tokens and D is the token dimension.
        """
        dinov2_tokens = self.dinov2.emb_images(images)  # (..., num_patches, D)
        goal_tokens = self.projector(dinov2_tokens)     # (..., N, D)
        return goal_tokens


def grid_pooling(vision_tokens: torch.Tensor, H: int, W: int, kernel_size: int, stride: int):
    """
    Pool vision tokens arranged in a grid back to a smaller grid.

    Args:
    - vision_tokens: Tensor with shape (..., L, D), where L = H * W
    - H: Height of the original grid
    - W: Width of the original grid
    - kernel_size: Pooling kernel size
    - stride: Pooling stride

    Returns:
    - pooled_tokens: Tensor with shape (..., L_new, D), where L_new = H_new * W_new
    """
    *batch_dims, L, D = vision_tokens.shape
    assert L == H * W, "L must be equal to H * W"
    
    B = np.prod(batch_dims).item() if len(batch_dims) > 0 else 1
    # Reshape to (B, H, W, D) and permute to (B, D, H, W)
    vision_tokens = vision_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)

    x = vision_tokens
    # (B, D, H, W) -> (B, D, H_new, W_new)
    x_pooled = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, 
                            ceil_mode=True, count_include_pad=False)

    # (B, D, H_new, W_new) -> (B, H_new * W_new, D)
    pooled_tokens = x_pooled.permute(0, 2, 3, 1).reshape(B, -1, D)
    
    # back to (..., L_new, D)
    pooled_tokens = pooled_tokens.reshape(batch_dims + list(pooled_tokens.shape[1:]))
    pooled_tokens = pooled_tokens.contiguous()
    
    return pooled_tokens
    

def resize_ensure_multiple_of(x: torch.Tensor, multiple: int=14, mode: str='bicubic'):
    """
    x: Tensor with shape (..., C, H, W)
    """
    *batch_dims, C, H, W = x.shape

    new_H = math.ceil(H / multiple) * multiple
    new_W = math.ceil(W / multiple) * multiple

    if new_H == H and new_W == W:
        return x

    # 使用 interpolate 进行缩放
    return F.interpolate(
        x, size=(new_H, new_W),
        mode=mode,
        align_corners=False if mode in ["bilinear", "bicubic"] else None
    )


if __name__ == '__main__':
    DEVICE = 'cuda'
    PATCH_SIZE = 14
    
    raw_img = cv2.imread('assets/test.png')
    print(f"raw_img shape: {raw_img.shape} type: {type(raw_img)}")  # (578, 770, 3)
    image = torch.Tensor(raw_img).permute(2, 0, 1).unsqueeze(0).float()
    image = resize_ensure_multiple_of(image, multiple=PATCH_SIZE)
    print(f"resized image tensor shape: {image.shape} type: {type(image)}")
    
    image = image.to(DEVICE)
    N = 4
    # copy N times and stack
    images = image.repeat(N, 1, 1, 1)
    images = images.reshape(2, 2, 3, images.shape[-2], images.shape[-1])
    print(f"image tensor shape: {images.shape} type: {type(images)}")
    encoder = DinoV2Encoder(encoder='vits').to(DEVICE).eval()
    img_tokens = encoder.emb_images(images)
    print(f"img tokens shape: {img_tokens.shape} type: {type(img_tokens)}")

    # Test XNavVisionEncoder with custom projector dims
    num_queries = 16
    q_dim = 256   # Q-Former latent dim (can differ from DINOv2 output)
    out_dim = 128 # Final output dim
    
    ### Test Grid Pooling
    B = 2
    H = 15
    W = 15
    D = 128
    kernel_size = 2
    stride = 2
    vision_tokens = torch.randn(B, H * W, D).to(DEVICE)
    pooled_tokens = grid_pooling(vision_tokens, H, W, kernel_size, stride)
    H_new = math.ceil(H / stride)
    W_new = math.ceil(W / stride)
    print(f"pooled tokens shape: {pooled_tokens.shape} (expect: {(B, H_new * W_new, D)})")


    # from xnav.module.encoder.depth_anything_v2.dpt import DepthAnythingV2
    # model_configs = {
    #     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    #     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    #     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    #     'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    # }

    # encoder = 'vits' # or 'vits', 'vitb', 'vitg'

    # model = DepthAnythingV2(**model_configs[encoder])
    # model.load_state_dict(torch.load('/home/shrelic/models/depth_anything_v2_vits.pth', map_location='cpu'))
    # model = model.to(DEVICE).eval()

    # raw_img = cv2.imread('assets/test.png')
    # print(f"raw_img shape: {raw_img.shape} type: {type(raw_img)}")  # (578, 770, 3)
    # depth = model.infer_image(raw_img) # HxW raw depth map in numpy
    
    # print(depth.shape)  # (578, 770)
    
    # # draw depth map and save to assets/depth.png
    # depth_min = depth.min()
    # depth_max = depth.max()
    # depth_vis = (depth - depth_min) / (depth_max - depth_min)
    # depth_vis = (depth_vis * 255).astype('uint8')
    # depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    # cv2.imwrite('assets/depth.png', depth_vis)
    
    
    # INPUT_SIZE=518
    # image, (h, w) = model.image2tensor(raw_img, INPUT_SIZE)
    # image = image.to(DEVICE)
    # print(f"image tensor shape: {image.shape} type: {type(image)}")
    # img_tokens = model.pretrained.get_intermediate_layers(image)
    
    # for i, t in enumerate(img_tokens):
    #     print(f"img token {i} shape: {t.shape} type: {type(t)}")
    