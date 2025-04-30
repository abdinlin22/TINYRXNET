import torch
import torch.nn as nn
from mmseg.models.backbones import ResNetV1c
from mmseg.models.builder import BACKBONES
from segment_anything import sam_model_registry
import torchvision
import matplotlib.pyplot as plt
import os

def compute_local_rx_score(feature_map, tile_size=32, eps=1e-5):
    B, C, H, W = feature_map.shape
    rx_map = torch.zeros((B, 1, H, W), device=feature_map.device)

    for b in range(B):
        for i in range(0, H, tile_size):
            for j in range(0, W, tile_size):
                h_t = min(tile_size, H - i)
                w_t = min(tile_size, W - j)
                tile = feature_map[b, :, i:i+h_t, j:j+w_t]  # (C, h, w)
                tile = tile.reshape(C, -1).T  # (N, C)
                if tile.shape[0] < 2:
                    continue

                mu = tile.mean(dim=0)
                diff = tile - mu
                cov = diff.T @ diff / (tile.shape[0] - 1) + eps * torch.eye(C, device=feature_map.device)
                inv_cov = torch.linalg.inv(cov)
                mahal = torch.sum((diff @ inv_cov) * diff, dim=1)  # Mahalanobis
                rx_map[b, 0, i:i+h_t, j:j+w_t] = mahal.reshape(h_t, w_t)

    return rx_map

@BACKBONES.register_module()
class ResNetWithSAMRXLAST(nn.Module):
    def __init__(self, sam_checkpoint='sam_vit_b.pth', freeze_sam=True, fuse_mode='concat', **kwargs):
        super().__init__()
        self.backbone = ResNetV1c(**kwargs)
        self.sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).image_encoder

        if freeze_sam:
            for p in self.sam.parameters():
                p.requires_grad = False
            self.sam.eval()

        self.fuse_mode = fuse_mode

    def forward(self, x):
        B, C, H, W = x.shape

        sam_input = torch.nn.functional.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        with torch.no_grad():
            sam_feat = self.sam(sam_input)  # (B, 256, 64, 64)

        outs = list(self.backbone(x))  # ResNet outputs

        for i in range(len(outs)):
            target_size = outs[i].shape[-2:]
            sam_resized = torch.nn.functional.interpolate(
                sam_feat, size=target_size, mode='bilinear', align_corners=False
            )
            if self.fuse_mode == 'concat':
                outs[i] = torch.cat([outs[i], sam_resized], dim=1)  # concat SAM to ResNet

        
        
        #for i in range(outs[2].shape[1]):
            
        #    channel_to_save = outs[2][0,i]
            
        #    channel_norm = (channel_to_save - channel_to_save.min()) / (channel_to_save.max() - channel_to_save.min() + 1e-8)
            
        #    img = torchvision.transforms.functional.to_pil_image(channel_norm)
        #    img.save(os.path.join("inference_outputs", f"channel_{i:04d}.png"))


        
        rx_score2 = compute_local_rx_score(outs[2])
        rx_score3 = compute_local_rx_score(outs[3])


        #values = rx_score2.flatten().cpu().numpy()
        
        #plt.figure(figsize=(8, 5))
        #plt.hist(values, bins=100, color='purple')
        #plt.title("Histogram of tensor1 values")
        #plt.xlabel("Value")
        #plt.ylabel("Frequency")
        #plt.grid(True)
        #plt.tight_layout()
        #plt.savefig("inference_outputs/tensor1_histogram.png")
        #plt.close()

        #tensor1_rgb = torch.clamp(rx_score2[0], min=0, max=2500)        
        #img1 = torchvision.transforms.functional.to_pil_image(tensor1_rgb)
        #img1.save("inference_outputs/rx_score2.png")



        outs[2] = torch.cat([outs[2], rx_score2], dim=1)
        outs[3] = torch.cat([outs[3], rx_score3], dim=1)

        return tuple(outs)
