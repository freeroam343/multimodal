"""Image backbone for 4D-Former: ResNet-50 + FPN.

Produces multi-scale feature maps {s4, s8} (strides 4 and 8 w.r.t. the input
image), each projected to the decoder hidden dimension D. These correspond to
the paper's {I4, I8} feature maps used for point-level fusion (I4) and the
decoder's image cross-attention (I4, I8).
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor


class ResNetFPN(nn.Module):
    def __init__(self, out_dim=256, pretrained=True, freeze_bn=True):
        super().__init__()
        weights = (
            torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        resnet = torchvision.models.resnet50(weights=weights)
        # ResNet layer outputs and their strides / channel counts:
        #   layer1 -> stride 4,  256ch   (C2)
        #   layer2 -> stride 8,  512ch   (C3)
        #   layer3 -> stride 16, 1024ch  (C4)
        #   layer4 -> stride 32, 2048ch  (C5)
        self.body = create_feature_extractor(
            resnet,
            return_nodes={
                "layer1": "c2",
                "layer2": "c3",
                "layer3": "c4",
                "layer4": "c5",
            },
        )
        in_ch = {"c2": 256, "c3": 512, "c4": 1024, "c5": 2048}

        # Lateral 1x1 convs to out_dim
        self.lat_c5 = nn.Conv2d(in_ch["c5"], out_dim, 1)
        self.lat_c4 = nn.Conv2d(in_ch["c4"], out_dim, 1)
        self.lat_c3 = nn.Conv2d(in_ch["c3"], out_dim, 1)
        self.lat_c2 = nn.Conv2d(in_ch["c2"], out_dim, 1)

        # 3x3 smoothing convs for the two output levels we use (P2=s4, P3=s8)
        self.out_s4 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.out_s8 = nn.Conv2d(out_dim, out_dim, 3, padding=1)

        self.freeze_bn = freeze_bn
        if freeze_bn:
            self._freeze_bn()

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            self._freeze_bn()
        return self

    @staticmethod
    def _upsample_add(x, y):
        return nn.functional.interpolate(
            x, size=y.shape[-2:], mode="nearest"
        ) + y

    def forward(self, image):
        """image: (B, 3, H, W) normalized tensor. Returns dict of feature maps."""
        c = self.body(image)
        p5 = self.lat_c5(c["c5"])
        p4 = self._upsample_add(p5, self.lat_c4(c["c4"]))
        p3 = self._upsample_add(p4, self.lat_c3(c["c3"]))
        p2 = self._upsample_add(p3, self.lat_c2(c["c2"]))
        return {
            "s4": self.out_s4(p2),  # stride 4  -> I4
            "s8": self.out_s8(p3),  # stride 8  -> I8
        }
