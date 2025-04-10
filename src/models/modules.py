import torch
import torch.nn as nn

# Assuming layers.py is in the same directory or path is configured
from .layers import DepthConvBlock, ResidualBlock

__all__ = [
    "Generator",
    "RGB_Encoder",
    "Gaussian_Predictor",
    "Decoder_Fusion",
    "Label_Encoder",
]


class Generator(nn.Sequential):
    def __init__(self, input_nc, output_nc):
        super(Generator, self).__init__(
            DepthConvBlock(input_nc, input_nc),
            ResidualBlock(input_nc, input_nc // 2),
            DepthConvBlock(input_nc // 2, input_nc // 2),
            ResidualBlock(input_nc // 2, input_nc // 4),
            DepthConvBlock(input_nc // 4, input_nc // 4),
            ResidualBlock(input_nc // 4, input_nc // 8),
            DepthConvBlock(input_nc // 8, input_nc // 8),
            nn.Conv2d(input_nc // 8, output_nc, 1),
        )

    def forward(self, input):
        return super().forward(input)


class RGB_Encoder(nn.Sequential):
    def __init__(self, in_chans, out_chans):
        super(RGB_Encoder, self).__init__(
            ResidualBlock(in_chans, out_chans // 8),
            DepthConvBlock(out_chans // 8, out_chans // 8),
            ResidualBlock(out_chans // 8, out_chans // 4),
            DepthConvBlock(out_chans // 4, out_chans // 4),
            ResidualBlock(out_chans // 4, out_chans // 2),
            DepthConvBlock(out_chans // 2, out_chans // 2),
            nn.Conv2d(out_chans // 2, out_chans, 3, padding=1),
        )

    def forward(self, image):
        return super().forward(image)


class Label_Encoder(nn.Sequential):
    def __init__(self, in_chans, out_chans, norm_layer=nn.BatchNorm2d):
        super(Label_Encoder, self).__init__(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_chans, out_chans // 2, kernel_size=7, padding=0),
            norm_layer(out_chans // 2),
            nn.LeakyReLU(True),
            ResidualBlock(in_ch=out_chans // 2, out_ch=out_chans),
        )

    def forward(self, image):
        return super().forward(image)


class Gaussian_Predictor(nn.Sequential):
    def __init__(self, in_chans=48, out_chans=96):
        super(Gaussian_Predictor, self).__init__(
            ResidualBlock(in_chans, out_chans // 4),
            DepthConvBlock(out_chans // 4, out_chans // 4),
            ResidualBlock(out_chans // 4, out_chans // 2),
            DepthConvBlock(out_chans // 2, out_chans // 2),
            ResidualBlock(out_chans // 2, out_chans),
            nn.LeakyReLU(True),
            nn.Conv2d(out_chans, out_chans * 2, kernel_size=1), # Output channels doubled for mu and logvar
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, img, label):
        feature = torch.cat([img, label], dim=1)
        parm = super().forward(feature)
        mu, logvar = torch.chunk(parm, 2, dim=1)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


class Decoder_Fusion(nn.Sequential):
    def __init__(self, in_chans=48, out_chans=96):
        # Note: Adjust in_chans based on the concatenation: img_emb + label_emb + z
        # Example: if F_dim=24, L_dim=24, N_dim=48, then in_chans = 24+24+48 = 96
        # The provided code seems to have a mismatch in default values (in_chans=48).
        # Adjust based on actual dimensions used in VAE_Model.
        # Here, using the default in_chans from the provided code as a placeholder.

        super().__init__(
            DepthConvBlock(in_chans, in_chans),
            ResidualBlock(in_chans, in_chans // 4),
            DepthConvBlock(in_chans // 4, in_chans // 2),
            ResidualBlock(in_chans // 2, in_chans // 2),
            DepthConvBlock(in_chans // 2, out_chans // 2),
            nn.Conv2d(out_chans // 2, out_chans, 1, 1),
        )

    def forward(self, img, label, parm):
        # print(f"[Debug Fusion Input] img shape: {img.shape}, label shape: {label.shape}, parm shape: {parm.shape}")
        feature = torch.cat([img, label, parm], dim=1)
        return super().forward(feature) 