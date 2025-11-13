"""GAN modules for adversarial training."""

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for image quality assessment."""
    
    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 64
    ):
        """
        Initialize PatchGAN discriminator.
        
        Args:
            input_channels: Number of input channels
            hidden_channels: Number of hidden channels
        """
        super().__init__()
        
        def conv_block(in_ch, out_ch, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                ),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        self.model = nn.Sequential(
            nn.Conv2d(
                input_channels, hidden_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            conv_block(hidden_channels, hidden_channels * 2),
            conv_block(hidden_channels * 2, hidden_channels * 4),
            conv_block(hidden_channels * 4, hidden_channels * 8),
            
            # Final layer outputs patch-wise classification
            nn.Conv2d(
                hidden_channels * 8, 1,
                kernel_size=4, stride=1, padding=1
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator.
        
        Args:
            x: Input image tensor (B, C, H, W)
            
        Returns:
            Patch classification (B, 1, H/16, W/16)
        """
        return self.model(x)


class GANLoss(nn.Module):
    """GAN loss combining adversarial and pixel-wise losses."""
    
    def __init__(self, lambda_gan: float = 1.0, lambda_pixel: float = 1.0):
        """
        Initialize GAN loss.
        
        Args:
            lambda_gan: Weight for adversarial loss
            lambda_pixel: Weight for pixel-wise loss
        """
        super().__init__()
        self.lambda_gan = lambda_gan
        self.lambda_pixel = lambda_pixel
        
        self.gan_loss = nn.BCEWithLogitsLoss()
        self.pixel_loss = nn.L1Loss()
    
    def discriminator_loss(
        self,
        disc_real: torch.Tensor,
        disc_fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate discriminator loss.
        
        Args:
            disc_real: Discriminator output for real images
            disc_fake: Discriminator output for fake (generated) images
            
        Returns:
            Discriminator loss
        """
        real_labels = torch.ones_like(disc_real)
        fake_labels = torch.zeros_like(disc_fake)
        
        loss_real = self.gan_loss(disc_real, real_labels)
        loss_fake = self.gan_loss(disc_fake, fake_labels)
        
        return (loss_real + loss_fake) / 2
    
    def generator_loss(
        self,
        disc_fake: torch.Tensor,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate generator loss.
        
        Args:
            disc_fake: Discriminator output for generated images
            generated: Generated images
            target: Target images
            
        Returns:
            Generator loss
        """
        real_labels = torch.ones_like(disc_fake)
        
        loss_gan = self.gan_loss(disc_fake, real_labels)
        loss_pixel = self.pixel_loss(generated, target)
        
        return self.lambda_gan * loss_gan + self.lambda_pixel * loss_pixel
