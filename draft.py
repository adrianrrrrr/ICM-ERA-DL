import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Contracting path
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)
        
        # Expansive path
        self.upconv4 = self.upconv_block(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        self.dec1 = self.conv_block(128, 64)
        
        # Output layer
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # Contracting path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        enc5 = self.enc5(F.max_pool2d(enc4, 2))
        
        # Expansive path
        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        out = self.out_conv(dec1)
        return out

# Example usage
model = UNet()
input_image = torch.randn(1, 1, 256, 256)  # Batch size of 1, 1 channel, 256x256 image
output_image = model(input_image)
print(output_image.shape)  # Should be [1, 2, 256, 256]




'''
# This is the original paper architecture
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Contracting path
        self.encoder1 = self.contracting_block(in_channels, 64)
        self.encoder2 = self.contracting_block(64, 128)
        self.encoder3 = self.contracting_block(128, 256)
        self.encoder4 = self.contracting_block(256, 512)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Expansive path
        self.upconv4 = self.expansive_block(1024, 512, 512)
        self.upconv3 = self.expansive_block(1024, 256, 256)  # 1024 = 512 (upconv4 out_channels) + 512 (enc4 channels)
        self.upconv2 = self.expansive_block(512, 128, 128)  # 512 = 256 (upconv3 out_channels) + 256 (enc3 channels)
        self.upconv1 = self.expansive_block(256, 64, 64)  # 256 = 128 (upconv2 out_channels) + 128 (enc2 channels)
        
        # Final output
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)  # 128 = 64 (upconv1 out_channels) + 64 (enc1 channels)

    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )
        return block

    def expansive_block(self, in_channels, mid_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=2, stride=2),
        )
        return block

    def crop_and_concat(self, upsampled, bypass):
        crop_size = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-crop_size, -crop_size, -crop_size, -crop_size))
        return torch.cat((bypass, upsampled), 1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))
        
        # Decoder
        dec4 = self.crop_and_concat(self.upconv4(bottleneck), enc4)
        dec3 = self.crop_and_concat(self.upconv3(dec4), enc3)
        dec2 = self.crop_and_concat(self.upconv2(dec3), enc2)
        dec1 = self.crop_and_concat(self.upconv1(dec2), enc1)
        
        # Final output
        return self.final_conv(dec1)

# Ejemplo de uso:
if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1)
    input_tensor = torch.rand(1, 1, 572, 572)  # Batch size of 1, 1 channel, 572x572 image
    output = model(input_tensor)
    print(output.shape)  # Expected output: torch.Size([1, 1, 388, 388])

'''