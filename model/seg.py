import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers"""
    
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        return self.act2(out)
    
class UpBlock(nn.Module):
    """Upsampling block with transposed convolution"""
    
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res_block = ResidualBlock(out_channels * 2, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        # Center crop skip connection to match x dimensions
        diffZ = skip.size()[2] - x.size()[2]
        diffY = skip.size()[3] - x.size()[3]
        diffX = skip.size()[4] - x.size()[4]
        
        skip = F.pad(skip, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2,
            diffZ // 2, diffZ - diffZ // 2
        ])
        
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x

class bottle(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(bottle,self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self,x):

        out = self.act(self.norm(self.conv(x)))

        return out


class SegDecoder(nn.Module):
    def __init__(self, num_classes=2, base_filters=16):
        super().__init__()
          # Decoder
        base_filters = base_filters*2
        self.bottle = bottle(base_filters * 16,base_filters * 16)

        self.up4 = UpBlock(base_filters * 16, base_filters * 8)
        self.up3 = UpBlock(base_filters * 8, base_filters * 4)
        self.up2 = UpBlock(base_filters * 4, base_filters * 2)
        self.up1 = UpBlock(base_filters * 2, base_filters)

        self.head5 = nn.Conv3d(base_filters * 16, num_classes, kernel_size=1)
        self.head4 = nn.Conv3d(base_filters * 8, num_classes, kernel_size=1)
        self.head3 = nn.Conv3d(base_filters * 4, num_classes, kernel_size=1)
        self.head2 = nn.Conv3d(base_filters * 2, num_classes, kernel_size=1)
        self.head1 = nn.Conv3d(base_filters, num_classes, kernel_size=1)

        # Output layer
        self.out_conv = nn.Conv3d(base_filters, num_classes, kernel_size=1)

    def forward(self, feats):
        x4,x3,x2,x1,x0 = feats[4], feats[3], feats[2],feats[1],feats[0]
        # print(x0.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # x = self.up4(x4, x3)
        # x = self.up3(x, x2)
        # x = self.up2(x, x1)
        # x = self.up1(x, x0)
        b  = self.bottle(x4)
        s5s = self.head5(b)

        s4 = self.up4(b, x3)
        s4s = self.head4(s4)
        s3 = self.up3(s4, x2)
        s3s = self.head3(s3)
        s2 = self.up2(s3, x1)
        s2s = self.head2(s2)
        s1 = self.up1(s2, x0)
        s1s = self.head1(s1)
        
        # Output
        s0 = self.out_conv(s1)
        # print('seg',s0.shape)
        # print(s1.shape)
        # print(s2.shape)
        # print(s3.shape)
        # print(s4.shape)
        lf  = [s1s,s2s,s3s,s4s,s5s]
        return s0,lf
        # return s0


class EncoderDecoder(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base_filters=16):
        super().__init__()
        self.encoder = ShareEnc(in_channel=in_channels,
                               channels=8)  # 4->16/4=4
        self.decoder = SegDecoder(num_classes=num_classes,
                                  base_filters=base_filters)

    def forward(self, x):
        feats,fest2 = self.encoder(x,x)
        seg = self.decoder(feats)
        return seg


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderDecoder(in_channels=1, num_classes=2, base_filters=16).to(device)
    model.eval()

    dummy = torch.randn(1, 1, 160, 192, 160).to(device)

    with torch.no_grad():
        out = model(dummy)

    print("✅ Encoder-Decoder 前向成功！")
    print(f"输入尺寸: {tuple(dummy.shape)}")
    print(f"输出尺寸: {tuple(out.shape)}  (B, C, D, H, W)")