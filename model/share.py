import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
from torch.distributions.normal import Normal
from torch.autograd import Variable
class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out
class Encoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=4):
        super(Encoder, self).__init__()

        c = first_out_channel

        self.conv0 = nn.Sequential(
            ConvBlock(in_channel, c),
            ConvInsBlock(c, 2*c),
            ConvInsBlock(2*c, 2*c)
        )

        self.conv1 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(2 * c, 4 * c),
            ConvInsBlock(4 * c, 4 * c)
        )

        self.conv2 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(4 * c, 8 * c),
            ConvInsBlock(8 * c, 8 * c)
        )

        self.conv3 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(8 * c, 16* c),
            ConvInsBlock(16 * c, 16 * c)
        )

        self.conv4 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(16 * c, 32 * c),
            ConvInsBlock(32 * c, 32 * c)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        out4 = self.conv4(out3)  # 1/8
        # print(out4.shape)

        return (out0, out1, out2, out3 , out4)

# class ShareEnc(nn.Module):
#
#     def __init__(self, in_channel=1, channels=4):
#         super(ShareEnc, self).__init__()
#         self.channels = channels
#
#         self.encoder = Encoder(in_channel=in_channel, first_out_channel=self.channels)
#
#
#     def forward(self, moving, fixed):
#         # encode stage
#         Mouts = self.encoder(moving)
#         Fouts = self.encoder(fixed)
#         return Mouts,Fouts











if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShareEnc(inshape=(160, 192, 160), in_channel=1, channels=4).to(device)
    model.eval()

    # 构造随机 3D 图像对
    moving = torch.randn(1, 1, 160, 192, 160).to(device)
    fixed = torch.randn(1, 1, 160, 192, 160).to(device)

    with torch.no_grad():
        m_outs, f_outs = model(moving, fixed)

    print("✅ 前向传播成功！各级特征图尺寸如下：")
    for lvl, (m, f) in enumerate(zip(m_outs, f_outs)):
        print(f"Level {lvl}: moving={tuple(m.shape)}, fixed={tuple(f.shape)}")