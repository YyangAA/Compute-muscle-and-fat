# -*- coding: utf-8 -*-

# @Time : 2025/2/13 22:35
# @Author : ---
# @File : ResU_MNet.py
# @Project: pytorch-medical-image-segmentation-master
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)     # 将y的 HW维度删去，变为[batch_size, channels]
        y = self.fc(y).view(b, c, 1, 1)
        return self.conv_1(x * y.expand_as(x))


'''
DSC
'''
class DSC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DSC, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.depth_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=2,
                                    padding=1, groups=in_ch, )
        # 逐深度卷积 groups=in_ch根据每个通道生成对应的卷积核
        self.point_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        # 驻点卷积 1*1卷积核 保证图像尺寸(H,W)不变 从而对通道进行融合
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):

        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = self.relu(x)
        return x


'''
LDA
'''
class LDA(nn.Module):  #注意任务中h与w相同
    def __init__(self, in_ch, ratio=16,pool_size=None):
        super(LDA, self).__init__()

        self.Strip_Pool1 = nn.AdaptiveAvgPool2d([1, None]) #高度h 变为1,宽度保持不变  C*1*W
        self.Strip_Pool2 = nn.AdaptiveAvgPool2d([None, 1])       # C*H*1

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, in_ch//ratio, 1),
            nn.BatchNorm1d(in_ch//ratio),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch//ratio, in_ch, 1),
            # nn.BatchNorm2d(in_ch),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, h, w = x.size()
        x1 = self.Strip_Pool1(x)  # [1, w]
        x2 = self.Strip_Pool2(x)  # [h, 1]
        x1 = x1.view(b, c, w)     # [b,c,w]
        x2 = x2.view(b, c, h)     # [b,c,h]
        x3 = torch.cat([x1, x2], dim=-1)  #[b,c,(w+h)]
        x3 = self.conv1(x3)   # [b,c/radio,(w+h)]
        x4 = x3[:, :, 0:h]    # [b,c/radio,w]
        x5 = x3[:, :, h:]     # [b,c/radio,h]
        x4 = x4.unsqueeze(dim=-2)    # [b,c/radio,1,w]
        x5 = x5.unsqueeze(dim=-1)    # [b,c/radio,h,1]
        x4 = self.conv2(x4)    # [b,c,1,w]
        x5 = self.conv2(x5)    # [b,c,h,1]
        x = x * x4 * x5        # 对于通道维度不匹配的，torch会通过广播机制自动匹配相同维度 即x4与x5均会变成[b,c,h,w]的形状
        return x



class CLA(nn.Module):
    def __init__(self, in_ch, ratio=16,pool_size=None):
        super(CLA, self).__init__()

        self.Strip_Pool1 = nn.AdaptiveAvgPool2d([1, None])
        self.Strip_Pool2 = nn.AdaptiveAvgPool2d([None, 1])

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, in_ch//ratio, 1),
            nn.BatchNorm1d(in_ch//ratio),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch//ratio, in_ch, 1),
            # nn.BatchNorm2d(in_ch),
            nn.Sigmoid()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                # nn.AdaptiveAvgPool2d(1),
                nn.Linear(in_ch, 2 * in_ch, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(2 * in_ch, 2 * in_ch, bias=True),
                nn.ReLU(inplace=True)
        )
    def forward(self, x):

        b, c, h, w = x.size()

        x1 = self.Strip_Pool1(x)  # [1, w]
        x2 = self.Strip_Pool2(x)  # [h, 1]

        x1 = x1.view(b, c, w)
        x2 = x2.view(b, c, h)

        x3 = torch.cat([x1, x2], dim=-1)
        x3 = self.conv1(x3)

        x4 = x3[:, :, 0:h]
        x5 = x3[:, :, h:]
        x4 = x4.unsqueeze(dim=-2)
        x5 = x5.unsqueeze(dim=-1)
        x4 = self.conv2(x4)
        x5 = self.conv2(x5)
        x = x * x4 * x5
        '''
        类似SE模块；
        1. 为了将通道注意力权重赋予给下一层，通道进行变化
        2. 常规的CBAM模块顺序为：先加入通道，再加入空间，为了保留空间信息，
        此处调换顺序，且在通道注意力中不需要Sigmoid，目的是保留原本的空间信息
        '''
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, 2 * c, 1, 1)
        return y

# class conv_block(nn.Module):
#     '''
#     卷积层，（卷积+BN+RU）* 2
#     '''
#     def __init__(self, in_ch, out_ch):
#         super(conv_block, self).__init__()
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             # Strip_Pool(out_ch)
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#     def forward(self, x):
#         return self.conv(x)

class conv_block(nn.Module):
    '''
    卷积层，（卷积+BN+RU）* 2
    '''
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
        )
        self.skip_conv = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1,padding=1, bias=False),
        nn.BatchNorm2d(out_ch)  #为何没采用1*1卷积的模式
        )

    def forward(self, x):
        return self.conv(x) + self.skip_conv(x)

class conv_blockfirst(nn.Module):
    '''
    卷积层，（卷积+BN+RU）* 2
    '''
    def __init__(self, in_ch, out_ch):
        super(conv_blockfirst, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
        )
        self.skip_conv = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1,padding=1, bias=False),
        nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return self.conv(x) + self.skip_conv(x)



class up_conv(nn.Module):
    '''
    上采样，factor = 2 + BN + RELU
    '''
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.up(x)

class CLA_MNet(nn.Module):
    def __init__(self, in_ch=3, classses=1):
        super(CLA_MNet, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]   # 通道数为 32 64 128 256 512

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_blockfirst(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up5_conv = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up4_conv = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up3_conv = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up2_conv = conv_block(filters[1], filters[0])
        # CLA---------------------
        self.A_1 = CLA(32)
        self.A_2 = CLA(64)
        self.A_3 = CLA(128)
        self.A_4 = CLA(256)
        # ------------------------
        # LDA--------------------
        self.LDA_1 = LDA(32)
        self.LDA_2 = LDA(64)
        self.LDA_3 = LDA(128)
        self.LDA_4 = LDA(256)
        self.LDA_5 = LDA(512)
        # -------------------------
        # DSC---------------------
        self.DSC_1 = DSC(in_ch=32, out_ch=64)
        self.DSC_2 = DSC(in_ch=64, out_ch=128)
        self.DSC_3 = DSC(in_ch=128, out_ch=256)
        self.DSC_4 = DSC(in_ch=256, out_ch=512)
        # -------------------------------
        # SE-----------------------------------
        self.SE_1 = SELayer(channel=32)
        self.SE_2 = SELayer(channel=64)
        self.SE_3 = SELayer(channel=128)
        self.SE_4 = SELayer(channel=256)




        self.Conv = nn.Conv2d(filters[0], classses, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):                   # Suppose input is [BS, 1, 512, 512]

        e1 = self.Conv1(x)                  # [32, 512, 512]
        e1_LDA = self.LDA_1(e1)
        e1_DSC = self.DSC_1(e1_LDA)
        e1_skip = self.SE_1(e1_LDA)
        e2 = self.Maxpool1(e1)              # [32, 256, 256]

        a2 = self.A_1(e2)                   # Att[64, 256, 256]

        e2 = self.Conv2(e2)                 # [64, 256, 256]
        e2_LDA = self.LDA_2(e2)
        e2_DSC = self.DSC_2(e2_LDA)
        # print(e2_LDA.shape)
        # print(e2_DSC.shape)
        e2_multi = e2_LDA + e1_DSC
        e2_skip = self.SE_2(e2_multi)
        e3 = self.Maxpool2(e2)              # [64, 128, 128]

        # e3 = e3 * a2.expand_as(e3)          # 赋予注意力  ###
        a3 = self.A_2(e3)                   # Att[128, 128, 128]

        e3 = self.Conv3(e3)                 # [128, 128, 128]
        e3_LDA = self.LDA_3(e3)
        # print(e3_LDA.shape)
        e3_DSC = self.DSC_3(e3_LDA)
        e3_multi = e3_LDA + e2_DSC
        e3_skip = self.SE_3(e3_multi)
        e4 = self.Maxpool3(e3)              # [128, 64, 64]

        # e4 = e4 * a3.expand_as(e4)          # 赋予注意力   ###
        a4 = self.A_3(e4)                   #[256, 64, 64]

        e4 = self.Conv4(e4)                 # [256, 64, 64]
        e4_LDA = self.LDA_4(e4)
        e4_DSC = self.DSC_4(e4_LDA)
        e4_multi = e4_LDA + e3_DSC
        e4_skip = self.SE_4(e4_multi)
        e5 = self.Maxpool4(e4)              # [256, 32, 32]

        # e5 = e5 * a4.expand_as(e5)           ###
        # a5 = self.A_4(e5)

        e5 = self.Conv5(e5)                 # [512, 32, 32]
        e5_LDA = self.LDA_5(e5)
        e5_multi = e5_LDA + e4_DSC

        d5 = self.Up5(e5_multi)                   # [256, 32, 32]
        d5 = torch.cat((e4_skip, d5), dim=1)     # [512, 32, 32]
        d5 = self.Up5_conv(d5)              # [256, 32, 32]

        d4 = self.Up4(d5)                   # [128, 64, 64]
        d4 = torch.cat((e3_skip, d4), dim=1)     # [256, 64, 64]
        d4 = self.Up4_conv(d4)              # [128, 64, 64]

        d3 = self.Up3(d4)                   # [64, 128, 128]
        d3 = torch.cat((e2_skip, d3), dim=1)     # [128, 128, 128]
        d3 = self.Up3_conv(d3)              # [64, 128, 128]

        d2 = self.Up2(d3)                   # [32, 256, 256]
        d2 = torch.cat((e1_skip, d2), dim=1)     # [64, 256, 256]
        d2 = self.Up2_conv(d2)              # [32, 256, 256]

        out = self.Conv(d2)                 # [1, 256, 256]

        return out


# 第一次实验 去除CLA模块 检查实验效果 其中普通卷积块依然采用师兄之前的(Res_MNet)
# 第二次实验  去除CLA模块 其中普通卷积块加入之前去除的归一化和激活函数(ResU_MNet)
# 第三次实验   去除CLA模块 采用标准卷积 其中对于残差机制改kernel_size为1*1卷积
# 第四次实验   去除CLA模块  其中去除残差网络机制
# 第五次及以后的实验 保留cLA模块 重复上述步骤 找到最佳的实验组合效果
# 启用新损失函数进行实验



