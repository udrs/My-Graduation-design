import torch
from torch import nn

# 自定义STconv层
class STconv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding,dilation=1):
        super(STconv, self).__init__()
        # 二维卷积
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation)
        # 激活函数LeakyReLU
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        # 卷积
        x = self.conv(x)
        # 激活
        x = self.relu(x)
        return x

# 自定义inception层
class Inception(nn.Module):
    def __init__(self, in_channels, out_channels,padding,stride=1,dilation=1):
        super(Inception, self).__init__()
        self.out_channels = out_channels//2
        # 定义第一个3*3标准卷积层
        self.conv1 = STconv(in_channels=in_channels, out_channels=self.out_channels, kernel_size=3, stride=stride, padding=padding,dilation=dilation)
        # 定义两个连续的3*3标准卷积层
        self.conv2 = nn.Sequential(
            STconv(in_channels=in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=padding,dilation=dilation),
            STconv(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=stride, padding=padding,dilation=dilation)
        )

    def forward(self, x):
        # 对两个3*3卷积层进行拼接
        # print("@@@@@@@@@@@")
        # print(x.shape)
        # print(self.conv1(x).shape)
        # print(self.conv2(x).shape)
        # print("###############")
        x = torch.cat([self.conv1(x), self.conv2(x)], dim=1)
        return x

# 自定义HA卷积层
class HAconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,dilation=1):
        super(HAconv, self).__init__()
        # HA卷积块的最上层,先进行卷积，再进行sigmoid激活
        self.conv_top=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation),
            nn.Sigmoid()
        )
        # HA卷积块的中间层，先进行instance normalization，再进行卷积，再进行LeakyReLU激活
        self.conv_mid=nn.Sequential(
            nn.InstanceNorm2d(out_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # HA卷积块的最下层，也就是Inception层
        self.conv_bottom=Inception(in_channels=in_channels, out_channels=out_channels,stride=stride,padding=padding,dilation=dilation)

    def forward(self, x):
        # 将HA卷积最上层与中层先进行元素乘法
        y = self.conv_top(x) * self.conv_mid(x)
        # 将得到的结果y与经过最下层的卷积层进行元素加法
        y = y + self.conv_bottom(x)
        #  LeakyReLU（在论文5.1中提到每个块后面都跟随了一个LeakyReLU）
        y = nn.LeakyReLU(0.2, inplace=True)(y)
        return y



# 自定义MLP层
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        # 全连接层
        self.fc = nn.Linear(in_features=in_channels, out_features=out_channels)
        
    def forward(self, x):
        # 全连接层
        x = self.fc(x)
        # # 激活函数LeakyReLU(暂时未提到此处有激活函数)
        # x = nn.LeakyReLU(0.2, inplace=True)(x)
        return x
# 自定义HABranch层
class HABranch(nn.Module):
    def __init__(self):
        super(HABranch, self).__init__()
        # HA卷积层1、2、3、4，输出通道分别为64,64，64,128
        self.HAconv1 = nn.Sequential(
            HAconv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            HAconv(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            HAconv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            HAconv(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        )
        # HA卷积层5、6，输出通道分别为128,128
        self.HAconv2 = nn.Sequential(
            HAconv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            HAconv(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        )
        # HA卷积层7，输出通道为128，步长为1，空洞卷积系数为2
        self.HAconv3 = HAconv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2,dilation=2)
        # ST卷积层8，双线性插值，输出通道为128，步长为1
        self.HAconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            STconv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        )
        # ST卷积层9，双线性插值，输出通道为64，步长为1
        self.HAconv5 =nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            STconv(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
    def forward(self, x):
        # HA卷积块1
        x = self.HAconv1(x)
        # 复制x，并进行HA卷积块2
        y = x.clone()
        x = self.HAconv2(x)
        # 复制x，并进行HA卷积块3
        z = x.clone()
        x = self.HAconv3(x)
        # skip connection
        x = x + z
        # HA卷积块4
        x = self.HAconv4(x)
        # skip connection
        x = x + y
        # HA卷积块5
        x = self.HAconv5(x)
        return x

# 自定义STBranch层
class STBranch(nn.Module):
    def __init__(self):
        super(STBranch, self).__init__()
        # ST卷积，通道数为64,层1、2、3、4
        self.STconv1 = nn.Sequential(
            STconv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            STconv(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            STconv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            STconv(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
        )
        # ST卷积，通道数为128,层5、6
        self.STconv2 = nn.Sequential(
            STconv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            STconv(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        )
        # ST卷积，通道数为128,层7，步长为1,空洞卷积因子为2
        self.STconv3 = STconv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2,dilation=2)
        # ST卷积，通道数为128，双线性插值，步长为1
        self.STconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            STconv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        )
        # ST卷积，通道数为64，双线性插值，步长为1
        self.STconv5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            STconv(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # ST卷积，块1
        x = self.STconv1(x)
        # 复制x，准备skip connection
        x_copy = x.clone()
        # ST卷积，块2
        x = self.STconv2(x)
        # 复制x，准备skip connection
        x_copy2 = x.clone()
        # ST卷积，块3
        x = self.STconv3(x)
        # skip connection
        x = x + x_copy2
        # ST卷积，块4
        x = self.STconv4(x)
        # skip connection
        x = x + x_copy
        # ST卷积，块5
        x = self.STconv5(x)
        return x
# 自定义AFS层
class AFS(nn.Module):
    def __init__(self):
        super(AFS, self).__init__()
        # 全局均值池化
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # MLP层，输入通道数为128，输出通道数为128,单个隐藏层16个节点，激活函数为Sigmoid
        self.MLP = nn.Sequential(
            # 卷积层1，输入通道数为128，输出通道数为16，卷积核大小为1*1，步长为1，padding为0
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            # 卷积层2，输入通道数为16，输出通道数为128，卷积核大小为1*1，步长为1，padding为0
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            # 激活函数
            nn.Sigmoid()
        )
    def forward(self, x):
        # 全局均值池化
        y = self.global_avgpool(x)
        # MLP层
        y = self.MLP(y)
        # 将x与y相乘
        x = y * x
        return x

# 自定义FUBranch层
class FUBranch(nn.Module):
    def __init__(self):
        super(FUBranch, self).__init__()
        # AFS层
        self.AFS = AFS()
        # ST卷积，通道数为128
        self.STconv1 = STconv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # ST卷积，双线性插值，步长为1，通道为64
        self.STconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            STconv(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        # ST卷积,通道为64
        self.STconv3 = STconv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # ST卷积，通道数为3
        self.STconv4 = STconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        # AFS层
        x = self.AFS(x)
        # ST卷积
        x = self.STconv1(x)
        # ST卷积
        x = self.STconv2(x)
        # ST卷积
        x = self.STconv3(x)
        # ST卷积
        x = self.STconv4(x)
        # print(x.shape)
        return x




# if __name__ == '__main__':
#     # 打印网络结构进行测试
#     test = STBranch(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
#     print(test)
