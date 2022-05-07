#引用pytorch自定义模型所需要的库
import torch
import torch.nn as nn
import MyLayer

# 定义pytorch模型
class HAmodel(nn.Module):
    def __init__(self):
        super(HAmodel, self).__init__()
        # 定义卷积层1，通道为64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # MyLayer的STRranch块
        self.STBranch = MyLayer.STBranch()
        # MyLayer的HABranch块
        self.HABranch = MyLayer.HABranch()
        # 四个MyLayer的FUBranch块
        self.FUBranch1 = MyLayer.FUBranch()
        self.FUBranch2 = MyLayer.FUBranch()
        self.FUBranch3 = MyLayer.FUBranch()
        self.FUBranch4 = MyLayer.FUBranch()

    # 定义前向传播
    def forward(self, x):
        # 卷积层1
        x = self.conv1(x)
        # 将x分别进行HABranch和STBranch，之后进行concate
        x = torch.cat((self.HABranch(x), self.STBranch(x)), 1)
        #经过4次FUBranch，产生4个输出
        x1 = self.FUBranch1(x)
        x2 = self.FUBranch2(x)
        x3 = self.FUBranch3(x)
        x4 = self.FUBranch4(x)
        # 返回结果
        return x1, x2, x3, x4

    # 定义计算特征层的维度
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
