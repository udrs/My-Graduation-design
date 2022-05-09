from cv2 import log
import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision.models import vgg16
import environment as env
import utils



# 自定义advModel
class advModel(nn.Module):
    def __init__(self):
        super(advModel, self).__init__()
        # 定义全局判别器，输出为两个分类结果
        self.classifierxDG = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1296, 1),
            nn.Sigmoid()
        )
        # 定义局部判别器，输出为两个分类结果
        self.classifierxDL = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1296, 1),
            nn.Sigmoid()
        )
        ################################################################
        # 定义VGG16的卷积层
        ################################################################
        # 加载VGG16模型
        self.vgg = vgg16(pretrained=False).eval().cuda()
        # 加载模型参数
        self.vgg.load_state_dict(torch.load('./vgg16-397923af.pth'))
        # 注册hook函数，获取模型的“conv3_pool”层
        features = list(self.vgg.children())[0]
        hook_layer = features[16]
        hook_layer.register_forward_hook(self.hookFeature)
        # 卷积块，激活函数为ReLU，卷积核大小为3*3，输出通道数为64
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, y):
        ###全局判别器
        xDG=0
        # # VGG16的卷积层
        # self.vgg(x)
        # # 卷积层的输出
        # self.vgg16_feature = self.conv(self.vgg16_feature)
        # # 全局判别器
        # xDG = self.classifierxDG(self.vgg16_feature)
        ###局部判别器
        # 获取遮罩后的图像
        x = self.mask(x) * x
        # VGG16的卷积层
        self.vgg(x)
        # 卷积层的输出
        self.vgg16_feature = self.conv(self.vgg16_feature)
        # 局部判别器
        xDL = self.classifierxDL(self.vgg16_feature)
        # 返回结果
        return [xDG, xDL]

    # hook函数，获取中间层的特征图输出
    def hookFeature(self, module, inp, outp):
        self.vgg16_feature = outp

    # 用来对图像产生遮罩
    def mask(self, x):
        # 计算像素点的平均值
        mean = torch.mean(x)
        # 将0.92*mean的值作为阈值过滤
        mask = torch.where(mean > 0.92 * mean, torch.ones_like(mean), torch.zeros_like(mean))
        # 返回遮罩
        return mask


class SVBRDFL1Loss(nn.Module):
    def forward(self, input, target):

        input_normals,  input_diffuse,  input_roughness,  input_specular  = utils.unpack_svbrdf(input)
        target_normals, target_diffuse, target_roughness, target_specular = utils.unpack_svbrdf(target)

        input_normals = torch.clamp(input_normals, -1, 1)
        input_diffuse = torch.clamp(input_diffuse, 0, 1)
        input_roughness = torch.clamp(input_roughness, 0, 1)
        input_specular = torch.clamp(input_specular, 0, 1)


        epsilon_l1      = 1e-5#0.01
        input_diffuse   = torch.log(input_diffuse   + epsilon_l1)
        input_specular  = torch.log(input_specular  + epsilon_l1)
        target_diffuse  = torch.log(target_diffuse  + epsilon_l1)
        target_specular = torch.log(target_specular + epsilon_l1)

        return nn.functional.l1_loss(input_normals, target_normals) + nn.functional.l1_loss(input_diffuse, target_diffuse) + nn.functional.l1_loss(input_roughness, target_roughness) + nn.functional.l1_loss(input_specular, target_specular)

class RenderingLoss(nn.Module):
    def __init__(self, renderer):
        super(RenderingLoss, self).__init__()
        
        self.renderer = renderer
        self.random_configuration_count   = 3
        self.specular_configuration_count = 6

    def forward(self, input, target):
        batch_size = input.shape[0]

        batch_input_renderings = []
        batch_target_renderings = []
        for i in range(batch_size):
            scenes = env.generate_random_scenes(self.random_configuration_count) + env.generate_specular_scenes(self.specular_configuration_count)
            input_svbrdf  = input[i]
            target_svbrdf = target[i]
            input_renderings  = []
            target_renderings = []
            for scene in scenes:
                input_renderings.append(self.renderer.render(scene, input_svbrdf))
                target_renderings.append(self.renderer.render(scene, target_svbrdf))
            batch_input_renderings.append(torch.cat(input_renderings, dim=0))
            batch_target_renderings.append(torch.cat(target_renderings, dim=0))

        epsilon_render    = 1e-5#0.1
        batch_input_renderings_logged  = torch.log(torch.stack(batch_input_renderings, dim=0)  + epsilon_render)
        batch_target_renderings_logged = torch.log(torch.stack(batch_target_renderings, dim=0) + epsilon_render)

        loss = nn.functional.l1_loss(batch_input_renderings_logged, batch_target_renderings_logged)

        return loss


class MyLoss(nn.Module):
    def __init__(self,renderer):
        super(MyLoss, self).__init__()
        # L1 norm
        #self.loss_func = nn.L1Loss()
        #self.loss_func = MixedLoss()
        self.l1_loss        = SVBRDFL1Loss()
        self.rendering_loss = RenderingLoss(renderer)
    def forward(self, x, y, dis=None):
        dis_loss=0
        # 判断是否有判别器损失
        if dis is not None:
            # 计算目标图像均值
            mean = torch.mean(y)
            # 计算判别器损失
            #dis_loss = mean-torch.log(1-dis[1])
            
            dis_loss = (mean-torch.log(1-dis[0]))+(mean-torch.log(1-dis[1].cpu()))
            dis_loss = dis_loss.mean()
        # 计算损失
        #loss = self.loss_func(x, y)
        return 100*self.l1_loss+10*self.rendering_loss +1*dis_loss

