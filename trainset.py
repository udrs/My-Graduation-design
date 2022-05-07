from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# 自己写一个Dataset类，用来加载训练数据集
class TrainSet(Dataset):
    #传入训练数据集的路径
    def __init__(self, root_dir, transform=None):
        # 记录路径
        
        self.root_dir = root_dir
        self.transform = transform
        # 获取根目录下的所有图片的路径与名字
        self.imgs = os.listdir(root_dir)
    # 返回数据集的大小
    def __len__(self):
        return len(self.imgs)

    # 返回数据集中的元素
    def __getitem__(self, idx):
        # 读取对应的图片并输出
        img_name = self.imgs[idx]
        # 读取图片
        img_path = os.path.join(self.root_dir, img_name)
        # 读取图片
        image = Image.open(img_path)
        # 将图片横向分为五个区域
        image1 = image.crop((0, 0, image.width//5, image.height))
        image2 = image.crop((image.width//5, 0, image.width//5*2, image.height))
        image3 = image.crop((image.width//5*2, 0, image.width//5*3, image.height))
        image4 = image.crop((image.width//5*3, 0, image.width//5*4, image.height))
        image5 = image.crop((image.width//5*4, 0, image.width, image.height))
        # 将拆分的图片填入列表方便传递
        images = [image1, image2, image3, image4, image5]
        # 如果存在transform，则进行transform
        if self.transform:
            images = [self.transform(image) for image in images]
        return images

# 定义一个函数用来测试dataset类
def test_dataset():
    # 定义训练数据集的路径
    root_dir = 'D:/BRDF/svbrdf/Data/Data_Deschaintre18/train_part'
    # 定义训练数据集的变换
    transform = transforms.Compose([
        transforms.ToTensor()])
    # 实例化训练数据集
    trainset = TrainSet(root_dir, transform)
    # 输出数据集的大小
    print(len(trainset))
    # 实例化训练数据集的加载器
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
    # 循环训练数据集
    for i, datas in enumerate(trainloader):
        # 遍历datas并显示到屏幕上
        for data in datas:
            # 显示图片到屏幕上
            img = data[0].numpy()
            img = img.transpose(1, 2, 0)
            img = img * 255
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.show()

            
# 图片渲染

test_dataset()