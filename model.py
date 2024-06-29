import torch
from torch import nn
import torch.nn.functional as F
from torch import nn, einsum
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,channel_attention=False):
        """定义BasicBlock残差块类

        参数：
            inplanes (int): 输入的Feature Map的通道数
            planes (int): 第一个卷积层输出的Feature Map的通道数
            stride (int, optional): 第一个卷积层的步长
            downsample (nn.Sequential, optional): 旁路下采样的操作
        注意：
            残差块输出的Feature Map的通道数是planes*expansion
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.channel_attention=channel_attention
        self.downsample =nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        self.stride = stride

        self.avg_pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.to_qk = nn.Sequential(nn.ConvTranspose1d(1,1,kernel_size=1,stride=1,padding=0))
        self.to_v = nn.Sequential(nn.ConvTranspose1d(1,1,kernel_size=3,stride=1,padding=1))
        self.norm=nn.LayerNorm(planes)


    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.channel_attention :
            x= self.downsample[1](self.downsample[0](x))
            out_channel = self.avg_pool(x).squeeze()
            out_channel  = self.norm(out_channel)
            q, k = out_channel .clone(), out_channel .clone()
            v = self.to_v(out_channel .unsqueeze(1)).squeeze()
            dots = einsum('b c, a c -> b a', q, k)/q.norm(p=2)
            dots = F.softmax(dots)
            identity = einsum('b a, a c -> b c', dots, v)
            out =out+ out * identity.unsqueeze(-1).unsqueeze(-1)
        else:
            identity= self.downsample[1](self.downsample[0](x))
            out=out+identity

        out = self.relu(out)

        return out


class SMESC(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2,2,2,2], num_classes=16, zero_init_residual=False,start_channel=200):
        super(SMESC, self).__init__()
        self.inplanes = start_channel//2  # 第一个残差块的输入通道数
        self.conv1 = nn.ConvTranspose2d(start_channel, start_channel//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(start_channel//2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # Stage1 ~ Stage4
        self.layer1 = self._make_layer(block, start_channel//2, layers[0],stride=1,channel_attention=False)
        self.layer2 = self._make_layer(block, start_channel//4, layers[1], stride=3,channel_attention=True)
        self.layer3 = self._make_layer(block, start_channel//8, layers[2], stride=2,channel_attention=True)
        self.layer4 = self._make_layer(block, num_classes, layers[3], stride=1,channel_attention=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        # 网络参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if  isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride,channel_attention):




        layers = []
        layers.append(block(self.inplanes, planes, stride,channel_attention))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x




