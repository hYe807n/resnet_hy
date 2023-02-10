import torch
import torch.nn as nn

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

def conv3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d : 

    """3x3 convolution with padding"""

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size = 3,
        stride = stride,
        padding = dilation,
        groups= groups,
        bias = False,
        dilation=dilation,
    )

def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d :

    """1x1 convolution"""

    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size = 1, 
        stride = stride, 
        bias = False,
    )

def activation_func(activation):

    """List of activation functions. """

    return nn.ModuleDict([
        ["relu", nn.ReLU(inplace=True)],
        ["leaky_relu", nn.LeakyReLU(negative_slope = 0.01, inplace=True)],
        ["selu", nn.SELU(inplace=True)],
        ["none", nn.Identity()]
    ])[activation]

class BasicBlock(nn.Module) :

    expansion: int = 1

    def __init__(
            self, 
            inplanes: int, 
            planes: int, 
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            norm_layer = None,
        ) :
        super(BasicBlock, self).__init__() 
        if (norm_layer is None) :
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride) 
        self.bn1 = norm_layer(planes)
        self.activation = activation_func("relu")
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        identity = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if (self.downsample is not None) :
            identity = self.downsample(x)
        
        out = out + identity 
        out = self.activation(out)
        return out
    
class BottleNeck(nn.Module) :

    expansion: int = 4 # For BottleNeck block, output channels expansion(64, 64 -> 256).

    def __init__(
        self, 
        inplanes: int, 
        planes: int, 
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) :
        super(BottleNeck, self).__init__() 
        if (norm_layer is None) :
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, planes) 
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion) 
        self.bn3 = norm_layer(planes * self.expansion)
        self.activation = activation_func("relu")
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        identity = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if (self.downsample is not None) :
            identity = self.downsample(x)

        out = out + identity
        out = self.activation(out)

        return out
    
class ResNet(nn.Module) :

    __inplanes = 64

    def __init__(
        self, 
        block: Type[Union[BasicBlock, BottleNeck]],
        layer_counts: List[int],
        num_classes: int = 1000, # num_classes : It is used to implement fully connected layer. 
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) :
        super(ResNet, self).__init__()
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv2d(
            3, # rgb.
            self.__inplanes, 
            kernel_size = 7, 
            stride = 2, 
            padding = 3, 
            bias = False
        )
        self.bn1 = self.norm_layer(self.__inplanes)
        self.activation = activation_func("relu")
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self.__make_layer(block, layer_counts[0], 64, stride = 1)
        self.layer2 = self.__make_layer(block, layer_counts[1], 128, stride = 2)
        self.layer3 = self.__make_layer(block, layer_counts[2], 256, stride = 2)
        self.layer4 = self.__make_layer(block, layer_counts[3], 512, stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def __make_layer(self, block: Type[Union[BasicBlock, BottleNeck]], 
                     layer_count: int, planes: int, stride = 1) :
        norm_layer = self.norm_layer 
        downsample = None 
        if((stride != 1) or (self.__inplanes != planes * block.expansion)) :
            downsample = nn.Sequential(
                conv1x1(self.__inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = list()
        layers.append(block(self.__inplanes, planes, stride = stride, 
                            downsample = downsample, norm_layer = norm_layer))
        self.__inplanes = planes * block.expansion
        for _ in range(0, layer_count - 1) :
            layers.append(block(self.__inplanes, planes, norm_layer = norm_layer))

        return nn.Sequential(*layers)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor :
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        return self._forward_impl(x)
    
def ResNet18(num_classes, norm_layer) :
    return ResNet(block = BasicBlock, layer_counts = [2, 2, 2, 2], num_classes = num_classes, norm_layer = norm_layer)

def ResNet34(num_classes, norm_layer) :
    return ResNet(block = BasicBlock, layer_counts = [3, 4, 6, 3], num_classes = num_classes, norm_layer = norm_layer)

def ResNet50(num_classes, norm_layer) :
    return ResNet(block = BottleNeck, layer_counts = [3, 4, 6, 3], num_classes = num_classes, norm_layer = norm_layer)

def ResNet101(num_classes, norm_layer) :
    return ResNet(block = BottleNeck, layer_counts = [3, 4, 23, 3], num_classes = num_classes, norm_layer = norm_layer)

def ResNet152(num_classes, norm_layer) :
    return ResNet(block = BottleNeck, layer_counts = [3, 8, 36, 3], num_classes = num_classes, norm_layer = norm_layer)