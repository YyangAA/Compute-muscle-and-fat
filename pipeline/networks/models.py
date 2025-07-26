# encoding: utf-8

"""
The main CheXpert models implementation.
Including:
    DenseNet-121
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from . import densenet
from torchvision.models import resnet50
from transformers import ViTModel, ViTConfig
class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size, mode, drop_rate=0):
        super(DenseNet121, self).__init__()
        assert mode in ('U-Ones', 'U-Zeros', 'U-MultiClass')
        self.densenet121 = densenet.densenet121(pretrained=False, drop_rate=drop_rate)
        num_ftrs = self.densenet121.classifier.in_features  # 1024
        if mode in ('U-Ones', 'U-Zeros'):
            self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_ftrs, out_size),
                #nn.Sigmoid()
            )
        elif mode in ('U-MultiClass', ):
            self.densenet121.classifier = None
            self.densenet121.Linear_0 = nn.Linear(num_ftrs, out_size)
            self.densenet121.Linear_1 = nn.Linear(num_ftrs, out_size)
            self.densenet121.Linear_u = nn.Linear(num_ftrs, out_size)
            
        self.mode = mode
        
        # Official init from torch repo.
        for m in self.densenet121.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        
        
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)

        if self.drop_rate > 0:
            out = self.drop_layer(out)
        self.activations = out
        if self.mode in ('U-Ones', 'U-Zeros'):
            out = self.densenet121.classifier(out)
        elif self.mode in ('U-MultiClass', ):
            n_batch = x.size(0)
            out_0 = self.densenet121.Linear_0(out).view(n_batch, 1, -1)
            out_1 = self.densenet121.Linear_1(out).view(n_batch, 1, -1)
            out_u = self.densenet121.Linear_u(out).view(n_batch, 1, -1)
            out = torch.cat((out_0, out_1, out_u), dim=1)
            
        return self.activations, out

class DenseNet161(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size, mode, drop_rate=0):
        super(DenseNet161, self).__init__()
        assert mode in ('U-Ones', 'U-Zeros', 'U-MultiClass')
        self.densenet161 = densenet.densenet161(pretrained=True, drop_rate=drop_rate)
        num_ftrs = self.densenet161.classifier.in_features
        if mode in ('U-Ones', 'U-Zeros'):
            self.densenet161.classifier = nn.Sequential(
                nn.Linear(num_ftrs, out_size),
                #nn.Sigmoid()
            )
        elif mode in ('U-MultiClass', ):
            self.densenet161.classifier = None
            self.densenet161.Linear_0 = nn.Linear(num_ftrs, out_size)
            self.densenet161.Linear_1 = nn.Linear(num_ftrs, out_size)
            self.densenet161.Linear_u = nn.Linear(num_ftrs, out_size)
            
        self.mode = mode
        
        # Official init from torch repo.
        for m in self.densenet161.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        features = self.densenet161.features(x)
        out = F.relu(features, inplace=True)
        
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        
        if self.drop_rate > 0:
            out = self.drop_layer(out)
        self.activations = out
        
        if self.mode in ('U-Ones', 'U-Zeros'):
            out = self.densenet161.classifier(out)
        elif self.mode in ('U-MultiClass', ):
            n_batch = x.size(0)
            out_0 = self.densenet161.Linear_0(out).view(n_batch, 1, -1)
            out_1 = self.densenet161.Linear_1(out).view(n_batch, 1, -1)
            out_u = self.densenet161.Linear_u(out).view(n_batch, 1, -1)
            out = torch.cat((out_0, out_1, out_u), dim=1)
            
        return self.activations, out


class DenseNet169(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet169
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, out_size, mode, drop_rate=0):
        super(DenseNet169, self).__init__()
        assert mode in ('U-Ones', 'U-Zeros', 'U-MultiClass')
        self.densenet169 = densenet.densenet169(pretrained=True, drop_rate=drop_rate)
        num_ftrs = self.densenet169.classifier.in_features
        if mode in ('U-Ones', 'U-Zeros'):
            self.densenet169.classifier = nn.Sequential(
                nn.Linear(num_ftrs, out_size),
                # nn.Sigmoid()
            )
        elif mode in ('U-MultiClass',):
            self.densenet169.classifier = None
            self.densenet169.Linear_0 = nn.Linear(num_ftrs, out_size)
            self.densenet169.Linear_1 = nn.Linear(num_ftrs, out_size)
            self.densenet169.Linear_u = nn.Linear(num_ftrs, out_size)

        self.mode = mode

        # Official init from torch repo.
        for m in self.densenet169.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        features = self.densenet169.features(x)
        out = F.relu(features, inplace=True)

        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)

        if self.drop_rate > 0:
            out = self.drop_layer(out)
        self.activations = out

        if self.mode in ('U-Ones', 'U-Zeros'):
            out = self.densenet169.classifier(out)
        elif self.mode in ('U-MultiClass',):
            n_batch = x.size(0)
            out_0 = self.densenet169.Linear_0(out).view(n_batch, 1, -1)
            out_1 = self.densenet169.Linear_1(out).view(n_batch, 1, -1)
            out_u = self.densenet169.Linear_u(out).view(n_batch, 1, -1)
            out = torch.cat((out_0, out_1, out_u), dim=1)

        return self.activations, out


class ResNet50Custom(nn.Module):
    """Customized ResNet50.
    The architecture of this model is based on standard ResNet50,
    with modifications to the classifier layer and additional support for
    multi-class or multi-output modes.
    """

    def __init__(self, out_size, mode, drop_rate=0):
        super(ResNet50Custom, self).__init__()
        assert mode in ('U-Ones', 'U-Zeros', 'U-MultiClass')

        # Load the pretrained ResNet50 backbone
        self.resnet50 = resnet50(pretrained=True)
        num_ftrs = self.resnet50.fc.in_features

        # Modify the classifier layer
        if mode in ('U-Ones', 'U-Zeros'):
            self.resnet50.fc = nn.Sequential(
                nn.Linear(num_ftrs, out_size),
                # nn.Sigmoid()  # Uncomment if required for binary classification
            )
        elif mode in ('U-MultiClass',):
            self.resnet50.fc = None
            self.resnet50.Linear_0 = nn.Linear(num_ftrs, out_size)
            self.resnet50.Linear_1 = nn.Linear(num_ftrs, out_size)
            self.resnet50.Linear_u = nn.Linear(num_ftrs, out_size)

        self.mode = mode
        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

        # Official init from torch repo.
        for m in self.resnet50.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract features
        features = self.resnet50.conv1(x)
        features = self.resnet50.bn1(features)
        features = self.resnet50.relu(features)
        features = self.resnet50.maxpool(features)

        features = self.resnet50.layer1(features)
        features = self.resnet50.layer2(features)
        features = self.resnet50.layer3(features)
        features = self.resnet50.layer4(features)

        out = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)

        if self.drop_rate > 0:
            out = self.drop_layer(out)
        self.activations = out

        # Classifier logic
        if self.mode in ('U-Ones', 'U-Zeros'):
            out = self.resnet50.fc(out)
        elif self.mode in ('U-MultiClass',):
            n_batch = x.size(0)
            out_0 = self.resnet50.Linear_0(out).view(n_batch, 1, -1)
            out_1 = self.resnet50.Linear_1(out).view(n_batch, 1, -1)
            out_u = self.resnet50.Linear_u(out).view(n_batch, 1, -1)
            out = torch.cat((out_0, out_1, out_u), dim=1)

        return self.activations, out


class VisionTransformer(nn.Module):
    """
    Model modified for partial fine-tuning.
    Unfreeze the last layer of ViT and classifier head for training.
    """

    def __init__(self, out_size, mode, drop_rate=0, freeze_vit=True):
        super(VisionTransformer, self).__init__()
        assert mode in ('U-Ones', 'U-Zeros', 'U-MultiClass')

        # Load ViT model
        config = ViTConfig.from_pretrained("google/vit-small-patch16-224-in21k", num_labels=out_size)
        self.vit = ViTModel.from_pretrained("google/vit-small-patch16-224-in21k", config=config)

        # Freeze ViT parameters except the last transformer block
        if freeze_vit:
            for name, param in self.vit.named_parameters():
                param.requires_grad = False  # Freeze all params initially
            # 解冻最后一层Transformer block的参数
            for name, param in self.vit.encoder.layer[-1].named_parameters():
                param.requires_grad = True

        # Extract hidden size from the transformer
        hidden_size = self.vit.config.hidden_size

        # 自定义分类层，用于三分类
        self.classifier = nn.Linear(hidden_size, out_size)
        self.classifier.requires_grad_(True)  # 确保分类头是解冻的

        # Mode-specific custom classifier
        if mode in ('U-Ones', 'U-Zeros'):
            self.vit.classifier = nn.Sequential(
                nn.Linear(hidden_size, out_size)
            )
        elif mode in ('U-MultiClass',):
            self.vit.classifier = None
            self.vit.Linear_0 = nn.Linear(hidden_size, out_size)
            self.vit.Linear_1 = nn.Linear(hidden_size, out_size)
            self.vit.Linear_u = nn.Linear(hidden_size, out_size)

        self.mode = mode
        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

        # 初始化分类层的偏置
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        # Extract features from ViT encoder
        outputs = self.vit(x)
        features = outputs.last_hidden_state[:, 0, :]  # CLS token output

        # Apply dropout if specified
        if self.drop_rate > 0:
            features = self.drop_layer(features)

        self.activations = features

        # Custom classifier head
        if self.mode in ('U-Ones', 'U-Zeros'):
            out = self.vit.classifier(features)
        elif self.mode in ('U-MultiClass',):
            n_batch = x.size(0)
            out_0 = self.vit.Linear_0(features).view(n_batch, 1, -1)
            out_1 = self.vit.Linear_1(features).view(n_batch, 1, -1)
            out_u = self.vit.Linear_u(features).view(n_batch, 1, -1)
            out = torch.cat((out_0, out_1, out_u), dim=1)
        else:
            out = self.classifier(features)

        return self.activations, out
import torch
from torch import nn
from transformers import SwinModel, SwinConfig


class SwinTransformerModel(nn.Module):
    """
    Swin Transformer model modified for partial fine-tuning.
    Unfreeze the last layer of Swin and classifier head for training.
    """

    def __init__(self, out_size, mode, drop_rate=0, freeze_swin=True):
        super(SwinTransformerModel, self).__init__()
        assert mode in ('U-Ones', 'U-Zeros', 'U-MultiClass')

        # Load Swin-T model
        # config = SwinConfig.from_pretrained("microsoft/swin-tiny-patch4-window7-224", num_labels=out_size)
        # self.swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224", config=config)
        # 指定本地路径
        local_swin_path = "./local_swin_tiny"

        # 从本地路径加载配置和模型
        config = SwinConfig.from_pretrained(local_swin_path, num_labels=out_size)
        self.swin = SwinModel.from_pretrained(local_swin_path, config=config)
        # Freeze Swin parameters except the last layer
        if freeze_swin:
            for name, param in self.swin.named_parameters():
                param.requires_grad = False  # Freeze all params initially
            # 解冻最后一层Transformer block的参数
            for name, param in self.swin.encoder.named_parameters():
                # if "layers.3" in name:  # 解冻最后一个stage
                #     param.requires_grad = True
                if "layers.2" in name or "layers.3" in name or "layers.1" in name :  # 解冻最后两个 stage
                    param.requires_grad = True



        # Extract hidden size from the Swin model
        hidden_size = self.swin.config.hidden_size  # Swin-T 的隐藏层维度为 768

        # 自定义分类层，用于三分类
        self.classifier = nn.Linear(hidden_size, out_size)
        self.classifier.requires_grad_(True)  # 确保分类头是解冻的

        # Mode-specific custom classifier
        if mode in ('U-Ones', 'U-Zeros'):
            self.swin.classifier = nn.Sequential(
                nn.Linear(hidden_size, out_size)
            )
        elif mode in ('U-MultiClass',):
            self.swin.classifier = None
            self.swin.Linear_0 = nn.Linear(hidden_size, out_size)
            self.swin.Linear_1 = nn.Linear(hidden_size, out_size)
            self.swin.Linear_u = nn.Linear(hidden_size, out_size)

        self.mode = mode
        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

        # 初始化分类层的偏置
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        # Extract features from Swin encoder
        outputs = self.swin(x)
        pooled_output = outputs.pooler_output  # Swin 输出的池化特征

        # Apply dropout if specified
        if self.drop_rate > 0:
            pooled_output = self.drop_layer(pooled_output)

        self.activations = pooled_output

        # Custom classifier head
        if self.mode in ('U-Ones', 'U-Zeros'):
            out = self.swin.classifier(pooled_output)
        elif self.mode in ('U-MultiClass',):
            n_batch = x.size(0)
            out_0 = self.swin.Linear_0(pooled_output).view(n_batch, 1, -1)
            out_1 = self.swin.Linear_1(pooled_output).view(n_batch, 1, -1)
            out_u = self.swin.Linear_u(pooled_output).view(n_batch, 1, -1)
            out = torch.cat((out_0, out_1, out_u), dim=1)
        else:
            out = self.classifier(pooled_output)

        return self.activations, out
