import torch
import torch.nn as nn
from compared_models import square_activation, safe_log
import math
import torch.nn.functional as F


class WeightFreezing(nn.Module):
    def __init__(self, input_dim, output_dim, shared_ratio=0.3, multiple=0):
        super(WeightFreezing, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        mask = torch.rand(input_dim, output_dim) < shared_ratio
        self.register_buffer('shared_mask', mask)
        self.register_buffer('independent_mask', ~mask)

        self.multiple = multiple

    def forward(self, x, shared_weight):
        combined_weight = torch.where(self.shared_mask, shared_weight*self.multiple, self.weight.t())
        output = F.linear(x, combined_weight.t(), self.bias)
        return output


class ConvNetWeightFreezing(nn.Module):
    def __init__(self, num_classes=4, chans=22, samples=1125, shared_ratio=0.1):
        super(ConvNetWeightFreezing, self).__init__()
        self.conv_nums = 40
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, self.conv_nums, (1, 25)),
            nn.Conv2d(self.conv_nums, self.conv_nums, (chans, 1), bias=False),
            nn.BatchNorm2d(self.conv_nums)
        )
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout()

        out = torch.ones((1, 1, chans, samples))
        out = self.features(out)
        out = self.avgpool(out)
        n_out_time = out.cpu().data.numpy().shape  # [batch, self.conv_nums, 1, times]
        # share part weights
        shared_ratio = shared_ratio

        self.classifier = WeightFreezing(n_out_time[-1] * n_out_time[-2] * n_out_time[-3], num_classes,
                                         shared_ratio=shared_ratio)

        self.shared_weights = nn.Parameter(torch.Tensor(num_classes, n_out_time[-1] * n_out_time[-2] * n_out_time[-3]),
                                           requires_grad=False)
        self.bias = nn.Parameter(torch.Tensor(num_classes))

        nn.init.kaiming_uniform_(self.shared_weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.shared_weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        self.fixed_weight = self.shared_weights.t() * self.classifier.shared_mask

    def forward(self, x):
        x = self.features(x)
        x = square_activation(x)
        x = self.avgpool(x)
        x = safe_log(x)
        x = self.dropout(x)
        # x: [batch, 40, 1, times]
        features = torch.flatten(x, 1)  # 使用卷积网络代替全连接层进行分类, 因此需要返回x和卷积层个数

        cls = self.classifier(features, self.fixed_weight.to(features.device))

        return cls


class EEGNetWeightFreezing(nn.Module):
    def __init__(self, num_classes, chans, samples=1125, dropout_rate=0.5, kernel_length=64, F1=8,
                 F2=16, shared_ratio=0.1):
        super(EEGNetWeightFreezing, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1, kernel_size=(chans, 1), groups=F1, bias=False),  # groups=F1 for depthWiseConv
            nn.BatchNorm2d(F1),
            nn.ELU(inplace=True),
            # nn.ReLU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
            # for SeparableCon2D
            # SeparableConv2D(F1, F2, kernel1_size=(1, 16), bias=False),
            nn.Conv2d(F1, F1, kernel_size=(1, 16), groups=F1, bias=False),  # groups=F1 for depthWiseConv
            nn.BatchNorm2d(F1),
            nn.ELU(inplace=True),
            # nn.ReLU(),
            nn.Conv2d(F1, F2, kernel_size=(1, 1), groups=1, bias=False),  # point-wise cnn
            nn.BatchNorm2d(F2),
            # nn.ReLU(),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p=dropout_rate),
        )
        out = torch.ones((1, 1, chans, samples))
        out = self.features(out)
        n_out_time = out.cpu().data.numpy().shape
        # share part weights
        shared_ratio = shared_ratio

        self.classifier = WeightFreezing(n_out_time[-1] * n_out_time[-2] * n_out_time[-3], num_classes,
                                         shared_ratio=shared_ratio)

        self.shared_weights = nn.Parameter(torch.Tensor(num_classes, n_out_time[-1] * n_out_time[-2] * n_out_time[-3]),
                                           requires_grad=False)
        self.bias = nn.Parameter(torch.Tensor(num_classes))

        nn.init.kaiming_uniform_(self.shared_weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.shared_weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        self.fixed_weight = self.shared_weights.t() * self.classifier.shared_mask

    def forward(self, x):
        x = self.features(x)
        features = torch.flatten(x, 1)
        cls = self.classifier(features, self.fixed_weight.to(features.device))
        return cls


class LMDAWeightFreezing(nn.Module):

    def __init__(self, chans=22, samples=1125, num_classes=4, depth=9, kernel=75, channel_depth1=40, channel_depth2=40,
                ave_depth=1, avepool=5, shared_ratio=0.1):
        super(LMDAWeightFreezing, self).__init__()
        self.ave_depth = ave_depth
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.num_classes = num_classes

        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel),
                      groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )
        # self.avgPool1 = nn.AvgPool2d((1, 24))
        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(chans, 1), groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            # nn.AdaptiveAvgPool3d((9, 1, 35)),
            nn.Dropout(p=0.65),
        )

        # 定义自动填充模块
        out = torch.ones((1, 1, chans, samples))
        out = torch.einsum('bdcw, hdc->bhcw', out, self.channel_weight)
        out = self.time_conv(out)
        # out = self.avgPool1(out)
        out = self.chanel_conv(out)
        out = self.norm(out)
        n_out_time = out.cpu().data.numpy().shape
        print('In ShallowNet, n_out_time shape: ', n_out_time)
        # share part weights
        shared_ratio = shared_ratio

        self.classifier = WeightFreezing(n_out_time[-1] * n_out_time[-2] * n_out_time[-3], num_classes,
                                         shared_ratio=shared_ratio)

        self.shared_weights = nn.Parameter(torch.Tensor(num_classes, n_out_time[-1] * n_out_time[-2] * n_out_time[-3]),
                                           requires_grad=False)
        self.bias = nn.Parameter(torch.Tensor(num_classes))

        nn.init.kaiming_uniform_(self.shared_weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.shared_weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        self.fixed_weight = self.shared_weights.t() * self.classifier.shared_mask

    def EEGDepthAttention(self, x):
        # x: input features with shape [N, C, H, W]
        N, C, H, W = x.size()
        # K = W if W % 2 else W + 1
        k = 7
        adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k//2, 0), bias=True).to(x.device)  # original kernel k
        softmax = nn.Softmax(dim=-2)
        x_pool = adaptive_pool(x)
        x_transpose = x_pool.transpose(-2, -3)
        y = conv(x_transpose)
        y = softmax(y)
        y = y.transpose(-2, -3)
        return y * C * x

    def forward(self, x):
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)

        x_time = self.time_conv(x)  # batch, depth1, channel, samples_
        x_time = self.EEGDepthAttention(x_time)  # DA1

        x = self.chanel_conv(x_time)  # batch, depth2, 1, samples_
        x = self.norm(x)

        features = torch.flatten(x, 1)

        cls = self.classifier(features, self.fixed_weight.to(features.device))
        return cls



if __name__ == '__main__':
    a = torch.randn(32, 1, 22, 1125)
    model = ConvNetWeightFreezing()
    print(model(a).shape)
