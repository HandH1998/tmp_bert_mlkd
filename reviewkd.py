import math
import pdb
import torch.nn.functional as F
from torch import nn
import torch

# #from .mobilenetv2 import mobile_half
# from .shufflenetv1 import ShuffleV1
# from .shufflenetv2 import ShuffleV2
# from .resnet_cifar import build_resnet_backbone, build_resnetx4_backbone
# #from .vgg import build_vgg_backbone
# from .wide_resnet_cifar import wrn

import torch
from torch import nn
import torch.nn.functional as F



class ABF(nn.Module):
    def __init__(self, mid_channel):
        super(ABF, self).__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(mid_channel),
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(mid_channel, out_channel, kernel_size=3,
        #               stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channel),
        # )
        # if fuse:
        self.att_conv = nn.Sequential(
            nn.Conv2d(mid_channel*2, 2, kernel_size=1),
            nn.Sigmoid(),
        )
        # else:
        #     self.att_conv = None
        # nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        # nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        try:        
            n, _, h, w = x.shape
        except:
            x=torch.unsqueeze(x,1)
            y=torch.unsqueeze(y,1)
            n, _, h, w = x.shape
        # transform student features
        # x = self.conv1(x)
        # if self.att_conv is not None:
            # upsample residual features
            # 将高level的feature resize到与当前相同
            # y = F.interpolate(y, (shape, shape), mode="nearest")
        # fusion
        z = torch.cat([x, y], dim=1)
        z = self.att_conv(z)
        x = (x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w))
        x =torch.squeeze(x)
        # output
        # if x.shape[-1] != out_shape:
        #     # 这个x就是当前stage的融合特征，它的channel是mid_channel，shape和teacher的shape相同
        #     x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        # y = self.conv2(x)  # 将x的channel数和teacher的channel数相同
        return x


class ReviewKD(nn.Module):
    def __init__(
        self, mid_channel=12, student_layer_number=4,
    ):
        super(ReviewKD, self).__init__()
        # self.student = student
        # self.shapes = shapes
        # self.out_shapes = shapes if out_shapes is None else out_shapes

        abfs = nn.ModuleList()

        # mid_channel = min(512, in_channels[-1])
        # for idx, in_channel in enumerate(student_layer_number-1):
        for _ in range(student_layer_number-1):
            abfs.append(ABF(mid_channel))  # 最后一个student_feature不需要融合
        self.abfs = abfs[::-1]
        self.to('cuda')

    def forward(self, x):
        # student_features = self.student(x,is_feat=True)
        # logit = student_features[1]
        x = x[::-1]
        results = []
        out_features= x[0]
        results.append(out_features)
        for features, abf in zip(x[1:], self.abfs):
            out_features= abf(features, out_features)
            results.insert(0, out_features)
        return results


# def build_review_kd(model, num_classes, teacher = ''):
#     out_shapes = None
#     if 'x4' in model:
#         student = build_resnetx4_backbone(depth = int(model[6:-2]), num_classes = num_classes)
#         in_channels = [64,128,256,256]
#         out_channels = [64,128,256,256]
#         shapes = [1,8,16,32]
#     elif 'ResNet50' in model:
#         student = ResNet50(num_classes = num_classes)
#         in_channels = [16,32,64,64]
#         out_channels = [16,32,64,64]
#         shapes = [1,8,16,32,32]
#         assert False
#     elif 'resnet' in model:
#         student = build_resnet_backbone(depth = int(model[6:]), num_classes = num_classes)
#         in_channels = [16,32,64,64]
#         out_channels = [16,32,64,64]
#         shapes = [1,8,16,32,32]
#     elif 'vgg' in model:
#         student = build_vgg_backbone(depth = int(model[3:]), num_classes = num_classes)
#         in_channels = [128,256,512,512,512]
#         shapes = [1,4,4,8,16]
#         if 'ResNet50' in teacher:
#             out_channels = [256,512,1024,2048,2048]
#             out_shapes = [1,4,8,16,32]
#         else:
#             out_channels = [128,256,512,512,512]
#     elif 'mobile' in model:
#         student = mobile_half(num_classes = num_classes)
#         in_channels = [12,16,48,160,1280]
#         shapes = [1,2,4,8,16]
#         if 'ResNet50' in teacher:
#             out_channels = [256,512,1024,2048,2048]
#             out_shapes = [1,4,8,16,32]
#         else:
#             out_channels = [128,256,512,512,512]
#             out_shapes = [1,4,4,8,16]
#     elif 'shufflev1' in model:
#         student = ShuffleV1(num_classes = num_classes)
#         in_channels = [240,480,960,960]
#         shapes = [1,4,8,16]
#         if 'wrn' in teacher:
#             out_channels = [32,64,128,128]
#             out_shapes = [1,8,16,32]
#         else:
#             out_channels = [64,128,256,256]
#             out_shapes = [1,8,16,32]
#     elif 'shufflev2' in model:
#         student = ShuffleV2(num_classes = num_classes)
#         in_channels = [116,232,464,1024]
#         shapes = [1,4,8,16]
#         out_channels = [64,128,256,256]
#         out_shapes = [1,8,16,32]
#     elif 'wrn' in model:
#         student = wrn(depth=int(model[4:6]), widen_factor=int(model[-1:]), num_classes=num_classes)
#         r=int(model[-1:])
#         in_channels = [16*r,32*r,64*r,64*r]
#         out_channels = [32,64,128,128]
#         shapes = [1,8,16,32]
#     else:
#         assert False
#     backbone = ReviewKD(
#         student=student,
#         in_channels=in_channels,
#         out_channels=out_channels,
#         shapes = shapes,
#         out_shapes = out_shapes
#     )
#     return backbone

def hcl(fstudent, fteacher,hidden_size=312):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        try:
            n, c, h, w = fs.shape
        except:
            fs=torch.unsqueeze(fs,1)
            ft=torch.unsqueeze(ft,1)
            n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, hidden_size))
            tmpft = F.adaptive_avg_pool2d(ft, (l, hidden_size))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all
