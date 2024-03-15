import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.nn import Parameter


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1, stride=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.gn1 = nn.GroupNorm(8, out_planes)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                               dilation=dilation)
        self.gn2 = nn.GroupNorm(8, out_planes)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride),
                nn.GroupNorm(8, out_planes),
                nn.ReLU(out_planes)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act_func(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.act_func(out)
        out = out + identity

        return out


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=dilation,
                               dilation=dilation)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=dilation,
                               dilation=dilation)
        self.gn2 = nn.GroupNorm(8, out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.act_func(out)
        return out


"""============================ASPP==============================="""


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.act_func = act_func
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(8, in_channels),
            self.act_func)
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.GroupNorm(8, in_channels),
            self.act_func)
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.GroupNorm(8, in_channels),
            self.act_func)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        cat = torch.add(conv1, conv2, alpha=1)
        cat = torch.add(cat, conv3, alpha=1)
        cat = torch.add(cat, identity, alpha=1)
        # cat = torch.cat([conv1, conv2,conv3,identity], 1)
        out = self.conv1x1(cat)
        return out


class ASPP1(nn.Module):
    def __init__(self, in_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.act_func = act_func
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(8, out_channels),
            self.act_func)
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.GroupNorm(8, out_channels),
            self.act_func)
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.GroupNorm(8, out_channels),
            self.act_func)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        identity = self.conv1x1(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        cat = torch.add(conv1, conv2, alpha=1)
        cat = torch.add(cat, conv3, alpha=1)
        cat = torch.add(cat, identity, alpha=1)
        # cat = torch.cat([conv1, conv2,conv3,identity], 1)
        return cat


class ASPP1_(nn.Module):
    def __init__(self, in_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.act_func = act_func
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(8, out_channels),
            self.act_func)
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.GroupNorm(8, out_channels),
            self.act_func)
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.GroupNorm(8, out_channels),
            self.act_func)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv4 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(8, out_channels),
            self.act_func)

    def forward(self, x):
        identity = self.conv1x1(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        cat = torch.add(conv1, conv2, alpha=1)
        cat = torch.add(cat, conv3, alpha=1)
        cat = torch.add(cat, identity, alpha=1)
        out = self.conv4(cat)
        # cat = torch.cat([conv1, conv2,conv3,identity], 1)
        return out


class ASPP1ASPP1(nn.Module):

    def __init__(self, in_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.sapp1 = ASPP1(in_channels, out_channels, act_func)
        self.sapp1sapp1 = ASPP1(out_channels, out_channels, act_func)


    def forward(self, x):
        sapp1 = self.sapp1(x)
        out = self.sapp1sapp1(sapp1)
        return out


class ASPP2(nn.Module):
    def __init__(self, in_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.act_func = act_func
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(8, out_channels),
            self.act_func)
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(8, out_channels),
            self.act_func,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.GroupNorm(8, out_channels),
            self.act_func)
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(8, out_channels),
            self.act_func,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.GroupNorm(8, out_channels),
            self.act_func,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.GroupNorm(8, out_channels),
            self.act_func)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.conv1x1(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        cat = torch.add(conv1, conv2, alpha=1)
        cat = torch.add(cat, conv3, alpha=1)
        cat = torch.add(cat, identity, alpha=1)
        # cat = torch.cat([conv1, conv2,conv3,identity], 1)
        return cat


class ASPP3(nn.Module):
    def __init__(self, in_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.act_func = act_func
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(8, in_channels),
            self.act_func)
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(8, in_channels),
            self.act_func,
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.GroupNorm(8, in_channels),
            self.act_func)
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(8, in_channels),
            self.act_func,
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(8, in_channels),
            self.act_func,
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.GroupNorm(8, in_channels),
            self.act_func)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        cat = torch.add(conv1, conv2, alpha=1)
        cat = torch.add(cat, conv3, alpha=1)
        cat = torch.add(cat, identity, alpha=1)
        # cat = torch.cat([conv1, conv2,conv3,identity], 1)
        out = self.conv1x1(cat)
        return out


class ASPP4(nn.Module):
    def __init__(self, in_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.act_func = act_func
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(8, out_channels),
            self.act_func)
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(8, out_channels),
            self.act_func,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.GroupNorm(8, out_channels),
            self.act_func)
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(8, out_channels),
            self.act_func,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(8, out_channels),
            self.act_func,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.GroupNorm(8, out_channels),
            self.act_func)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.conv1x1(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        cat = torch.add(conv1, conv2, alpha=1)
        cat = torch.add(cat, conv3, alpha=1)
        cat = torch.add(cat, identity, alpha=1)
        # cat = torch.cat([conv1, conv2,conv3,identity], 1)
        return cat

class ASPP5(nn.Module):
    def __init__(self, in_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.act_func = act_func
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(8, out_channels),
            self.act_func)
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.GroupNorm(8, out_channels),
            self.act_func)
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.GroupNorm(8, out_channels),
            self.act_func)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        identity = self.conv1x1(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        cat = torch.add(conv1, conv2, alpha=1)
        cat = torch.add(cat, conv3, alpha=1)
        cat = torch.add(cat, identity, alpha=1)
        # cat = torch.cat([conv1, conv2,conv3,identity], 1)
        return cat

class ASPP6(nn.Module):
    def __init__(self, in_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.act_func = act_func
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(8, out_channels),
            self.act_func)
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.GroupNorm(8, out_channels),
            self.act_func)
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.GroupNorm(8, out_channels),
            self.act_func)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        identity = self.conv1x1(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        cat = torch.add(conv1, conv2, alpha=1)
        cat = torch.add(cat, conv3, alpha=1)
        cat = torch.add(cat, identity, alpha=1)
        # cat = torch.cat([conv1, conv2,conv3,identity], 1)
        return cat

class ASPP1ASPP6(nn.Module):

    def __init__(self, in_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.sapp1 = ASPP1(in_channels, out_channels, act_func)
        self.sapp1sapp6= ASPP6(out_channels, out_channels, act_func)


    def forward(self, x):
        sapp1 = self.sapp1(x)
        out = self.sapp1sapp6(sapp1)
        return out


class ASPP5ASPP1(nn.Module):

    def __init__(self, in_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.sapp5 = ASPP5(in_channels, out_channels, act_func)
        self.sapp5sapp1= ASPP1(out_channels, out_channels, act_func)


    def forward(self, x):
        sapp5 = self.sapp5(x)
        out = self.sapp5sapp1(sapp5)
        return out


"""=================ASPP+self attention============================="""
class ASPPSelfAtt(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.sapp = ASPP(in_channels, out_channels, act_func)
        self.self_att = SelfAttBlock(out_channels, dilation, act_func)

    def forward(self, x):
        sapp = self.sapp(x)
        out = self.self_att(sapp)
        return out


class ASPP1SelfAtt(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.sapp = ASPP1(in_channels, out_channels, act_func)
        self.self_att = SelfAttBlock(out_channels, dilation, act_func)

    def forward(self, x):
        sapp = self.sapp(x)
        out = self.self_att(sapp)
        return out


class ASPP2SelfAtt(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.sapp = ASPP2(in_channels, out_channels, act_func)
        self.self_att = SelfAttBlock(out_channels, dilation, act_func)

    def forward(self, x):
        sapp = self.sapp(x)
        out = self.self_att(sapp)
        return out


class ASPP1SelfAtt2(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.sapp = ASPP1(in_channels, out_channels, act_func)
        self.self_att = SelfAtt2Block(out_channels, dilation, act_func)

    def forward(self, x):
        sapp = self.sapp(x)
        out = self.self_att(sapp)
        return out


class ASPP1SelfAtt3(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.sapp = ASPP1(in_channels, out_channels, act_func)
        self.self_att = SelfAtt3Block(out_channels, dilation, act_func)

    def forward(self, x):
        sapp = self.sapp(x)
        out = self.self_att(sapp)
        return out

class ASPP1SelfAtt144(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.sapp = ASPP1(in_channels, out_channels, act_func)
        self.self_att = SelfAtt144Block(out_channels, dilation, act_func)

    def forward(self, x):
        sapp = self.sapp(x)
        out = self.self_att(sapp)
        return out

class ASPP1SelfAtt112(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.sapp = ASPP1(in_channels, out_channels, act_func)
        self.self_att = SelfAtt112Block(out_channels, dilation, act_func)

    def forward(self, x):
        sapp = self.sapp(x)
        out = self.self_att(sapp)
        return out

class ASPP3SelfAtt(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.sapp = ASPP3(in_channels, out_channels, act_func)
        self.self_att = SelfAttBlock(out_channels, dilation, act_func)

    def forward(self, x):
        sapp = self.sapp(x)
        out = self.self_att(sapp)
        return out


class ASPP4SelfAtt(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.sapp = ASPP4(in_channels, out_channels, act_func)
        self.self_att = SelfAttBlock(out_channels, dilation, act_func)

    def forward(self, x):
        sapp = self.sapp(x)
        out = self.self_att(sapp)
        return out


class ASPP5SelfAtt(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.sapp = ASPP5(in_channels, out_channels, act_func)
        self.self_att = SelfAttBlock(out_channels, dilation, act_func)

    def forward(self, x):
        sapp = self.sapp(x)
        out = self.self_att(sapp)
        return out

class ASPP1ASPP1SelfAtt(nn.Module):

    def __init__(self, in_channels, out_channels, dilation=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.sapp1 = ASPP1(in_channels, out_channels, act_func)
        self.sapp1sapp1 = ASPP1(out_channels, out_channels, act_func)
        self.self_att = SelfAttBlock(out_channels, dilation, act_func)

    def forward(self, x):
        sapp1 = self.sapp1(x)
        out = self.sapp1sapp1(sapp1)
        out = self.self_att(out)
        return out


"""=================self attention============================="""


class SelfAttBlock(nn.Module):
    def __init__(self, in_planes, dilation=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.act_func = act_func
        self.conv1_1 = nn.Conv3d(in_planes, in_planes // 2, kernel_size=1, stride=1)
        self.GAP = nn.AdaptiveAvgPool3d(1)
        self.conv1 = nn.Conv3d(in_planes // 2, in_planes // 2, kernel_size=3, stride=1, padding=dilation,
                               dilation=dilation)
        self.Sig = nn.Sigmoid()
        self.k4 = nn.Sequential(
            nn.Conv3d(in_planes // 2, in_planes // 2, kernel_size=3, stride=1, padding=dilation,
                      dilation=dilation),
            nn.GroupNorm(8, in_planes // 2),
            self.act_func)
        self.k1 = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // 2, kernel_size=1, stride=1),
            nn.Conv3d(in_planes // 2, in_planes // 2, kernel_size=3, stride=1, padding=dilation, dilation=dilation))

    def forward(self, x):
        k3 = self.conv1_1(x)
        identity = k3
        k3 = self.Sig(identity + self.GAP(k3))
        k3 = k3 * self.conv1(identity)
        out = torch.cat([self.k4(k3), self.k1(x)], 1)

        return out


class SelfAtt1Block(nn.Module):
    def __init__(self, in_planes, dilation=1, stride=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.act_func = act_func
        self.conv1_1 = nn.Conv3d(in_planes, in_planes // 2, kernel_size=1, stride=1)
        self.GAP = nn.AdaptiveAvgPool3d(1)
        self.conv1 = nn.Conv3d(in_planes // 2, in_planes // 2, kernel_size=3, stride=1, padding=dilation,
                               dilation=dilation)
        self.Sig = nn.Sigmoid()
        self.k4 = Parameter(torch.zeros(1))
        self.k1 = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // 2, kernel_size=1, stride=1),
            nn.Conv3d(in_planes // 2, in_planes // 2, kernel_size=3, stride=1, padding=dilation, dilation=dilation))

    def forward(self, x):
        k3 = self.conv1_1(x)
        identity = k3
        k3 = self.Sig(identity + self.GAP(k3))
        k3 = k3 * self.conv1(identity)
        out = torch.cat([self.k4 * k3, self.k1(x)], 1)

        return out


class SelfAtt2Block(nn.Module):
    def __init__(self, in_planes, dilation=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.act_func = act_func
        self.conv1_1 = nn.Conv3d(in_planes, in_planes, kernel_size=1, stride=1)
        self.GAP = nn.AdaptiveAvgPool3d(1)
        self.conv1 = nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.Sig = nn.Sigmoid()
        self.k4 = nn.Sequential(
            nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=1, padding=dilation,
                      dilation=dilation),
            nn.GroupNorm(8, in_planes),
            self.act_func)
        self.k1 = nn.Sequential(
            nn.Conv3d(in_planes, in_planes, kernel_size=1, stride=1),
            nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation))

    def forward(self, x):
        k3 = self.conv1_1(x)
        identity = k3
        k3 = self.Sig(identity + self.GAP(k3))
        k3 = k3 * self.conv1(identity)
        out = torch.cat([self.k4(k3), self.k1(x)], 1)

        return out


class SelfAtt3Block(nn.Module):
    def __init__(self, in_planes, dilation=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.act_func = act_func
        self.conv1_1 = nn.Conv3d(in_planes, 192, kernel_size=1, stride=1)
        self.GAP = nn.AdaptiveAvgPool3d(1)
        self.conv1 = nn.Conv3d(192, 192, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.Sig = nn.Sigmoid()
        self.k4 = nn.Sequential(
            nn.Conv3d(192, in_planes, kernel_size=3, stride=1, padding=dilation,
                      dilation=dilation),
            nn.GroupNorm(8, in_planes),
            self.act_func)
        self.k1 = nn.Sequential(
            nn.Conv3d(in_planes, 192, kernel_size=1, stride=1),
            nn.Conv3d(192, in_planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation))

    def forward(self, x):
        k3 = self.conv1_1(x)
        identity = k3
        k3 = self.Sig(identity + self.GAP(k3))
        k3 = k3 * self.conv1(identity)
        out = torch.cat([self.k4(k3), self.k1(x)], 1)

        return out

class SelfAtt144Block(nn.Module):
    def __init__(self, in_planes, dilation=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.act_func = act_func
        self.conv1_1 = nn.Conv3d(in_planes, 144, kernel_size=1, stride=1)
        self.GAP = nn.AdaptiveAvgPool3d(1)
        self.conv1 = nn.Conv3d(144, 144, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.Sig = nn.Sigmoid()
        self.k4 = nn.Sequential(
            nn.Conv3d(144, in_planes, kernel_size=3, stride=1, padding=dilation,
                      dilation=dilation),
            nn.GroupNorm(8, in_planes),
            self.act_func)
        self.k1 = nn.Sequential(
            nn.Conv3d(in_planes, 144, kernel_size=1, stride=1),
            nn.Conv3d(144, in_planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation))

    def forward(self, x):
        k3 = self.conv1_1(x)
        identity = k3
        k3 = self.Sig(identity + self.GAP(k3))
        k3 = k3 * self.conv1(identity)
        out = torch.cat([self.k4(k3), self.k1(x)], 1)

        return out

class SelfAtt112Block(nn.Module):
    def __init__(self, in_planes, dilation=1, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.act_func = act_func
        self.conv1_1 = nn.Conv3d(in_planes, 112, kernel_size=1, stride=1)
        self.GAP = nn.AdaptiveAvgPool3d(1)
        self.conv1 = nn.Conv3d(112, 112, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.Sig = nn.Sigmoid()
        self.k4 = nn.Sequential(
            nn.Conv3d(112, in_planes, kernel_size=3, stride=1, padding=dilation,
                      dilation=dilation),
            nn.GroupNorm(8, in_planes),
            self.act_func)
        self.k1 = nn.Sequential(
            nn.Conv3d(in_planes, 112, kernel_size=1, stride=1),
            nn.Conv3d(112, in_planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation))

    def forward(self, x):
        k3 = self.conv1_1(x)
        identity = k3
        k3 = self.Sig(identity + self.GAP(k3))
        k3 = k3 * self.conv1(identity)
        out = torch.cat([self.k4(k3), self.k1(x)], 1)

        return out


"""=================spatial attention============================="""


class SABlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv3d(F_g, F_int, kernel_size=1, stride=1)
        self.W_x = nn.Conv3d(F_l, F_int, kernel_size=1, stride=1)
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class SA1Block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv3d(F_g, F_int, kernel_size=1, stride=1)
        self.W_x = nn.Conv3d(F_l, F_int, kernel_size=1, stride=1)
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return g * psi + x


"""=================channel attention============================="""


class CABlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv3d(F_g, F_int, kernel_size=1, stride=1)
        self.W_x = nn.Conv3d(F_l, F_int, kernel_size=1, stride=1)
        self.psi = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(F_int, F_int // 16, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(F_int // 16, F_int, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)

        return x * psi


class CA1Block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv3d(F_g, F_int // 16, kernel_size=1, stride=1)
        self.W_x = nn.Conv3d(F_l, F_int // 16, kernel_size=1, stride=1)
        self.con_g = nn.Conv3d(F_g, F_int, kernel_size=1, stride=1)
        self.GAP = nn.AdaptiveAvgPool3d(1)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(F_int // 16, F_int, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.GAP(g)
        x1 = self.GAP(x)
        g2 = self.con_g(g)
        g1 = self.W_g(g1)
        x1 = self.W_x(x1)
        psi = self.psi(g1 + x1)

        return g2 * psi + x


"""=====================dual pathway spatial & channel attention========================="""


class DABlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.spatial_att = SABlock(F_g, F_l, F_int)
        self.channel_att = CABlock(F_g, F_l, F_int)
        self.combine_att = nn.Sequential(
            nn.Conv3d(F_int * 2, F_int, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(8, F_int),
            nn.ReLU(inplace=True))

    def forward(self, g, x):
        spatial_att = self.spatial_att(g, x)
        channel_att = self.channel_att(g, x)
        combine_att = self.combine_att(torch.cat([spatial_att, channel_att], 1))
        return combine_att


class DA1Block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.spatial_att = SA1Block(F_g, F_l, F_int)
        self.channel_att = CA1Block(F_g, F_l, F_int)
        self.combine_att = nn.Sequential(
            nn.Conv3d(F_int * 2, F_int, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(8, F_int),
            nn.ReLU(inplace=True))

    def forward(self, g, x):
        spatial_att = self.spatial_att(g, x)
        channel_att = self.channel_att(g, x)
        combine_att = self.combine_att(torch.cat([spatial_att, channel_att], 1))
        return combine_att


class DADropBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.spatial_att = SABlock(F_g, F_l, F_int)
        self.channel_att = CABlock(F_g, F_l, F_int)
        self.combine_att = nn.Sequential(
            nn.Conv3d(F_int * 2, F_int, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(8, F_int),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1))

    def forward(self, g, x):
        spatial_att = self.spatial_att(g, x)
        channel_att = self.channel_att(g, x)
        combine_att = self.combine_att(torch.cat([spatial_att, channel_att], 1))
        return combine_att


class DA1DropBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.spatial_att = SA1Block(F_g, F_l, F_int)
        self.channel_att = CA1Block(F_g, F_l, F_int)
        self.combine_att = nn.Sequential(
            nn.Conv3d(F_int * 2, F_int, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(8, F_int),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1))

    def forward(self, g, x):
        spatial_att = self.spatial_att(g, x)
        channel_att = self.channel_att(g, x)
        combine_att = self.combine_att(torch.cat([spatial_att, channel_att], 1))
        return combine_att


"""========================================================"""
class Baseline(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3])

        self.conv2_2 = VGGBlock(nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class UNetSA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.satt0 = SABlock(nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.satt1 = SABlock(nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.satt2 = SABlock(nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3])

        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        up3 = self.up(x3_0)
        satt2 = self.satt2(up3, x2_0)
        x2_2 = self.conv2_2(torch.cat([satt2, up3], 1))
        up2 = self.up(x2_2)
        satt1 = self.satt1(up2, x1_0)
        x1_3 = self.conv1_3(torch.cat([satt1, up2], 1))
        up1 = self.up(x1_3)
        satt0 = self.satt0(up1, x0_0)
        x0_4 = self.conv0_4(torch.cat([satt0, up1], 1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNetDA1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.datt0 = DA1Block(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.datt1 = DA1Block(nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.datt2 = DA1Block(nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3])

        self.conv2_2 = VGGBlock(nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        up3 = self.up3(x3_0)
        datt2 = self.datt2(up3, x2_0)
        x2_2 = self.conv2_2(torch.add(datt2, up3, alpha=1))
        up2 = self.up2(x2_2)
        datt1 = self.datt1(up2, x1_0)
        x1_3 = self.conv1_3(torch.add(datt1, up2, alpha=1))
        up1 = self.up1(x1_3)
        datt0 = self.datt0(up1, x0_0)
        x0_4 = self.conv0_4(torch.add(datt0, up1, alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


"""======================UNet+ASPP+SelfAtt=================================="""
class UNetSelfAtt1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAtt1Block(nb_filter[2])
        self.conv1_3 = SelfAtt1Block(nb_filter[1])
        self.conv0_4 = SelfAtt1Block(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNetSelfAtt(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2], 2)
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNetSelfAtt0(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2])
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNetASPPSelfAtt(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPPSelfAtt(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2])
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

"""======================UNet+ASPP1+SelfAtt=================================="""
class UNetASPP1SelfAtt(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP1SelfAtt(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2])
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNetASPP1SelfAtt0(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP1SelfAtt(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2], 2)
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNetASPP1SelfAtt2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(2 * nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP1SelfAtt2(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2], 2)
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNetASPP1SelfAtt2_0(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(2 * nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP1SelfAtt2(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2])
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNetASPP1_SelfAtt(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP1_(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2])
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNetASPP1SelfAtt3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(2 * nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP1SelfAtt3(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2], 2)
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNetASPP1SelfAtt3_0(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(2 * nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP1SelfAtt3(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2])
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNetASPP1SelfAtt144(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(2 * nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP1SelfAtt144(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2], 2)
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNetASPP1SelfAtt112(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(2 * nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP1SelfAtt112(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2], 2)
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class UNetASPP1ASPP1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP1ASPP1(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2],2)
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class UNetASPP1ASPP1_0(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP1ASPP1(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2])
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class UNetASPP1ASPP1SelfAtt(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP1ASPP1SelfAtt(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2],2)
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class UNetASPP1ASPP1SelfAtt_0(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP1ASPP1SelfAtt(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2])
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class UNetASPP1ASPP6(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP1ASPP6(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2],2)
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class UNetASPP1ASPP6_0(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP1ASPP6(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2])
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class UNetASPP5ASPP1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP5ASPP1(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2],2)
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class UNetASPP5ASPP1_0(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP5ASPP1(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2])
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

        self._initialize_weights()

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class UNetASPP5ASPP1_VGG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP5ASPP1(nb_filter[2], nb_filter[3])

        self.conv2_2 = VGGBlock(nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

        self._initialize_weights()

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class ResUNetASPP5ASPP1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = BasicBlock(4, nb_filter[0])
        self.conv1_0 = BasicBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = BasicBlock(nb_filter[1], nb_filter[2],2)
        self.conv3_0 = ASPP5ASPP1(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2],2)
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ResUNetASPP5ASPP1_0(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = BasicBlock(4, nb_filter[0])
        self.conv1_0 = BasicBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = BasicBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP5ASPP1(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2])
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

"""======================UNet+ASPP2+SelfAtt=================================="""
class UNetASPP2_0SelfAtt(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], 2)
        self.conv3_0 = ASPP2SelfAtt(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2], 2)
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNetASPP2_0SelfAtt01(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP2SelfAtt(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2], 2)
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
"""======================UNet+ASPP3+SelfAtt=================================="""
class UNetASPP3SelfAtt(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP3SelfAtt(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2],2)
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class UNetASPP3SelfAtt_0(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP3SelfAtt(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2])
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

"""======================UNet+ASPP4+SelfAtt=================================="""
class UNetASPP4SelfAtt(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP4SelfAtt(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2],2)
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class UNetASPP4SelfAtt_0(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP4SelfAtt(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2])
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

"""======================UNet+ASPP5+SelfAtt=================================="""
class UNetASPP5SelfAtt(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP5SelfAtt(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2],2)
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class UNetASPP5SelfAtt_0(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ASPP54SelfAtt(nb_filter[2], nb_filter[3])

        self.conv2_2 = SelfAttBlock(nb_filter[2])
        self.conv1_3 = SelfAttBlock(nb_filter[1])
        self.conv0_4 = SelfAttBlock(nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



"""==================UNetdilate============================"""


class UNetdilate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.conv1_0 = BasicBlock(in_planes=nb_filter[0], out_planes=nb_filter[1], dilation=1, stride=2)
        self.conv2_0 = BasicBlock(in_planes=nb_filter[1], out_planes=nb_filter[2], dilation=2, stride=2)
        self.conv3_0 = BasicBlock(in_planes=nb_filter[2], out_planes=nb_filter[3], dilation=4, stride=2)

        self.conv2_2 = BasicBlock(nb_filter[2], nb_filter[2], 2)
        self.conv1_3 = BasicBlock(nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)

        x2_2 = self.conv2_2(torch.add(x2_0, self.up3(x3_0), alpha=1))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up2(x2_2), alpha=1))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up1(x1_3), alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


'''UNetdilate+single pathway spatial attention(attention gate)'''


class UdilateSSA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.satt0 = SABlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_0 = BasicBlock(in_planes=nb_filter[0], out_planes=nb_filter[1], dilation=1, stride=2)
        self.satt1 = SABlock(nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv2_0 = BasicBlock(in_planes=nb_filter[1], out_planes=nb_filter[2], dilation=2, stride=2)
        self.satt2 = SABlock(nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv3_0 = BasicBlock(in_planes=nb_filter[2], out_planes=nb_filter[3], dilation=4, stride=2)

        self.conv2_2 = BasicBlock(nb_filter[2], nb_filter[2], 2)
        self.conv1_3 = BasicBlock(nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)

        up3 = self.up3(x3_0)
        satt2 = self.satt2(up3, x2_0)
        x2_2 = self.conv2_2(torch.add(satt2, up3, alpha=1))
        up2 = self.up2(x2_2)
        satt1 = self.satt1(up2, x1_0)
        x1_3 = self.conv1_3(torch.add(satt1, up2, alpha=1))
        up1 = self.up1(x1_3)
        satt0 = self.satt0(up1, x0_0)
        x0_4 = self.conv0_4(torch.add(satt0, up1, alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UdilateSSA1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.satt0 = SA1Block(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_0 = BasicBlock(in_planes=nb_filter[0], out_planes=nb_filter[1], dilation=1, stride=2)
        self.satt1 = SA1Block(nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv2_0 = BasicBlock(in_planes=nb_filter[1], out_planes=nb_filter[2], dilation=2, stride=2)
        self.satt2 = SA1Block(nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv3_0 = BasicBlock(in_planes=nb_filter[2], out_planes=nb_filter[3], dilation=4, stride=2)

        self.conv2_2 = BasicBlock(nb_filter[2], nb_filter[2], 2)
        self.conv1_3 = BasicBlock(nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)

        up3 = self.up3(x3_0)
        satt2 = self.satt2(up3, x2_0)
        x2_2 = self.conv2_2(torch.add(satt2, up3, alpha=1))
        up2 = self.up2(x2_2)
        satt1 = self.satt1(up2, x1_0)
        x1_3 = self.conv1_3(torch.add(satt1, up2, alpha=1))
        up1 = self.up1(x1_3)
        satt0 = self.satt0(up1, x0_0)
        x0_4 = self.conv0_4(torch.add(satt0, up1, alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


'''UNetdilate+single pathway channel attention(SE block)'''


class UdilateSCA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.catt0 = CABlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_0 = BasicBlock(in_planes=nb_filter[0], out_planes=nb_filter[1], dilation=1, stride=2)
        self.catt1 = CABlock(nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv2_0 = BasicBlock(in_planes=nb_filter[1], out_planes=nb_filter[2], dilation=2, stride=2)
        self.catt2 = CABlock(nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv3_0 = BasicBlock(in_planes=nb_filter[2], out_planes=nb_filter[3], dilation=4, stride=2)

        self.conv2_2 = BasicBlock(nb_filter[2], nb_filter[2], 2)
        self.conv1_3 = BasicBlock(nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)

        up3 = self.up3(x3_0)
        catt2 = self.catt2(up3, x2_0)
        x2_2 = self.conv2_2(torch.add(catt2, up3, alpha=1))
        up2 = self.up2(x2_2)
        catt1 = self.catt1(up2, x1_0)
        x1_3 = self.conv1_3(torch.add(catt1, up2, alpha=1))
        up1 = self.up1(x1_3)
        catt0 = self.catt0(up1, x0_0)
        x0_4 = self.conv0_4(torch.add(catt0, up1, alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UdilateSCA1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.catt0 = CA1Block(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_0 = BasicBlock(in_planes=nb_filter[0], out_planes=nb_filter[1], dilation=1, stride=2)
        self.catt1 = CA1Block(nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv2_0 = BasicBlock(in_planes=nb_filter[1], out_planes=nb_filter[2], dilation=2, stride=2)
        self.catt2 = CA1Block(nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv3_0 = BasicBlock(in_planes=nb_filter[2], out_planes=nb_filter[3], dilation=4, stride=2)

        self.conv2_2 = BasicBlock(nb_filter[2], nb_filter[2], 2)
        self.conv1_3 = BasicBlock(nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)

        up3 = self.up3(x3_0)
        catt2 = self.catt2(up3, x2_0)
        x2_2 = self.conv2_2(torch.add(catt2, up3, alpha=1))
        up2 = self.up2(x2_2)
        catt1 = self.catt1(up2, x1_0)
        x1_3 = self.conv1_3(torch.add(catt1, up2, alpha=1))
        up1 = self.up1(x1_3)
        catt0 = self.catt0(up1, x0_0)
        x0_4 = self.conv0_4(torch.add(catt0, up1, alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


'''UNetdilate+dual pathway spatial & channel attention'''


class UdilateDA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.datt0 = DABlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_0 = BasicBlock(in_planes=nb_filter[0], out_planes=nb_filter[1], dilation=1, stride=2)
        self.datt1 = DABlock(nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv2_0 = BasicBlock(in_planes=nb_filter[1], out_planes=nb_filter[2], dilation=2, stride=2)
        self.datt2 = DABlock(nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv3_0 = BasicBlock(in_planes=nb_filter[2], out_planes=nb_filter[3], dilation=4, stride=2)

        self.conv2_2 = BasicBlock(nb_filter[2], nb_filter[2], 2)
        self.conv1_3 = BasicBlock(nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)

        up3 = self.up3(x3_0)
        datt2 = self.datt2(up3, x2_0)
        x2_2 = self.conv2_2(torch.add(datt2, up3, alpha=1))
        up2 = self.up2(x2_2)
        datt1 = self.datt1(up2, x1_0)
        x1_3 = self.conv1_3(torch.add(datt1, up2, alpha=1))
        up1 = self.up1(x1_3)
        datt0 = self.datt0(up1, x0_0)
        x0_4 = self.conv0_4(torch.add(datt0, up1, alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UdilateDA1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.datt0 = DA1Block(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_0 = BasicBlock(in_planes=nb_filter[0], out_planes=nb_filter[1], dilation=1, stride=2)
        self.datt1 = DA1Block(nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv2_0 = BasicBlock(in_planes=nb_filter[1], out_planes=nb_filter[2], dilation=2, stride=2)
        self.datt2 = DA1Block(nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv3_0 = BasicBlock(in_planes=nb_filter[2], out_planes=nb_filter[3], dilation=4, stride=2)

        self.conv2_2 = BasicBlock(nb_filter[2], nb_filter[2], 2)
        self.conv1_3 = BasicBlock(nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)

        up3 = self.up3(x3_0)
        datt2 = self.datt2(up3, x2_0)
        x2_2 = self.conv2_2(torch.add(datt2, up3, alpha=1))
        up2 = self.up2(x2_2)
        datt1 = self.datt1(up2, x1_0)
        x1_3 = self.conv1_3(torch.add(datt1, up2, alpha=1))
        up1 = self.up1(x1_3)
        datt0 = self.datt0(up1, x0_0)
        x0_4 = self.conv0_4(torch.add(datt0, up1, alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


'''UNetdilate+dual pathway spatial & channel attention+dropout'''


class UdilateDAdrop(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.datt0 = DADropBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_0 = BasicBlock(in_planes=nb_filter[0], out_planes=nb_filter[1], dilation=1, stride=2)
        self.datt1 = DADropBlock(nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv2_0 = BasicBlock(in_planes=nb_filter[1], out_planes=nb_filter[2], dilation=2, stride=2)
        self.datt2 = DADropBlock(nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv3_0 = BasicBlock(in_planes=nb_filter[2], out_planes=nb_filter[3], dilation=4, stride=2)

        self.conv2_2 = BasicBlock(nb_filter[2], nb_filter[2], 2)
        self.conv1_3 = BasicBlock(nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)

        up3 = self.up3(x3_0)
        datt2 = self.datt2(up3, x2_0)
        x2_2 = self.conv2_2(torch.add(datt2, up3, alpha=1))
        up2 = self.up2(x2_2)
        datt1 = self.datt1(up2, x1_0)
        x1_3 = self.conv1_3(torch.add(datt1, up2, alpha=1))
        up1 = self.up1(x1_3)
        datt0 = self.datt0(up1, x0_0)
        x0_4 = self.conv0_4(torch.add(datt0, up1, alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UdilateDA1drop(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_filter = [32, 64, 128, 256]
        d, h, w = map(int, args.input_size.split(','))
        self.d = d
        self.h = h
        self.w = w
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up1 = nn.ConvTranspose3d(nb_filter[1], nb_filter[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose3d(nb_filter[2], nb_filter[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose3d(nb_filter[3], nb_filter[2], kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv0_0 = VGGBlock(4, nb_filter[0])
        self.datt0 = DA1DropBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_0 = BasicBlock(in_planes=nb_filter[0], out_planes=nb_filter[1], dilation=1, stride=2)
        self.datt1 = DA1DropBlock(nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv2_0 = BasicBlock(in_planes=nb_filter[1], out_planes=nb_filter[2], dilation=2, stride=2)
        self.datt2 = DA1DropBlock(nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv3_0 = BasicBlock(in_planes=nb_filter[2], out_planes=nb_filter[3], dilation=4, stride=2)

        self.conv2_2 = BasicBlock(nb_filter[2], nb_filter[2], 2)
        self.conv1_3 = BasicBlock(nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv3d(nb_filter[2], args.num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[1], args.num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], args.num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)

        up3 = self.up3(x3_0)
        datt2 = self.datt2(up3, x2_0)
        x2_2 = self.conv2_2(torch.add(datt2, up3, alpha=1))
        up2 = self.up2(x2_2)
        datt1 = self.datt1(up2, x1_0)
        x1_3 = self.conv1_3(torch.add(datt1, up2, alpha=1))
        up1 = self.up1(x1_3)
        datt0 = self.datt0(up1, x0_0)
        x0_4 = self.conv0_4(torch.add(datt0, up1, alpha=1))

        if self.args.deepsupervision:
            output1 = self.final1(x2_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x0_4)
            output1 = F.interpolate(output1, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(int(self.d), int(self.h), int(self.w)),
                                    mode='trilinear', align_corners=True)
            return [output1, output2, output3]

        else:
            output = self.final(x0_4)
            return output
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

