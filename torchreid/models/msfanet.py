"""
MSFANet: Multi-Scale Focusing Attention Network (Huang et al., Appl. Sci. 2021, 11, 2010)
Re-implementation from the paper text/figures, in the style of Torchreid's OSNet.

Structure (Fig. 1): 7x7 conv -> avg-pool -> stage1 (1 MSFA, 256) -> transition
                    -> stage2 (2 MSFA, 384) -> transition -> stage3 (3 MSFA, 512)
                    -> 1x1 conv -> global avg-pool -> heads (FC/softmax, triplet)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['msfanet_123', 'msfanet_222']


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):  # 1x1 conv + BN, no activation
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class ChannelGate(nn.Module):
    """GAP -> FC(c, c/r) -> LeakyReLU -> FC(c/r, c) -> Sigmoid  (Fig. 3b).
    Used both as the SE block inside FA blocks and as the Aggregation Gate (AG)."""
    def __init__(self, c, reduction=16):
        super().__init__()
        r = max(c // reduction, 1)
        self.fc1 = nn.Conv2d(c, r, 1)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.fc2 = nn.Conv2d(r, c, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        return torch.sigmoid(self.fc2(self.act(self.fc1(w))))


class SEBlock(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        self.gate = ChannelGate(c, reduction)

    def forward(self, x):
        return x * self.gate(x)


class FABlock(nn.Module):
    """Focusing Attention block (Fig. 3a):
    1x1 conv -> 3x3 depthwise conv -> BN -> LeakyReLU -> SE."""
    def __init__(self, c, reduction=16):
        super().__init__()
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.se = SEBlock(c, reduction)

    def forward(self, x):
        return self.se(self.act(self.bn(self.dw(self.pw(x)))))


class MSFABlock(nn.Module):
    """Multi-Scale Focusing Attention block (Fig. 4, Eq. 1-3).
    Three branches with 1, 3, 5 FA blocks (receptive fields 3, 7, 11).
    Branch1 is averaged with the 1st FA output of branch2;
    branch2 is averaged with the 3rd FA output of branch3 (t < T);
    each is gated by an Aggregation Gate, summed, 1x1 conv, residual, LeakyReLU."""
    def __init__(self, cin, cout, bottleneck_reduction=4, se_reduction=16):
        super().__init__()
        mid = cout // bottleneck_reduction
        self.conv1 = ConvBNAct(cin, mid, 1)
        self.b1 = nn.ModuleList([FABlock(mid, se_reduction) for _ in range(1)])
        self.b2 = nn.ModuleList([FABlock(mid, se_reduction) for _ in range(3)])
        self.b3 = nn.ModuleList([FABlock(mid, se_reduction) for _ in range(5)])
        self.ag1 = ChannelGate(mid, se_reduction)
        self.ag2 = ChannelGate(mid, se_reduction)
        self.ag3 = ChannelGate(mid, se_reduction)
        self.conv3 = Conv1x1Linear(mid, cout)
        self.down = Conv1x1Linear(cin, cout) if cin != cout else None
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        identity = self.down(x) if self.down is not None else x
        x1 = self.conv1(x)

        a1 = self.b1[0](x1)

        b = [x1]
        for m in self.b2:
            b.append(m(b[-1]))           # b[1], b[2], b[3]

        c = [x1]
        for m in self.b3:
            c.append(m(c[-1]))           # c[1] ... c[5]

        m1 = 0.5 * (a1 + b[1])           # branch1 + 1st FA of branch2
        m2 = 0.5 * (b[3] + c[3])         # branch2 + 3rd FA of branch3
        m3 = c[5]

        agg = m1 * self.ag1(m1) + m2 * self.ag2(m2) + m3 * self.ag3(m3)
        out = self.conv3(agg) + identity
        return self.act(out)


class Transition(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = ConvBNAct(c, c, 1)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))


class MSFANet(nn.Module):
    def __init__(self, num_classes, blocks=(1, 2, 3), channels=(64, 256, 384, 512),
                 feature_dim=512, loss='softmax'):
        super().__init__()
        self.loss = loss
        self.feature_dim = feature_dim
        self.conv1 = ConvBNAct(3, channels[0], 7, stride=2, padding=3)
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)

        layers, cin = [], channels[0]
        for i, nb in enumerate(blocks):
            cout = channels[i + 1]
            for _ in range(nb):
                layers.append(MSFABlock(cin, cout))
                cin = cout
            if i < len(blocks) - 1:
                layers.append(Transition(cout))
        self.features = nn.Sequential(*layers)
        self.conv5 = ConvBNAct(cin, cin, 1)

        self.fc = self._make_fc(cin, feature_dim)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    @staticmethod
    def _make_fc(cin, dim):
        return nn.Sequential(nn.Linear(cin, dim), nn.BatchNorm1d(dim), nn.ReLU(inplace=True))

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_featuremaps=False):
        x = self.pool(self.conv1(x))
        x = self.conv5(self.features(x))
        if return_featuremaps:
            return x
        v = F.adaptive_avg_pool2d(x, 1).flatten(1)
        v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        raise KeyError(f"Unsupported loss: {self.loss}")


def msfanet_123(num_classes=1000, loss='softmax', pretrained=False, **kw):
    return MSFANet(num_classes, blocks=(1, 2, 3), loss=loss, **kw)


def msfanet_222(num_classes=1000, loss='softmax', pretrained=False, **kw):
    return MSFANet(num_classes, blocks=(2, 2, 2), loss=loss, **kw)


if __name__ == '__main__':
    for name, fn in [('1-2-3', msfanet_123), ('2-2-2', msfanet_222)]:
        m = fn(751).eval()
        p = sum(t.numel() for t in m.parameters()) / 1e6
        print(name, f'params (incl. 751-ID classifier): {p:.3f}M')
        print(m(torch.randn(2, 3, 256, 128)).shape)