import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from torch import Tensor
from networks.layers import *
import itertools
from torch.distributions.uniform import Uniform
import numpy as np

def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)  # 直接对某层的data进行复制
        m.bias.data.fill_(0)

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ShareEncoder(nn.Module):
    def __init__(self, nin, pretrained=True,has_dropout=False):
        # 把unet11的拿过来了
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.has_dropout = has_dropout
        # self.encoder = models.vgg11(pretrained=pretrained).features

        self.encoder = models.vgg11(pretrained=pretrained).features
        # self.encoder = self.add_dropout(pretrained)

        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.dropout = nn.Dropout2d(p=0.3, inplace=False)

    def forward(self, x):
        x = x.float()
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))
        if self.has_dropout:
            conv5 = self.dropout(conv5)
        return conv5, conv1, conv2, conv3, conv4

class Decoder1(nn.Module):
    def __init__(self,num_filters=32,nout=2):
        super().__init__()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.center1 = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec51 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec41 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec31 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec21 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec11 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final1 = nn.Conv2d(num_filters, nout, kernel_size=1) # 2通道出去

    def forward(self, conv5, conv1, conv2, conv3, conv4):
        center1 = self.center1(self.pool1(conv5))
        dec51 = self.dec51(torch.cat([center1, conv5], 1))
        dec41 = self.dec41(torch.cat([dec51, conv4], 1))
        dec31 = self.dec31(torch.cat([dec41, conv3], 1))
        dec21 = self.dec21(torch.cat([dec31, conv2], 1))
        dec11 = self.dec11(torch.cat([dec21, conv1], 1))
        final1 = self.final1(dec11)

        return final1, dec11

class Decoder2(nn.Module):
    def __init__(self,num_filters=32,nout=2):
        super().__init__()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.center2 = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec52 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec42 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec32 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec22 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec12 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final2 = nn.Conv2d(num_filters, nout, kernel_size=1) # 2通道出去

    def forward(self, conv5, conv1, conv2, conv3, conv4):
        center2 = self.center2(self.pool2(conv5))
        dec52 = self.dec52(torch.cat([center2, conv5], 1))
        dec42 = self.dec42(torch.cat([dec52, conv4], 1))
        dec32 = self.dec32(torch.cat([dec42, conv3], 1))
        dec22 = self.dec22(torch.cat([dec32, conv2], 1))
        dec12 = self.dec12(torch.cat([dec22, conv1], 1))
        final2 = self.final2(dec12)

        return final2, dec12

class UnFNet(nn.Module):
    def __init__(self, nin, nout, device, l_rate, pretrained=True, has_dropout=True, num_filters=32):
        super().__init__()
        self.encoder = ShareEncoder(nin, pretrained=pretrained, has_dropout=has_dropout).to(device)
        self.decoder1 = Decoder1(num_filters=num_filters, nout=nout).to(device)
        self.decoder2 = Decoder2(num_filters=num_filters, nout=nout).to(device)

        # self.encoder.apply(weights_init)
        self.decoder1.apply(weights_init)
        self.decoder2.apply(weights_init)

        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=l_rate, betas=(0.9, 0.99))
        optimizer1 = torch.optim.Adam(self.decoder1.parameters(), lr=l_rate, betas=(0.9, 0.99))
        optimizer2 = torch.optim.Adam(self.decoder2.parameters(), lr=l_rate, betas=(0.9, 0.99))
        self.lr = l_rate
        self.optimizers = [optimizer, optimizer1, optimizer2]
    def forward(self, input):
        conv5, conv1, conv2, conv3, conv4 = self.encoder(input)
        # print(conv1.size(), conv2.size(), conv3.size(), conv4.size(), conv5.size())
        res1, fea1 = self.decoder1(conv5, conv1, conv2, conv3, conv4)
        res2, fea2 = self.decoder2(conv5, conv1, conv2, conv3, conv4)
        return [res1, res2]

    def optimize(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
    # def update_lr(self):
    def update_lr(self):
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr / 10
        self.lr = self.lr / 10

class UnFNet_singal(nn.Module):
    def __init__(self, nin, nout, device, l_rate, pretrained=True, has_dropout=True, num_filters=32):
        super().__init__()
        self.encoder = ShareEncoder(nin, pretrained=pretrained, has_dropout=has_dropout).to(device)
        self.decoder1 = Decoder1(num_filters=num_filters, nout=nout).to(device)

        # self.encoder.apply(weights_init)
        self.decoder1.apply(weights_init)

        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=l_rate, betas=(0.9, 0.99))
        optimizer1 = torch.optim.Adam(self.decoder1.parameters(), lr=l_rate, betas=(0.9, 0.99))

        self.optimizers = [optimizer, optimizer1]
        self.lr = l_rate
    def forward(self, input):
        conv5, conv1, conv2, conv3, conv4 = self.encoder(input)
        # print(conv1.size(), conv2.size(), conv3.size(), conv4.size(), conv5.size())
        res1, fea1 = self.decoder1(conv5, conv1, conv2, conv3, conv4)
        return res1

    def optimize(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
    def update_lr(self):
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr / 10
        self.lr = self.lr / 10

#####################################################################################

class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)



class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x

class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x

def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class Decoder_URPC(nn.Module):
    def __init__(self, params):
        super(Decoder_URPC, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        # print(x0.size(), x1.size(), x2.size(), x3.size(), x4.size())
        x = self.up1(x4, x3)
        if self.training:
            dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5))
        else:
            dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        if self.training:
            dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg



class UNet_URPC(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_URPC, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_URPC(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg = self.decoder(
            feature, shape)
        return dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg
    

class UnFNet_6(nn.Module):
    def __init__(self, nin, nout, device, l_rate, pretrained=True, has_dropout=True, num_filters=32):
        super().__init__()
        self.encoder = ShareEncoder(nin, pretrained=pretrained, has_dropout=has_dropout).to(device)
        self.decoder1 = Decoder1(num_filters=num_filters, nout=nout).to(device)
        self.decoder2 = Decoder2(num_filters=num_filters, nout=nout).to(device)
        self.decoder3 = Decoder2(num_filters=num_filters, nout=nout).to(device)
        self.decoder4 = Decoder2(num_filters=num_filters, nout=nout).to(device)
        self.decoder5 = Decoder2(num_filters=num_filters, nout=nout).to(device)
        self.decoder6 = Decoder2(num_filters=num_filters, nout=nout).to(device)


        # self.encoder.apply(weights_init)
        self.decoder1.apply(weights_init)
        self.decoder2.apply(weights_init)
        self.decoder3.apply(weights_init)
        self.decoder4.apply(weights_init)
        self.decoder5.apply(weights_init)
        self.decoder6.apply(weights_init)


        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=l_rate, betas=(0.9, 0.99))
        optimizer1 = torch.optim.Adam(self.decoder1.parameters(), lr=l_rate, betas=(0.9, 0.99))
        optimizer2 = torch.optim.Adam(self.decoder2.parameters(), lr=l_rate, betas=(0.9, 0.99))
        optimizer3 = torch.optim.Adam(self.decoder2.parameters(), lr=l_rate, betas=(0.9, 0.99))
        optimizer4 = torch.optim.Adam(self.decoder2.parameters(), lr=l_rate, betas=(0.9, 0.99))
        optimizer5 = torch.optim.Adam(self.decoder2.parameters(), lr=l_rate, betas=(0.9, 0.99))
        optimizer6 = torch.optim.Adam(self.decoder2.parameters(), lr=l_rate, betas=(0.9, 0.99))

        self.lr = l_rate
        self.optimizers = [optimizer, optimizer1,optimizer2, optimizer3, optimizer4, optimizer5, optimizer6]
    def forward(self, input):
        conv5, conv1, conv2, conv3, conv4 = self.encoder(input)
        # print(conv1.size(), conv2.size(), conv3.size(), conv4.size(), conv5.size())
        res1, fea1 = self.decoder1(conv5, conv1, conv2, conv3, conv4)
        res2, fea2 = self.decoder2(conv5, conv1, conv2, conv3, conv4)
        res3, fea3 = self.decoder3(conv5, conv1, conv2, conv3, conv4)
        res4, fea4 = self.decoder4(conv5, conv1, conv2, conv3, conv4)
        res5, fea5 = self.decoder5(conv5, conv1, conv2, conv3, conv4)
        res6, fea6 = self.decoder6(conv5, conv1, conv2, conv3, conv4)

        return [res1, res2, res3, res4, res5, res6]

    def optimize(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
    # def update_lr(self):
    def update_lr(self):
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr / 10
        self.lr = self.lr / 10


if __name__ == '__main__':
    device = torch.device("cuda")
    lr = 0.01
    input = torch.randn((4,3,256,256)).to(device)
    # model = UnFNet(3, 2, device, lr, pretrained=True, has_dropout=True)
    # # print(model)
    # res = model(input)
    # for i in range(5):
    #     out = model(input)
    # print(out[0][0].size())

    net = UNet_URPC(in_chns=3, class_num=1).cuda()
    res = net(input)
    # print(res[2].size())



