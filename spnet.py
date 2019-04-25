import torch
import torch.nn as nn
import torchvision.models as models

class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, stride, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super(InvertedResidual, self).__init__()
        self.benchmodel = 1 if inp == oup else 2
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x = self._concat(x[:, :(x.shape[1] // 2), :, :], self.banch2(x[:, (x.shape[1] // 2):, :, :]))
        elif 2 == self.benchmodel:
            x = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(x, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, stage_out_channels, stage_repeats=(4, 8, 4)):
        super(ShuffleNetV2, self).__init__()
        self.stage_repeats = stage_repeats

        # building first layer
        input_channel = stage_out_channels[0]
        self.conv1 = conv_bn(3, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # building stages
        self.stage2 = self._make_stage(self.stage_repeats[0], stage_out_channels[0], stage_out_channels[1])
        self.stage3 = self._make_stage(self.stage_repeats[1], stage_out_channels[1], stage_out_channels[2])
        self.stage4 = self._make_stage(self.stage_repeats[2], stage_out_channels[2], stage_out_channels[3])

        # building last several layers
        self.conv_last = conv_1x1_bn(stage_out_channels[-2], stage_out_channels[-1], 2)

    def _make_stage(self, numrepeat, input_channel, output_channel):
        stage = []
        for i in range(numrepeat):
            if i == 0:
                # inp, oup, stride):
                stage.append(InvertedResidual(input_channel, output_channel, 2))
            else:
                stage.append(InvertedResidual(input_channel, output_channel, 1))
            input_channel = output_channel
        return nn.Sequential(*stage)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x2 = self.stage2(x)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x = self.conv_last(x4)
        return [x2, x3, x4, x]


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, upscale_factor=2):
        super(UpSampleBlock, self).__init__()

        self.conv_1 = conv2DBatchNormRelu(in_channels=in_channels,
                                          n_filters=in_channels/2,
                                          k_size=1,
                                          stride=1,
                                          padding=0)

        self.conv_2 = conv2DBatchNormRelu(in_channels=in_channels,
                                          n_filters=in_channels/2,
                                          k_size=3,
                                          stride=1,
                                          dilation=2,
                                          padding=2)

        self.conv_3 = conv2DBatchNormRelu(in_channels=in_channels,
                                          n_filters=in_channels/2,
                                          k_size=5,
                                          stride=1,
                                          dilation=1,
                                          padding=2)

        self.conv_4 = conv2DBatchNormRelu(in_channels=in_channels,
                                          n_filters=in_channels/2,
                                          k_size=7,
                                          stride=1,
                                          dilation=1,
                                          padding=3)

        self.sub_pixel = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = torch.cat([self.conv_1(x), self.conv_2(x), self.conv_3(x), self.conv_4(x)], 1)
        x = self.sub_pixel(x)
        return x


class TAM(nn.Module):
    def __init__(self, in_channels):
        super(TAM, self).__init__()
        self.BU_conv1 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0)
        self.BU_conv2 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0)
        self.BU_sigmoid = nn.Sigmoid()

        self.downsample1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.rb1 = InvertedResidual(in_channels, in_channels)
        self.rb2 = InvertedResidual(in_channels, in_channels)
        self.rb3 = InvertedResidual(in_channels, in_channels)
        self.rb4 = InvertedResidual(in_channels, in_channels)

        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.M_sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, fs, fd):
        # temp->B
        temp = self.BU_sigmoid(self.BU_conv1(torch.cat([fs, fd], 1)))
        temp = self.BU_conv2(torch.cat([temp * fd, temp*(1-temp)], 1))
        temp = self.downsample1(temp)
        temp = self.rb1(temp)
        temp = self.downsample2(temp)
        temp = self.rb2(temp)
        temp = self.upsample1(temp)
        temp = self.rb3(temp)
        temp = self.upsample2(temp)
        temp = self.rb4(temp)

        # temp -> M
        temp = self.M_sigmoid(temp)

        temp = self.conv(torch.cat([(1+temp)*fd, (1+temp)*fs], 1))

        return temp


class ResidualBlockDecode(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockDecode, self).__init__()
        self.bottleneck1 = InvertedResidual(in_channels, out_channels)
        self.bottleneck2 = InvertedResidual(out_channels, out_channels)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        return x


class SPNet(nn.Module):
    def __init__(self, n_classes, stage_out_channels=(24, 128, 256, 512, 1024)):
        super(SPNet, self).__init__()
        self.stage_outps = stage_out_channels
        self.ShuffleNet = ShuffleNetV2(stage_out_channels=stage_out_channels)

        self.upsample_res_5 = UpSampleBlock(self.stage_outps[4])
        self.upsample_res_d1 = UpSampleBlock(self.stage_outps[3])
        self.upsample_res_d2 = UpSampleBlock(self.stage_outps[3])
        self.upsample_res_d3 = UpSampleBlock(self.stage_outps[2])
        self.upsample_res_d4 = UpSampleBlock(self.stage_outps[2])
        self.upsample_res_d5 = UpSampleBlock(self.stage_outps[1])
        self.upsample_res_d6 = UpSampleBlock(self.stage_outps[1])

        self.TAM_res_d3 = TAM(self.stage_outps[2])
        self.TAM_res_d4 = TAM(self.stage_outps[2])
        self.TAM_res_d5 = TAM(self.stage_outps[1])
        self.TAM_res_d6 = TAM(self.stage_outps[1])
        self.TAM_res_d7 = TAM(self.stage_outps[1]//2)
        self.TAM_res_d8 = TAM(self.stage_outps[1]//2)

        self.res_d1 = ResidualBlockDecode(self.stage_outps[4], self.stage_outps[3])
        self.res_d2 = ResidualBlockDecode(self.stage_outps[3] * 3, self.stage_outps[3])
        self.res_d3 = ResidualBlockDecode(self.stage_outps[3], self.stage_outps[2])
        self.res_d4 = ResidualBlockDecode(self.stage_outps[3], self.stage_outps[2])
        self.res_d5 = ResidualBlockDecode(self.stage_outps[2], self.stage_outps[1])
        self.res_d6 = ResidualBlockDecode(self.stage_outps[2], self.stage_outps[1])
        self.res_d7 = ResidualBlockDecode(self.stage_outps[1]//2, self.stage_outps[1]//2)
        self.res_d8 = ResidualBlockDecode(self.stage_outps[1]//2, self.stage_outps[1]//2)

        self.conv_d1 = nn.Conv2d(in_channels=self.stage_outps[3], out_channels=1, kernel_size=1)
        self.conv_d2 = nn.Conv2d(in_channels=self.stage_outps[3], out_channels=n_classes, kernel_size=1)
        self.conv_d3 = nn.Conv2d(in_channels=self.stage_outps[2], out_channels=1, kernel_size=1)
        self.conv_d4 = nn.Conv2d(in_channels=self.stage_outps[2], out_channels=n_classes, kernel_size=1)
        self.conv_d5 = nn.Conv2d(in_channels=self.stage_outps[1], out_channels=1, kernel_size=1)
        self.conv_d6 = nn.Conv2d(in_channels=self.stage_outps[1], out_channels=n_classes, kernel_size=1)
        self.conv_d7 = nn.Conv2d(in_channels=self.stage_outps[1]//2, out_channels=1, kernel_size=1)
        self.conv_d8 = nn.Conv2d(in_channels=self.stage_outps[1]//2, out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        x2, x3, x4, x = self.ShuffleNet(x)

        x = self.upsample_res_5(x)
        r1 = self.res_d1(torch.cat((x, x4), 1))
        depth_1 = self.conv_d1(r1)
        r2 = self.res_d2(torch.cat((x, r1, x4), 1))
        segmentation_1 = self.conv_d2(r2)

        x = self.upsample_res_d2(r2)
        r1 = self.res_d3(torch.cat((self.TAM_res_d3(x, self.upsample_res_d1(r1)), x3), 1))
        depth_2 = self.conv_d3(r1)
        r2 = self.res_d4(torch.cat((self.TAM_res_d4(x, r1), x3), 1))
        segmentation_2 = self.conv_d4(r2)

        x = self.upsample_res_d4(r2)
        r1 = self.res_d5(torch.cat((self.TAM_res_d5(x, self.upsample_res_d3(r1)), x2), 1))
        depth_3 = self.conv_d5(r1)
        r2 = self.res_d6(torch.cat((self.TAM_res_d6(x, r1), x2), 1))
        segmentation_3 = self.conv_d6(r2)

        x = self.upsample_res_d6(r2)
        r1 = self.res_d7(self.TAM_res_d7(x, self.upsample_res_d5(r1)))
        depth_4 = self.conv_d7(r1)
        r2 = self.res_d8(self.TAM_res_d8(x, r1))
        segmentation_4 = self.conv_d8(r2)

        return [(depth_1, segmentation_1),
                (depth_2, segmentation_2),
                (depth_3, segmentation_3),
                (depth_4, segmentation_4)]


if __name__ == '__main__':
    import time
    from utils import params_size
    model = SPNet(19)
    params_size(model)
    t1 = time.time()
    x = torch.rand(1, 3, 512, 1024)
    x = model(x)
    print(time.time()-t1)







