import torch.nn as nn

# initialization
def weight_initialization(m):
    '''
    initialize Conv, BatchNorm layer
    :param m: module classes(nn.~)
    '''
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.InstanceNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, instance_norm_init=False):
        '''
        implementation of a Residual Block
        :param in_channels: C of input from (N, C, H, W)
        :param instance_norm_init: if it is True, affine parameter of InstanceNorm2d is True.
        '''
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_channels, affine=instance_norm_init),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_channels, affine=instance_norm_init)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, in_channels, num_residual_blocks=9, instance_norm_init=False):
        '''
        implementation of Generator
        :param in_channels: the number of channels of input shape(C,H,W)
        :param num_residual_blocks: the number of residual blocks
        :param instance_norm_init : InstanceNorm2d layer's weights and bias are learnable if it is True.
        '''
        super(Generator, self).__init__()
        # resnet-34
        # (7,64,s=2), (3,64)x6, (3,128,s=2), (3,128)x7, (3,256,s=2), (3,256)x11, (3,512,s=2),(3,512)x5

        origin_in_channels = in_channels
        # initial conv block
        out_channels = 64
        model = [nn.Conv2d(in_channels, out_channels, 7, padding=3, padding_mode='reflect'),
                 nn.InstanceNorm2d(out_channels, affine=instance_norm_init),
                 nn.ReLU(inplace=True)
                 ]

        # downsampling
        in_channels = out_channels
        for _ in range(2):
            out_channels *= 2
            model += [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_channels, affine=instance_norm_init),
                      nn.ReLU(inplace=True)
                      ]
            in_channels = out_channels

        # add residual blocks
        model += [ResidualBlock(out_channels)] * num_residual_blocks

        # Upsampling
        # nn.Upsampling = simple interpolation
        # nn.ConvTranspose2d = Deconvolution
        # so I used nn.ConvTranspose2d layer instead of nn.Upsample layer
        for _ in range(2):
            out_channels //= 2
            model += [
                nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels, affine=instance_norm_init),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels

        # Output layer
        model += [nn.Conv2d(out_channels, origin_in_channels, 7, padding=3, padding_mode='reflect'),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels, instance_norm_init=False):
        '''
        PatchGAN Discriminator (patch size = 70)
        input size = (3, 256, 256)
        (#F=64,k=4,s=2,p=1)->((#F=128,k=4,s=2,p=1)->(#F=256,k=4,s=2,p=1)->(#F=512,k=4,s=2,p=1)->(#F=1,k=4,s=2,p=1)
        output size = (3, 30, 30)
        :param in_channels: the number of channels of input shape(C,H,W)
        :param instance_norm_init: InstanceNorm2d layer's weights and bias are learnable if it is True.
        '''

        super(Discriminator, self).__init__()

        model = [nn.Conv2d(in_channels, 64, 4, stride=2, padding=1), # (3,128,128)
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(64, 128, 4, stride=2, padding=1), # (3,64,64)
                 nn.InstanceNorm2d(128, affine=instance_norm_init),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(128, 256, 4, stride=2, padding=1), # (3,32,32)
                 nn.InstanceNorm2d(256, affine=instance_norm_init),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(256, 512, 4, padding=1), # (3,31,31)
                 nn.InstanceNorm2d(512, affine=instance_norm_init),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(512, 1, 4, padding=1) # (3,30,30)
                 ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

