import torch
import torch.nn as nn

""" segmentation model example
"""

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 32)
        self.e2 = encoder_block(32, 64)
        self.e3 = encoder_block(64, 128)
        self.e4 = encoder_block(128, 256)

        """ Bottleneck """
        self.b = conv_block(256, 512)

        """ Decoder """
        self.d3_0 = decoder_block(512, 512, 256)
        self.d2_0 = decoder_block(256, 256, 128)
        self.d1_0 = decoder_block(128, 128, 64)
        self.d0_0 = decoder_block(64, 64, 32)
        
        self.d2_1 = decoder_block(256, 3*128, 128)
        self.d1_1 = decoder_block(128, 3*64, 64)
        self.d0_1 = decoder_block(64, 3*32, 32)

        self.d1_2 = decoder_block(128, 4*64, 64)
        self.d0_2 = decoder_block(64, 4*32, 32)

        self.d0_3 = decoder_block(64, 5*32, 32)

        """ Classifier """
        self.outputs = nn.Conv2d(32, 19, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d0_1 = self.d0_0(s2, s1)
        d1_1 = self.d1_0(s3, s2)
        d2_1 = self.d2_0(s4, s3)
        d3_1 = self.d3_0(b, s4)

        d0_2 = self.d0_1(d1_1, torch.cat((d0_1, s1),1))
        d1_2 = self.d1_1(d2_1, torch.cat((d1_1, s2),1))
        d2_2 = self.d2_1(d3_1, torch.cat((d2_1, s3),1))

        d0_3 = self.d0_2(d1_2, torch.cat((d0_2, d0_1, s1),1))
        d1_3 = self.d1_2(d2_2, torch.cat((d1_2, d1_1, s2),1))

        d0_4 = self.d0_3(d1_3, torch.cat((d0_3, d0_2, d0_1, s1),1))

        """ Segmentation output """
        outputs = self.outputs(d0_4)

        return outputs


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.dropout = nn.Dropout(p=0.25) #Add a dropout layer
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, upsample_inchannel, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(upsample_inchannel, upsample_inchannel // 2, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(in_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x
