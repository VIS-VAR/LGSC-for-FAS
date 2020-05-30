import logging
import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn
from utils import build_norm_layer, build_conv_layer, Sequential


class BasicBlock(fluid.dygraph.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(BasicBlock, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias_attr=False)
        self.add_sublayer(self.norm1_name, norm1)

        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias_attr=False)
        self.add_sublayer(self.norm2_name, norm2)

        self.relu = fluid.layers.relu
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = fluid.layers.elementwise_add(out, identity)
        out = self.relu(out)

        return out


class Bottleneck(fluid.dygraph.Layer):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        """Bottleneck block for ResNet.
        the stride-two layer is the 3x3 conv layer,.
        """
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.conv1_stride = 1
        self.conv2_stride = stride

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            1,
            stride=1,
            bias_attr=False)
        self.add_sublayer(self.norm1_name, norm1)

        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias_attr=False)
        self.add_sublayer(self.norm2_name, norm2)

        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            1,
            bias_attr=False)
        self.add_sublayer(self.norm3_name, norm3)

        self.relu = fluid.layers.relu
        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = fluid.layers.elementwise_add(out, identity)

        out = self.relu(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN')):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = Sequential(
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * block.expansion,
                1,
                stride=stride,
                bias_attr=False),
            build_norm_layer(norm_cfg, planes * block.expansion)[1]
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                1,
                dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg))

    return Sequential(*layers)


class ResNet(fluid.dygraph.Layer):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=True,
                 zero_init_residual=True):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_sublayer(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            3,
            64,
            7,
            stride=2,
            padding=3,
            bias_attr=False)
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.add_sublayer(self.norm1_name, norm1)
        self.relu = fluid.layers.relu
        self.maxpool = nn.Pool2D(pool_size=3, pool_stride=2, pool_padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for layer in [self.conv1, self.norm1]:
                layer.eval()
                for param in layer.parameters():
                    param.stop_gradient = True

        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, 'layer{}'.format(i))
            layer.eval()
            for param in layer.parameters():
                param.stop_gradient = True

    def init_weights(self, pretrained=None):
        logger = logging.getLogger()
        if isinstance(pretrained, str):
            logger.info('Loading pretrained model from {}'.format(pretrained))
            self.set_dict(fluid.dygraph.load_dygraph(pretrained)[0])
        elif pretrained is None:
            logger.warning('No pretrained model for Resnet')
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        outs.append(x)   # add for encoder
        x = self.maxpool(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self):
        super(ResNet, self).train()
        self._freeze_stages()
        if self.norm_eval:
            for layer in self.sublayers():
                # trick: eval have effect on BatchNorm only
                if isinstance(layer, nn.BatchNorm):
                    layer.eval()
