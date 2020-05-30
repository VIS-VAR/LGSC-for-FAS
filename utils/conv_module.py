import warnings
import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn
from .norm_module import build_norm_layer

conv_cfg = {
    'Conv': nn.Conv2D,
    'Conv3D': nn.Conv3D,
}


def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer

    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.

    Returns:
        layer (fluid.dygrah.Layer): created conv layer
    """
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


class ConvModule(fluid.dygraph.Layer):
    """Conv-Norm-Activation block.

    Args:

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=None,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 activation='relu',
                 activate_last=True):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.activate_last = activate_last

        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        # build convolution layer
        self._conv = build_conv_layer(
            conv_cfg,
            num_channels=in_channels,
            num_filters=out_channels,
            filter_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias,
            param_attr=fluid.initializer.MSRA(uniform=False))

        # build normalization layers
        if self.with_norm:
            self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
            self.add_sublayer(self.norm_name, norm)

        # build activation layer
        if self.with_activatation:
            if self.activation not in ['relu']:
                raise ValueError('{} is currently not supported.'.format(
                    self.activation))
            if activation == 'relu':
                self.relu = fluid.layers.relu

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        pass

    def forward(self, x, activate=True, norm=True):
        x = self._conv(x)
        if norm and self.with_norm:
            x = self.norm(x)
        if activate and self.with_activatation:
            x = self.relu(x)
        return x


class Sequential(fluid.dygraph.Layer):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        for idx, layer in enumerate(args):
            self.add_sublayer(str(idx), layer)

    def forward(self, input):
        for _, layer in self._sub_layers.items():
            input = layer(input)
        return input