import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn

from paddle.fluid import core
from paddle.fluid.initializer import Constant
from paddle.fluid.framework import in_dygraph_mode, _dygraph_tracer, _varbase_creator
from paddle.fluid.dygraph.nn import dygraph_utils


class InstanceNorm(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 act=None,
                 is_test=False,
                 momentum=0.9,
                 epsilon=1e-05,
                 param_attr=None,
                 bias_attr=None,
                 dtype='float32',
                 data_layout='NCHW',
                 in_place=False,
                 use_global_stats=False,
                 trainable_statistics=False):
        super(InstanceNorm, self).__init__()
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._act = act

        assert bias_attr is not False, "bias_attr should not be False in batch_norm."

        if dtype == "float16":
            self._dtype = "float32"
        else:
            self._dtype = dtype

        param_shape = [num_channels]

        # create parameter
        self.weight = self.create_parameter(
            attr=self._param_attr,
            shape=param_shape,
            dtype=self._dtype,
            default_initializer=Constant(1.0))
        self.weight.stop_gradient = use_global_stats and self._param_attr.learning_rate == 0.

        self.bias = self.create_parameter(
            attr=self._bias_attr,
            shape=param_shape,
            dtype=self._dtype,
            is_bias=True)
        self.bias.stop_gradient = use_global_stats and self._param_attr.learning_rate == 0.

        self._in_place = in_place
        self._data_layout = data_layout
        self._momentum = momentum
        self._epsilon = epsilon
        self._is_test = is_test
        self._fuse_with_relu = False
        self._use_global_stats = use_global_stats
        self._trainable_statistics = trainable_statistics

    def forward(self, input):
        # create output
        # mean and mean_out share the same memory
        # variance and variance out share the same memory

        attrs = {
            "momentum": self._momentum,
            "epsilon": self._epsilon,
            "is_test": self._is_test,
            "data_layout": self._data_layout,
            "use_mkldnn": False,
            "fuse_with_relu": self._fuse_with_relu,
            "use_global_stats": self._use_global_stats,
            "trainable_statistics": self._trainable_statistics
        }

        inputs = {
            "X": [input],
            "Scale": [self.weight],
            "Bias": [self.bias]
        }

        if in_dygraph_mode():
            attrs['is_test'] = not _dygraph_tracer()._train_mode
            saved_mean = _varbase_creator(dtype=self._dtype)
            saved_variance = _varbase_creator(dtype=self._dtype)
            instance_norm_out = _varbase_creator(dtype=self._dtype)
            instance_norm_out.stop_gradient = False
            # inplace is not supported currently
        else:
            saved_mean = self._helper.create_variable_for_type_inference(
                dtype=self._dtype, stop_gradient=True)
            saved_variance = self._helper.create_variable_for_type_inference(
                dtype=self._dtype, stop_gradient=True)
            instance_norm_out = input if self._in_place else self._helper.create_variable_for_type_inference(
                self._dtype)

        outputs = {
            "Y": [instance_norm_out],
            "SavedMean": [saved_mean],
            "SavedVariance": [saved_variance]
        }

        if in_dygraph_mode():
            outs = core.ops.instance_norm(inputs, attrs, outputs)
            return dygraph_utils._append_activation_in_dygraph(
                instance_norm_out, act=self._act)

        self._helper.append_op(
            type="instance_norm", inputs=inputs, outputs=outputs, attrs=attrs)

        # Currently, we don't support inplace in dygraph mode
        return self._helper.append_activation(instance_norm_out, self._act)


norm_cfg = {
    # format: layer_type: (abbreviation, module)
    'BN': ('bn', nn.BatchNorm),
    'IN': ('in', InstanceNorm)
}


def build_norm_layer(cfg, num_channels, postfix=''):
    """ Build normalization layer

    Args:

    Returns:
        layer (fluid.dygrah.Layer): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    stop_gradient = cfg_.pop('stop_gradient', False)
    cfg_.setdefault('epsilon', 1e-5)

    layer = norm_layer(num_channels=num_channels, **cfg_)

    # for param in layer.parameters():
    #     param.stop_gradient = stop_gradient

    return name, layer
