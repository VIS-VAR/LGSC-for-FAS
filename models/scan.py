import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn
import paddle.fluid.layers as L
from utils import ConvModule, model_size
from models.resnet import ResNet, BasicBlock, make_res_layer
from models.triple_loss import TripletLoss


class DeCoder(fluid.dygraph.Layer):

    def __init__(self,
                 in_channels=(64, 64, 128, 256, 512),
                 out_channels=(512, 256, 128, 64, 64, 3),
                 num_outs=6,
                 conv_cfg=None,
                 norm_cfg=dict(type='IN'),
                 activation='relu'):
        super(DeCoder, self).__init__()
        assert isinstance(in_channels, tuple)
        self.in_channels = in_channels  # [64, 64, 128, 256, 512]
        self.out_channels = out_channels  # [512, 256, 128, 64, 64, 3]
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation

        self.deres_layers = []
        self.conv2x2 = []
        self.conv1x1 = []

        for i in range(self.num_ins-1, -1, -1):  # 43210
            deres_layer = make_res_layer(
                BasicBlock,
                inplanes=128 if i == 1 else in_channels[i],
                planes=out_channels[-i-1],
                blocks=2,
                stride=1,
                dilation=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            conv2x2 = ConvModule(
                in_channels[i],
                out_channels=in_channels[i] if i < 2 else int(in_channels[i] / 2),
                kernel_size=2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation)
            conv1x1 = ConvModule(
                in_channels=128 if i == 1 else in_channels[i],
                out_channels=out_channels[-i-1],
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=None)
            self.add_sublayer('deres{}'.format(5 - i), deres_layer)
            self.add_sublayer('conv_module2x2.{}'.format(5 - i), conv2x2)
            self.add_sublayer('conv_module1x1.{}'.format(5 - i), conv1x1)
            self.deres_layers.append(deres_layer)
            self.conv2x2.append(conv2x2)
            self.conv1x1.append(conv1x1)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        pass

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        outs = []
        out = inputs[-1]
        outs.append(out)

        for i in range(self.num_ins):
            out = L.resize_nearest(out, scale=2, align_corners=False)
            out = L.pad2d(out, [0, 1, 0, 1])
            out = self.conv2x2[i](out)
            if i < 4:
                out = L.concat([out, inputs[-i-2]], axis=1)
            identity = self.conv1x1[i](out)
            out = self.deres_layers[i](out) + identity
            outs.append(out)
        outs[-1] = L.tanh(outs[-1])

        return tuple(outs)


class SCAN(fluid.dygraph.Layer):
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SCAN, self).__init__()
        self.dropout = head.pop('dropout')
        self.backbone = ResNet(**backbone)
        self.neck = DeCoder(**neck)
        self.head = ResNet(**head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.triple_loss = TripletLoss()
        self.avgpool = nn.Pool2D(pool_type='avg', global_pooling=True)
        self.fc = nn.Linear(512, 2, act='softmax')
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights()
        self.head.init_weights()
        model_size(self)

    def get_losses(self, out, cls_out, mask, gt_labels):
        loss_cls = L.mean(L.cross_entropy(cls_out, gt_labels)) * self.train_cfg['w_cls']
        cue = out[-1] if self.train_cfg['with_mask'] else L.elementwise_mul(
            out[-1], L.cast(gt_labels, 'float32'), axis=0)
        num_reg = L.cast(L.reduce_sum(gt_labels) * cue.shape[1] * cue.shape[2] * cue.shape[3], 'float32')
        loss_reg = L.reduce_sum(L.abs(mask - cue)) / (num_reg + 1e-8) * self.train_cfg['w_reg']
        loss_tir = 0
        for feat in out[:-1]:
            feat = L.squeeze(self.avgpool(feat), axes=[2, 3])
            loss_tir += self.triple_loss(feat, gt_labels) * self.train_cfg['w_tri']
        loss = loss_cls + loss_reg + loss_tir
        return dict(loss_cls=loss_cls, loss_reg=loss_reg, loss_tir=loss_tir, loss=loss)

    def forward(self, img, label, mask=None, return_loss=True):
        outs = self.backbone(img)
        outs = self.neck(outs)
        if return_loss:
            s = img + outs[-1]
            cls_out = self.avgpool(self.head(s)[-1])
            cls_out = L.squeeze(cls_out, axes=[2, 3])
            if self.dropout:
                cls_out = L.dropout(cls_out, dropout_prob=self.dropout)
            cls_out = self.fc(cls_out)
            losses = self.get_losses(outs, cls_out, mask, label)
            return losses
        else:
            cue = L.abs(outs[-1]).numpy()
            return cue

