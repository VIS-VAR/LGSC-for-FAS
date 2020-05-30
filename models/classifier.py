import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn
import paddle.fluid.layers as L
from utils import model_size, build_norm_layer, build_conv_layer
from models.resnet import ResNet
from models.triple_loss import TripletLoss


class Classifier(fluid.dygraph.Layer):
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Classifier, self).__init__()
        self.dropout = backbone.pop('dropout')
        self.backbone = ResNet(**backbone)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.triple_loss = TripletLoss()
        self.avgpool = nn.Pool2D(pool_type='avg', global_pooling=True)
        self.fc = nn.Linear(512, 2, act='softmax')
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        model_size(self)

    def get_losses(self, out, cls_out, mask, gt_labels):
        loss_cls = L.mean(L.cross_entropy(cls_out, gt_labels)) * self.train_cfg['w_cls']
        loss_tir = 0
        for feat in out[:-1]:
            feat = L.squeeze(self.avgpool(feat), axes=[2, 3])
            loss_tir += self.triple_loss(feat, gt_labels) * self.train_cfg['w_tri']
        loss = loss_cls + loss_tir
        return dict(loss_cls=loss_cls, loss_tir=loss_tir, loss=loss)

    def forward(self, img, label, mask=None, return_loss=True):
        outs = self.backbone(img)
        cls_out = self.avgpool(outs[-1])
        if return_loss:
            cls_out = L.dropout(cls_out, dropout_prob=self.dropout, is_test=False)
            cls_out = self.fc(L.squeeze(cls_out, axes=[2, 3]))
            losses = self.get_losses(outs, cls_out, mask, label)
            return losses
        else:
            cls_out = self.fc(L.squeeze(cls_out, axes=[2, 3]))
            cls_out = L.softmax(cls_out).numpy()[:, 0]
            return cls_out


class ClsLite(fluid.dygraph.Layer):
    def __init__(self):
        super(ClsLite, self).__init__()
        self.conv_cfg = dict(type='Conv')
        self.norm_cfg = dict(type='BN')
        self._make_stem_layer()
        self.avgpool = nn.Pool2D(pool_type='avg', global_pooling=True)
        self.fc = nn.Linear(128, 2, act='softmax')

    def _make_stem_layer(self):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            3,
            64,
            7,
            stride=2,
            padding=3,
            bias_attr=False)
        _, self.norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.relu = fluid.layers.relu
        self.maxpool = nn.Pool2D(pool_size=3, pool_stride=2, pool_padding=1)
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            64,
            128,
            3,
            stride=2,
            padding=1,
            bias_attr=False)
        _, self.norm2 = build_norm_layer(self.norm_cfg, 128, postfix=1)

    def forward(self, cue, label, return_loss=True):
        out = self.conv1(cue)
        out = self.norm1(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.avgpool(out)
        if return_loss:
            cls_out = L.dropout(out, dropout_prob=0.5, is_test=False)
            cls_out = self.fc(L.squeeze(cls_out, axes=[2, 3]))
            loss_cls = L.mean(L.cross_entropy(cls_out, label))
            losses = dict(loss_cls=loss_cls, loss=loss_cls)
            return losses
        else:
            cls_out = self.fc(L.squeeze(out, axes=[2, 3]))
            cls_out = L.softmax(cls_out).numpy()[:, 0]
            return cls_out
