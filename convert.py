import paddle.fluid as fluid
from models.resnet import ResNet
from utils.util import torch_weight_to_paddle_model

with fluid.dygraph.guard():
    resnet18 = ResNet(depth=18, norm_cfg=dict(type='BN'))
    torch_weight_to_paddle_model('./pretrained/resnet18-5c106cde.pth', resnet18)
