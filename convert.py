from models.resnet import ResNet
from utils.util import torch_weight_to_paddle_model

resnet18 = ResNet(depth=18)
torch_weight_to_paddle_model('./pretrained/resnet18-5c106cde.pth', resnet18)
