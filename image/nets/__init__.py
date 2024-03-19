
from .wrn_32 import wrn16_32, wrn28_10
from .vgg_32 import vgg11_32, vgg16_32
from .resnet_32 import resnet20, resnet32, resnet44, resnet56, resnet110

from .mobilenetv2 import mobilenetv2
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .vgg import vgg11, vgg13, vgg16, vgg11_bn, vgg13_bn, vgg16_bn
from .vision_transformer import vit_b_16
from .swin_transformer import swin_transformer_base, swin_transformer_small, swin_transformer_tiny


get_model_from_name = {
    "resnet20"                  : resnet20,
    "resnet32"                  : resnet32,
    "resnet44"                  : resnet44,
    "resnet56"                  : resnet56,
    "resnet110"                 : resnet110,
    "wrn16_32"                  : wrn16_32,
    "wrn28_10"                  : wrn28_10,
    "vgg11_32"                  : vgg11_32,
    "vgg16_32"                  : vgg16_32,
    ##########################################
    "mobilenetv2"               : mobilenetv2,
    "resnet18"                  : resnet18,
    "resnet34"                  : resnet34,
    "resnet50"                  : resnet50,
    "resnet101"                 : resnet101,
    "resnet152"                 : resnet152,
    "vgg11"                     : vgg11,
    "vgg13"                     : vgg13,
    "vgg16"                     : vgg16,
    "vgg11_bn"                  : vgg11_bn,
    "vgg13_bn"                  : vgg13_bn,
    "vgg16_bn"                  : vgg16_bn,
    "vit_b_16"                  : vit_b_16,
    "swin_transformer_tiny"     : swin_transformer_tiny,
    "swin_transformer_small"    : swin_transformer_small,
    "swin_transformer_base"     : swin_transformer_base
}