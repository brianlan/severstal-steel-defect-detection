# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch._utils
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tqdm import tqdm

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


######################
#      Utilities
######################
def get_colors(n_desired_colors, cmap_name="rainbow"):
    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap(cmap_name, n_desired_colors)
    try:
        colors = cmap.colors
    except AttributeError:
        colors = cmap(np.linspace(0, 1, n_desired_colors))
    colors = (colors[:, :3] * 255).astype(np.int32)
    return colors


#########################################
#      Backbone Network Definition
#########################################
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(
        self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True
    ):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                            nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM),
                                    nn.ReLU(False),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class HighResolutionNet(nn.Module):
    def __init__(self, model_cfg_extra, **kwargs):
        super(HighResolutionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.stage1_cfg = model_cfg_extra["STAGE1"]
        num_channels = self.stage1_cfg["NUM_CHANNELS"][0]
        block = blocks_dict[self.stage1_cfg["BLOCK"]]
        num_blocks = self.stage1_cfg["NUM_BLOCKS"][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = model_cfg_extra["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = model_cfg_extra["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = model_cfg_extra["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        # # Classification Head
        # self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(pre_stage_channels)
        #
        # self.classifier = nn.Linear(2048, 1000)

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block, channels, head_channels[i], 1, stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        return y_list

    def init_weights(self, pretrained=""):
        logger.info("=> init weights from normal distribution")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info("=> loading pretrained model {}".format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info("=> loading {} pretrained model {}".format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


def get_cls_net(config, **kwargs):
    model = HighResolutionNet(config, **kwargs)
    model.init_weights()
    return model


#####################################
#   Segmentation Head Definition
#####################################
class SegmentationHead(nn.Module):
    def __init__(self, out_channels, model_cfg_extra=None, upsample_mode="nearest", final_upsample=False, **kwargs):
        super().__init__()
        channels = model_cfg_extra['STAGE4']['NUM_CHANNELS']
        in_channels = sum(channels)
        self.upsample_layers = [nn.Upsample(scale_factor=c / channels[0], mode=upsample_mode) for c in channels[1:]]
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        if final_upsample:
            self.final_upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, *inputs):
        """inputs are fmaps from branches with different spatial sizes. They should be ordered from largest to smallest.
        For example, r4 (56 x 56), r8 (28 x 28), r16 (14 x 14), r32 (7 x 7).
        """
        upsampled_fmaps = [up(i) for i, up in zip(inputs[1:], self.upsample_layers)]
        concat_features = torch.cat((inputs[0], *upsampled_fmaps), dim=1)
        out = self.conv2(self.relu(self.bn(self.conv1(concat_features))))
        if hasattr(self, "final_upsample"):
            out = getattr(self, "final_upsample")(out)
        return out


#################################
#      Transformations
#################################
class ZeroBlackMargin:
    def __init__(self, *, black_thresh=30, **kwargs):
        super().__init__(**kwargs)
        self.black_thresh = black_thresh

    def __call__(self, img):
        left, _, right, _ = self.find_tight_boundaries(img, self.black_thresh)
        img[:, :left, :] = 0
        img[:, right + 1 :, :] = 0
        return img

    @staticmethod
    def find_tight_boundaries(img, black_thresh):
        tmp_gray = img[..., 0].copy()
        tmp_gray[tmp_gray < black_thresh] = 0
        squashed_as_a_row = tmp_gray.sum(axis=0)
        row_nonzero = squashed_as_a_row.nonzero()[0]
        left, right = 0, tmp_gray.shape[1] - 1
        if len(row_nonzero) > 0:
            left, right = row_nonzero[0], row_nonzero[-1]
        return left, 0, right, tmp_gray.shape[0] - 1


class Div:
    def __init__(self, by):
        self.by = by

    def __call__(self, img):
        return img / self.by


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img -= self.mean
        img /= self.std
        return img


class ToTensor:
    def __call__(self, img):
        return torch.tensor(img.transpose(2, 0, 1)).float()


###################
#      Dataset
###################
class ImageReader:
    def __call__(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


class SeverstalSteelDefectDetection:
    def __init__(self, im_dir, transforms=None):
        self.im_dir = Path(im_dir)
        self.im_paths = [str(p) for p in self.im_dir.glob("*.jpg")]
        self.im_reader = ImageReader()
        self.transforms = transforms

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        img = self.im_reader(im_path)
        for t in self.transforms:
            img = t(img)
        return im_path, img


############################
#      IntegratedModel
############################
class HRSegNet(nn.Module):
    def __init__(self):
        super(HRSegNet, self).__init__()

        hrnet_config_extra = {
            "WITH_HEAD": True,
            "STAGE1": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 1,
                "BLOCK": "BOTTLENECK",
                "NUM_BLOCKS": [2],
                "NUM_CHANNELS": [64],
                "FUSE_METHOD": "SUM",
            },
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2],
                "NUM_CHANNELS": [18, 36],
                "FUSE_METHOD": "SUM",
            },
            "STAGE3": {
                "NUM_MODULES": 3,
                "NUM_BRANCHES": 3,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2],
                "NUM_CHANNELS": [18, 36, 72],
                "FUSE_METHOD": "SUM",
            },
            "STAGE4": {
                "NUM_MODULES": 2,
                "NUM_BRANCHES": 4,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2, 2],
                "NUM_CHANNELS": [18, 36, 72, 144],
                "FUSE_METHOD": "SUM",
            },
        }

        # Sub-Network Definition
        self.backbone = HighResolutionNet(hrnet_config_extra)
        self.seg = SegmentationHead(4, hrnet_config_extra)

    def forward(self, im):
        features = self.backbone(im)
        seg_logits = self.seg(*features)
        return seg_logits


def vis_seg_results(seg_scores, im_path):
    _im = cv2.imread(im_path)
    for _msk, _clr in zip(seg_scores, colors):
        colored_msk = (np.repeat(_msk[..., None], 3, axis=2) * _clr).astype(np.uint8)
        resized_colored_msk = cv2.resize(colored_msk, _im.shape[:2][::-1], interpolation=cv2.INTER_AREA)
        _im = cv2.addWeighted(_im, 1, resized_colored_msk, 0.5, 0)
    save_path = vis_dir / Path(im_path).relative_to(test_image_dir)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), _im)


# test_image_dir = "/kaggle/input/severstal-steel-defect-detection/test_images"
test_image_dir = "/data2/datasets/kaggle/severstal-steel-defect-detection/raw/test"
model_path = "/home/rlan/deploy/pytorch-detection/checkpoints/severstal/w18smallv2_bs14_alltransform#SeverstalSteelDefectDetection#HRSegNet#0.01#200#best.pth"
vis_dir = Path('/tmp/vis-eval')
colors = get_colors(4, cmap_name="tab10")
transforms = [
    ZeroBlackMargin(black_thresh=30),
    Div(by=255.0),
    Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ToTensor(),
]
dataset = SeverstalSteelDefectDetection(test_image_dir, transforms=transforms)
dataloader = DataLoader(dataset)
model = HRSegNet()
model.load_state_dict(torch.load(model_path)['integrated_model'])
model.cuda()

for im_paths, im in tqdm(dataloader):
    with torch.no_grad():
        seg_logits = model(im.cuda())
        seg_scores = (seg_logits > 0).cpu().numpy()[0]
        vis_seg_results(seg_scores, im_paths[0])
