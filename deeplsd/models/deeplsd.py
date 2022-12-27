"""
Regress the distance function map to all the line segments of an image.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .base_model import BaseModel
from .backbones.vgg_unet import VGGUNet


class DeepLSD(BaseModel):
    default_conf = {
        'tiny': False,
        'sharpen': True,
        'line_neighborhood': 5,
        'loss_weights': {
            'df': 1.,
            'angle': 1.,
        },
        'multiscale': False,
        'scale_factors': [1., 1.5],
        'detect_lines': False,
        'line_detection_params': {
            'merge': False,
            'grad_nfa': True,
            'optimize': False,
            'use_vps': False,
            'optimize_vps': False,
            'filtering': 'normal',
            'grad_thresh': 3,
            'lambda_df': 1.,
            'lambda_grad': 1.,
            'lambda_vp': 0.5,
        },
    }
    required_data_keys = ['image']

    def _init(self, conf):
        # Base network
        self.backbone = VGGUNet(tiny=self.conf.tiny)
        dim = 32 if self.conf.tiny else 64

        # Predict the distance field and angle to the nearest line
        # DF head
        self.df_head = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(),
        )

        # Closest line direction head
        self.angle_head = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Loss
        self.l1_loss_fn = nn.L1Loss(reduction='none')
        self.l2_loss_fn = nn.MSELoss(reduction='none')

    def normalize_df(self, df):
        return -torch.log(df / self.conf.line_neighborhood + 1e-6)

    def denormalize_df(self, df_norm):
        return torch.exp(-df_norm) * self.conf.line_neighborhood

    def _forward(self, image):

        if self.conf.multiscale:
            df, line_level = self.ms_forward(image)
        else:
            base = self.backbone(image)

            # DF prediction
            if self.conf.sharpen:
                df_norm = self.df_head(base).squeeze(1)
                df = self.denormalize_df(df_norm)
            else:
                df = self.df_head(base).squeeze(1)

            # Closest line direction prediction
            line_level = self.angle_head(base).squeeze(1) * np.pi

        return df, line_level

    def ms_forward(self, image):
        """ Do several forward passes at multiple image resolutions
            and aggregate the results before extracting the lines. """
        img_size = image.shape[2:]

        # Forward pass for each scale
        pred_df, pred_angle = [], []
        for s in self.conf.scale_factors:
            img = F.interpolate(image, scale_factor=s, mode='bilinear')
            with torch.no_grad():
                base = self.backbone(img)
                if self.conf.sharpen:
                    pred_df.append(self.denormalize_df(self.df_head(base)))
                else:
                    pred_df.append(self.df_head(base))
                pred_angle.append(self.angle_head(base) * np.pi)\

        # Fuse the outputs together
        for i in range(len(self.conf.scale_factors)):
            pred_df[i] = F.interpolate(pred_df[i], img_size,
                                       mode='bilinear').squeeze(1)
            pred_angle[i] = F.interpolate(pred_angle[i], img_size,
                                          mode='nearest').squeeze(1)
        fused_df = torch.stack(pred_df, dim=0).mean(dim=0)
        fused_angle = torch.median(torch.stack(pred_angle, dim=0), dim=0)[0]

        return fused_df, fused_angle

    def loss(self, pred, data):
        outputs = {}
        loss = 0

        # Retrieve the mask of valid pixels
        valid_mask = data['ref_valid_mask']
        valid_norm = valid_mask.sum(dim=[1, 2])
        valid_norm[valid_norm == 0] = 1

        # Retrieve the mask of pixels close to GT lines
        line_mask = (valid_mask
                     * (data['df'] < self.conf.line_neighborhood).float())
        line_norm = line_mask.sum(dim=[1, 2])
        line_norm[line_norm == 0] = 1

        # DF loss, with supervision only on the lines neighborhood
        if self.conf.sharpen:
            df_loss = self.l1_loss_fn(pred['df_norm'],
                                      self.normalize_df(data['df']))
        else:
            df_loss = self.l1_loss_fn(pred['df'], data['df'])
            df_loss /= self.conf.line_neighborhood
        df_loss = (df_loss * line_mask).sum(dim=[1, 2]) / line_norm
        df_loss *= self.conf.loss_weights.df
        loss += df_loss
        outputs['df_loss'] = df_loss

        # Angle loss, with supervision only on the lines neighborhood
        angle_loss = torch.minimum(
            (pred['line_level'] - data['line_level']) ** 2,
            (np.pi - (pred['line_level'] - data['line_level']).abs()) ** 2)
        angle_loss = (angle_loss * line_mask).sum(dim=[1, 2]) / line_norm
        angle_loss *= self.conf.loss_weights.angle
        loss += angle_loss
        outputs['angle_loss'] = angle_loss

        outputs['total'] = loss
        return outputs

    def metrics(self, pred, data):
        return {}
