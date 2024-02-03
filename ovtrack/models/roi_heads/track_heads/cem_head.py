import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models import HEADS, build_loss


@HEADS.register_module(force=True)
class ClsExemplarHead(nn.Module):
    def __init__(
        self,
        num_convs=4,
        num_fcs=1,
        roi_feat_size=7,
        in_channels=256,
        conv_out_channels=256,
        fc_out_channels=1024,
        embed_channels=256,
        conv_cfg=None,
        norm_cfg=None,
        softmax_temp=-1,
        loss_track=dict(type="MultiPosCrossEntropyLoss", loss_weight=1),
    ):
        super(ClsExemplarHead, self).__init__()

        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.embed_channels = embed_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.relu = nn.ReLU(inplace=True)
        self.convs, self.fcs, last_layer_dim = self._add_conv_fc_branch(
            self.num_convs, self.num_fcs, self.in_channels
        )
        self.fc_embed = nn.Linear(last_layer_dim, embed_channels)

        self.softmax_temp = softmax_temp
        self.loss_track = build_loss(loss_track)

    def _add_conv_fc_branch(self, num_convs, num_fcs, in_channels):
        last_layer_dim = in_channels
        # add branch specific conv layers
        convs = nn.ModuleList()
        if num_convs > 0:
            for i in range(num_convs):
                conv_in_channels = last_layer_dim if i == 0 else self.conv_out_channels
                convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                    )
                )
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        fcs = nn.ModuleList()
        if num_fcs > 0:
            last_layer_dim *= self.roi_feat_size * self.roi_feat_size
            for i in range(num_fcs):
                fc_in_channels = last_layer_dim if i == 0 else self.fc_out_channels
                fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return convs, fcs, last_layer_dim

    def init_weights(self):

        for m in self.fcs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.fc_embed.weight, 0, 0.01)
        nn.init.constant_(self.fc_embed.bias, 0)

    def forward(self, x):

        if self.num_convs > 0:
            for i, conv in enumerate(self.convs):
                x = conv(x)
        x = x.view(x.size(0), -1)
        if self.num_fcs > 0:
            for i, fc in enumerate(self.fcs):
                x = self.relu(fc(x))
        x = self.fc_embed(x)

        return x

    def sup_contra_loss(self, features, labels):

        losses = dict()
        loss_track = self.loss_track(features, labels)
        losses["loss_cem"] = loss_track

        return losses
