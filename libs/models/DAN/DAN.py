#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Haoxin Chen
# @File    : DAN.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        resnet = models.resnet50(pretrained=True, progress=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                    resnet.maxpool)
        self.layer1 = resnet.layer1  # 1/4, 256
        self.layer2 = resnet.layer2  # 1/8, 512
        self.layer3 = resnet.layer3  # 1/16, 1024
        self.layer4 = resnet.layer4  # 1/32, 2048
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        x = self.layer0(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        return [l1, l2, l3, l4]


class ResBlock(nn.Module):

    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim is None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim,
                                        outdim,
                                        kernel_size=3,
                                        padding=1,
                                        stride=stride)

        self.conv1 = nn.Conv2d(indim,
                               outdim,
                               kernel_size=3,
                               padding=1,
                               stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Refine(nn.Module):

    def __init__(self, inplanes, planes):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes,
                                planes,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(
            pm, size=s.shape[2:], mode='bilinear', align_corners=True)
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):

    def __init__(self, inplane, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(inplane,
                                mdim,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim)  # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim,
                               1,
                               kernel_size=(3, 3),
                               padding=(1, 1),
                               stride=1)

    def forward(self, r4, r3, r2, f):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4)  # out: 1/8, 256
        m2 = self.RF2(r2, m3)  # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))

        p = F.interpolate(p2,
                          size=f.shape[2:],
                          mode='bilinear',
                          align_corners=True)
        return p


class QueryKeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(QueryKeyValue, self).__init__()
        self.query = nn.Conv2d(indim,
                               keydim,
                               kernel_size=3,
                               padding=1,
                               stride=1)
        self.Key = nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.Value = nn.Conv2d(indim,
                               valdim,
                               kernel_size=3,
                               padding=1,
                               stride=1)

    def forward(self, x):
        return self.query(x), self.Key(x), self.Value(x)


class DomainAgentAttention(nn.Module):

    def __init__(self, in_dim, key_dim, val_dim):
        super(DomainAgentAttention, self).__init__()
        self.q_dim = key_dim
        self.k_dim = key_dim
        self.v_dim = val_dim
        self.support_qkv = QueryKeyValue(indim=in_dim,
                                         keydim=key_dim,
                                         valdim=val_dim)
        self.query_qkv = QueryKeyValue(indim=in_dim,
                                       keydim=key_dim,
                                       valdim=val_dim)

    def attention(self, Q, K, V):
        '''
        Q: Batch, Length, Channels
        K: Batch, Length, Channels
        V: Batch, Length, Channels
        out: Batch, Length, Channels
        '''
        QK = torch.bmm(Q, K.transpose(1, 2))
        QK = QK / (K.shape[-1]**0.5)
        QK = F.softmax(QK, dim=2)
        out = torch.bmm(QK, V)
        return out

    def forward(self, support, query):
        '''
        support: batch_size, support_frames, channels, height, width
        query: batch_size, query_frames, channels, height, width
        '''
        b, _, c, h, w = support.shape
        support_frames = support.shape[1]
        query_frames = query.shape[1]

        _, support_k, support_v = self.support_qkv(
            support.view(b * support_frames, c, h, w))
        query_q, query_k, _ = self.query_qkv(
            query.view(b * query_frames, c, h, w))

        support_k = support_k.view(b, support_frames, self.k_dim, h, w)
        support_v = support_v.view(b, support_frames, self.v_dim, h, w)

        query_q = query_q.view(b, query_frames, self.q_dim, h, w)
        query_k = query_k.view(b, query_frames, self.k_dim, h, w)

        domain_frame_idx = int(query_frames / 2)
        domain_q = query_q[:, domain_frame_idx, :, :, :]
        domain_k = query_k[:, domain_frame_idx, :, :, :]

        # attention(domain_q, support_k, support_v)
        domain_q = domain_q.view(b, 1, self.q_dim * h * w)
        support_k = support_k.view(b, support_frames, self.k_dim * h * w)
        support_v = support_v.view(b, support_frames, self.v_dim * h * w)
        domain_attention = self.attention(domain_q, support_k, support_v)

        # attention(query_q, domain_k, domain_attention)
        query_q = query_q.view(b, query_frames, self.q_dim * h * w)
        domain_k = domain_k.view(b, 1, self.k_dim * h * w)
        query_attention = self.attention(query_q, domain_k, domain_attention)

        query_attention = query_attention.view(b, query_frames, self.v_dim, h,
                                               w)

        return query_attention


class DAN(nn.Module):

    def __init__(self):
        super(DAN, self).__init__()
        self.encoder = Encoder()  # output 2048
        encoder_dim = 1024
        self.k_dim = 128
        self.v_dim = int(encoder_dim / 2)

        self.daa = DomainAgentAttention(encoder_dim, self.k_dim, self.v_dim)

        self.conv_q = nn.Conv2d(encoder_dim,
                                self.v_dim,
                                kernel_size=1,
                                stride=1,
                                padding=0)

        # low_level_features to 48 channels
        self.Decoder = Decoder(encoder_dim, 256)

    def forward(self, img, support_image, support_mask, time=None):

        batch, frame, in_channels, height, width = img.shape
        _, sframe, mask_channels, Sheight, Swidth = support_mask.shape
        assert height == Sheight
        assert width == Swidth

        batch_frame = batch * frame
        img = img.view(-1, in_channels, height, width)
        support_image = support_image.view(-1, in_channels, height, width)
        support_mask = support_mask.view(-1, mask_channels, height, width)

        in_f = torch.cat((img, support_image), dim=0)
        encoder_f = self.encoder(in_f)

        encoder_f_l1 = encoder_f[0]
        encoder_f_l2 = encoder_f[1]
        encoder_f_l3 = encoder_f[2]
        # encoder_f_l4 = encoder_f[3]

        if time is not None:
            time.t1()

        support_mask = F.interpolate(support_mask,
                                     encoder_f_l3.size()[2:],
                                     mode='bilinear',
                                     align_corners=True)
        query_feat_l1 = encoder_f_l1[:batch_frame]
        query_feat_l2 = encoder_f_l2[:batch_frame]
        query_feat = encoder_f_l3[:batch_frame]
        support_feat = encoder_f_l3[batch_frame:]

        support_fg_feat = support_feat * support_mask
        # support_bg_feat = support_feat * (1 - support_mask)

        # Domain Agent Attention
        support_fg_feat = support_fg_feat.view(batch, sframe,
                                               support_fg_feat.shape[1],
                                               support_fg_feat.shape[2],
                                               support_fg_feat.shape[3])
        query_feat = query_feat.view(batch, frame, query_feat.shape[1],
                                     query_feat.shape[2], query_feat.shape[3])
        after_transform = self.daa(support_fg_feat, query_feat)

        # [batch*frames, 1024, h/16,w/16]
        after_transform = after_transform.view(batch * frame,
                                               after_transform.shape[2],
                                               after_transform.shape[3],
                                               after_transform.shape[4])

        query_feat = self.conv_q(query_feat)
        after_transform = torch.cat((after_transform, query_feat), dim=1)
        # aspp
        # x = self.aspp(after_transform)
        if time is not None:
            time.t2()
        x = self.Decoder(after_transform, query_feat_l2, query_feat_l1, img)

        pred_map = torch.nn.Sigmoid()(x)

        # batch, frame, outchannel, height, width
        pred_map = pred_map.view(batch, frame, 1, height, width)
        return pred_map


if __name__ == '__main__':
    model = DAN()
    img = torch.FloatTensor(2, 3, 3, 224, 224)
    support_mask = torch.FloatTensor(2, 5, 1, 30, 30)
    support_img = torch.FloatTensor(2, 5, 3, 30, 30)
    pred_map = model(img, support_img, support_mask)
    print(pred_map.shape)
