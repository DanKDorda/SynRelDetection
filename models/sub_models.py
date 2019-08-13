import torch
from torch import nn
import numpy as np
from data.proposal_dataset import ProposalDs


class FeatureNet(nn.Module):
    def __init__(self, opts):
        super(FeatureNet, self).__init__()
        self.opts = opts

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(num_features=8)
        self.relu0 = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(num_features=32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.relu3 = nn.ReLU()

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.relu4 = nn.ReLU()

        self.resnet = ResnetBlock(128)

        self.conv5 = nn.Conv2d(128, self.opts.FeatNet.feat_out, 4, 1, 0)

        self.av_pool = nn.AvgPool2d(kernel_size=4)

    def forward(self, im_in):
        output = self.relu0(self.bn0(self.conv0(im_in)))
        output = self.relu1(self.bn1(self.conv1(output)))
        output = self.pool1(output)  # 48 x 48
        output = self.relu2(self.bn2(self.conv2(output)))  # 25 x 25
        output = self.relu3(self.bn3(self.conv3(output)))  # 13 x 13
        output = self.pool3(output)
        output = self.relu4(self.bn4(self.conv4(output)))
        # output = self.av_pool(output)
        output = self.resnet(output)
        output = self.conv5(output)
        return output


class GraphProposalNetwork(nn.Module):
    def __init__(self, opts):
        super(GraphProposalNetwork, self).__init__()
        self.opts = opts
        self.N_heads = opts.GPN.N_heads
        self.feat_in = opts.GPN.feat_in
        self.geometry_feat = 4
        self.feat_out = 256
        self.feat_concat = self.feat_out * 2

        self.preliminary_transform = nn.Linear(self.feat_in, self.feat_out)
        self.geometry_add = nn.Sequential(nn.Linear(self.geometry_feat, 32), nn.LeakyReLU(0.02),
                                          nn.Linear(32, self.feat_out))
        self.geometry_mult = nn.Sequential(nn.Linear(self.geometry_feat, 32), nn.LeakyReLU(0.02),
                                           nn.Linear(32, self.feat_out))
        self.attention_net = nn.Sequential(nn.Linear(self.feat_concat, 128), nn.LeakyReLU(0.02),
                                           nn.Linear(128, self.N_heads))
        self.connectivity_net = nn.Sequential(nn.Linear(self.feat_concat, 128), nn.ReLU(),
                                              nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, object_features, scene_geometry, d_max=10):
        # object_features is a list of tensors of D x 128 x 1 x 1
        # input tensor dim = batch x D_max x feature_len

        # visual_features = torch.cat((object_features, scene_geometry), 2)

        # adjacency_tensor = torch.zeros(len(object_features), d_max, d_max, self.N_heads)

        embed_vertices = self.preliminary_transform(
            torch.cat([of.unsqueeze(0) for of in object_features]).view(self.opts.batch_size, 10, self.feat_in))
        embed_geometry_m = self.geometry_mult(scene_geometry)
        embed_geometry_a = self.geometry_add(scene_geometry)

        embed_vertices = embed_vertices * embed_geometry_m
        embed_vertices = embed_vertices + embed_geometry_a
        mega_compound_tensor = torch.zeros(len(object_features), d_max, self.feat_concat)

        for batch_idx, batch_vertices in enumerate(embed_vertices.split(1)):
            # batch_vertices.squeeze_(2)
            # batch_vertices.squeeze_(2)
            # geom_cat = torch.cat((batch_vertices, scene_geometry[batch_idx, ...]), dim=1)
            # embed_vertices = self.preliminary_transform(batch_vertices)
            # embed_vertices = embed_vertices * embed_geometry_m[batch_idx]
            # embed_vertices = embed_vertices + embed_geometry_a[batch_idx]
            batch_vertices.squeeze_(0)
            vlist = torch.split(batch_vertices, 1)

            for i, vi in enumerate(vlist):
                compound_tensor = torch.cat((vi, vi), dim=1)
                for j, vj in enumerate([v for k, v in enumerate(vlist) if k != i]):
                    compound_tensor = torch.cat((compound_tensor, torch.cat((vi, vj), dim=1)))
                # compound_tensor = compound_tensor[1:]
                mega_compound_tensor[batch_idx] = compound_tensor
                # edge_vals = self.connectivity_net(compound_tensor)
                # adjacency_tensor[batch_idx, i] = edge_vals.transpose(1, 0)

        big_edge = self.connectivity_net(mega_compound_tensor)
        adjacency_tensor = big_edge.permute(0, 2, 1)
        return adjacency_tensor


class ResnetBlock(nn.Module):
    def __init__(self, num_lay):
        super(ResnetBlock, self).__init__()
        self.conv = nn.Conv2d(num_lay, num_lay, 3, 1, 1)
        self.norm = nn.BatchNorm2d(num_lay)
        self.relu = nn.ReLU()

    def forward(self, x, leakage=0.1):
        return self.relu(self.norm(self.conv(x))) + leakage * x
