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
        self.feat_in = 3  # opts.GPN.feat_in

        if self.opts.GPN.net_mode == 'complex':
            self.feat_out = 64
            self.feat_concat = self.feat_out * 2
            self.preliminary_transform = nn.Sequential(nn.Linear(self.feat_in, 32), nn.LeakyReLU(0.2),
                                                       nn.Linear(32, self.feat_out))
            self.connectivity_net = nn.Sequential(nn.Linear(self.feat_concat, 32), nn.LeakyReLU(0.2),
                                                  nn.Linear(32, 8), nn.LeakyReLU(0.2), nn.Linear(8, 1))
        elif self.opts.GPN.net_mode == 'simple':
            self.feat_out = 16
            self.feat_concat = self.feat_out * 2
            self.preliminary_transform = nn.Linear(self.feat_in, self.feat_out)
            self.connectivity_net = nn.Sequential(nn.Linear(self.feat_concat, 32), nn.LeakyReLU(0.2),
                                                  nn.Linear(32, 8), nn.LeakyReLU(0.2), nn.Linear(8, 2))

    def forward(self, position_tensor, orientation_tensor, d_max=10):
        # inputs are the object positions and orientations

        # size: batch x n_obj x 3
        full_feature_tensor = torch.cat([position_tensor, orientation_tensor], dim=2)
        full_feature_tensor = self.preliminary_transform(full_feature_tensor)

        # create pairwise feature groupings
        mega_compound_tensor = torch.empty(self.opts.batch_size, 10, 10, self.feat_concat)
        for batch_idx, batch_vertices in enumerate(full_feature_tensor.split(1)):
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
                mega_compound_tensor[batch_idx, i] = compound_tensor
                # edge_vals = self.connectivity_net(compound_tensor)
                # adjacency_tensor[batch_idx, i] = edge_vals.transpose(1, 0)

        big_edge = self.connectivity_net(mega_compound_tensor)
        # big_edge.squeeze_(3)
        #big_edge = big_edge.permute(0, 2, 1, 3)
        # print(big_edge.shape)
        # adjacency_tensor = big_edge.permute(0, 2, 1, 3)
        adjacency = torch.argmax(big_edge, dim=3)
        return big_edge, adjacency


class ResnetBlock(nn.Module):
    def __init__(self, num_lay):
        super(ResnetBlock, self).__init__()
        self.conv = nn.Conv2d(num_lay, num_lay, 3, 1, 1)
        self.norm = nn.BatchNorm2d(num_lay)
        self.relu = nn.ReLU()

    def forward(self, x, leakage=0.1):
        return self.relu(self.norm(self.conv(x))) + leakage * x
