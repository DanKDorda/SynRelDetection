import numpy as np
import torch
import torch.nn as nn

import copy

import utils
from models.sub_models import FeatureNet, GraphProposalNetwork
from models.self_super_transformer_lays import FinalPredictor


class SyntheticGraphLearner(nn.Module):
    def __init__(self, opts):
        super(SyntheticGraphLearner, self).__init__()
        self.opts = opts
        self.method = opts.train.method
        self.use_cuda = self.opts.train.cuda

        # inputs!
        self.image, self.objects, self.relationships, self.indices_removed = torch.FloatTensor(), [], [], []

        # Declare output properties
        self.adjacency_tensor = torch.FloatTensor()
        self.gt_adjacency_tensor = torch.FloatTensor()
        self.connectivity_matrix = torch.LongTensor()
        self.gt_connectivity_matrix = torch.LongTensor()
        self.predicted_image = torch.FloatTensor()
        self.eval_dict = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

        # Define networks
        self.feature_net = FeatureNet(opts)
        self.graph_proposal_net = GraphProposalNetwork(opts)
        self.final_predictor = FinalPredictor(opts)

        if self.use_cuda:
            self.feature_net.cuda()
            self.graph_proposal_net.cuda()
            self.final_predictor.cuda()

        # Define optimizers
        param_list = list(self.feature_net.parameters()) + list(self.graph_proposal_net.parameters())
        self.supervised_optimizer = torch.optim.Adam(param_list, lr=self.opts.train.lr)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.supervised_optimizer,
                                                         self.opts.train.schedule.lr_step_every,
                                                         gamma=self.opts.train.schedule.gamma)

        # now for the unsupervised part...
        self.unsupervised_optimizer = torch.optim.Adam(param_list + list(self.feature_net.parameters()),
                                                       lr=self.opts.train.lr)
        # TODO: scheduler
        # Define loss functions
        self.l1_critetion = nn.L1Loss()
        self.softmax_criterion = nn.CrossEntropyLoss()
        self.loss = torch.FloatTensor()

    def forward(self, input_data):
        #### PAIRWISE METHOD ==> LIST OPTION

        # get detections
        self.image, self.objects, self.relationships, self.indices_removed = input_data['visual'], input_data[
            'objects'], input_data['relationships'], input_data['indices_removed']

        d_max = max([len(obj) for obj in self.objects])
        self.gt_adjacency_tensor = utils.rel_list_to_adjacency_tensor(self.relationships, self.opts.batch_size,
                                                                      d_max)
        self.gt_connectivity_matrix = utils.rel_list_to_connectivity_matrix(self.relationships, self.opts.batch_size,
                                                                            d_max)
        ### MOVE TO GPU
        if self.use_cuda:
            self.gt_adjacency_tensor.cuda()
            self.image.cuda()

        if self.method == 'unsupervised':
            #image_masked, chosen_idx = self.masker(self.image, self.objects)
            chosen_idx = 3

        # find features from images -> list
        imagelets_batched = self.get_imagelets(self.image, self.objects)

        vertex_feature_list = [self.feature_net(imagelets) for imagelets in imagelets_batched]

        # proposal of edges from object features and geometry
        geometry_tensor = torch.zeros(self.opts.batch_size, d_max, 4)

        for i, img_obs in enumerate(self.objects):
            for j, obj in enumerate(img_obs):
                bb = obj['bbox']
                mid_x = int((bb[0] + bb[1]) / 2)
                mid_y = int((bb[2] + bb[3]) / 2)
                size_x = abs(int((bb[0] - bb[1])))
                size_y = abs(int((bb[2] - bb[3])))
                geometry_tensor[i, j, :] = torch.tensor([mid_x, mid_y, size_x, size_y])

        self.connectivity_matrix = self.graph_proposal_net(vertex_feature_list, geometry_tensor)

        with torch.no_grad():
            self.connectivity_argmaxed = torch.zeros_like(self.connectivity_matrix)
            self.connectivity_argmaxed.scatter_(2, torch.argmax(self.connectivity_matrix, dim=2, keepdim=True), 1)
        # with torch.no_grad():
        #     re_list = utils.adjacency_tensor_to_rel_list(self.adjacency_tensor)
        # print(ct)
        if self.method == 'unsupervised':
            # propose image for missing boy
            proposals = self.final_predictor.generate_image_proposals(self.objects, chosen_idx)
            proposal_feats = self.feature_net(proposals)
            predicted_image = self.final_predictor(vertex_feature_list, self.adjacency_tensor, chosen_idx)
            self.predicted_image = predicted_image

    def compute_loss(self):
        if self.method == 'supervised':
            # self.loss = 0 #self.l1_critetion(self.adjacency_tensor, self.gt_adjacency_tensor) * 0.1

            ce_loss = 0
            bad_idcs = self.indices_removed
            items_left = 5  # sum(bad_idcs[0])
            num_items = 10
            # small_one = torch.zeros(self.opts.batch_size, items_left, num_items)
            # small_one.scatter_(2, bad_idcs, self.connectivity_matrix)
            for b in range(self.opts.batch_size):
                i = 0
                for row, row_gt, bad_idx in zip(self.connectivity_matrix[b].split(1),
                                                self.gt_connectivity_matrix[b].split(1), bad_idcs[b]):
                    if bad_idx == 0:
                        # small_one[b, i] = row
                        target = torch.argmax(row_gt)
                        ce_loss += self.softmax_criterion(row, target.unsqueeze(0))

            # for row_idx, row in enumerate(self.connectivity_matrix.split(1, dim=1)):
            #     if row_idx in bad_idcs:
            #         continue
            #     target = torch.argmax(self.gt_connectivity_matrix[:, row_idx, :], dim=1)
            #     row.squeeze_(1)
            #
            #     ce_loss += self.softmax_criterion(row, target)
            self.loss = ce_loss
        elif self.method == 'unsupervised':
            #self.loss = self.l1_criterion(self.predicted_image, self.desired_out)
            raise NotImplementedError('no unsupervised loss implemented')

    def optimize_params(self):
        if self.method == 'supervised':
            self.supervised_optimizer.zero_grad()
            self.loss.backward()
            self.supervised_optimizer.step()
        else:
            raise NotImplementedError('only supervised implemented')

    def evaluate(self, input_data):
        self.forward(input_data)

        TP = torch.sum(
            torch.where((self.gt_connectivity_matrix == 1) * (self.connectivity_argmaxed == 1), torch.tensor(1),
                        torch.tensor(0)))
        FP = torch.sum(
            torch.where((self.gt_connectivity_matrix == 0) * (self.connectivity_argmaxed == 1), torch.tensor(1),
                        torch.tensor(0)))
        TN = torch.sum(
            torch.where((self.gt_connectivity_matrix == 0) * (self.connectivity_argmaxed == 0), torch.tensor(1),
                        torch.tensor(0)))
        FN = torch.sum(
            torch.where((self.gt_connectivity_matrix == 1) * (self.connectivity_argmaxed == 0), torch.tensor(1),
                        torch.tensor(0)))

        self.eval_dict['TP'] += TP.item()
        self.eval_dict['FP'] += FP.item()
        self.eval_dict['TN'] += TN.item()
        self.eval_dict['FN'] += FN.item()

    def scheduler_step(self):
        self.scheduler.step()

    def get_eval_dict(self):
        return self.eval_dict

    def clear_eval_dict(self):
        self.eval_dict = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

    @staticmethod
    def masker(image_tensor, object_batch):
        # TESTING REQUIRED
        # NOT SURE THIS WORKS AS INTENDED
        # ALSO POTENTIAL PROBLEM OF

        image_masked = copy.deepcopy(image_tensor)
        masked_objects = []
        # pretend it's a list of dicts
        for i, object_list in enumerate(object_batch):
            masked_object_idx = np.random.choice(len(object_list))
            bb = object_list[masked_object_idx]['bbox']
            image_masked[i, :, bb[2]:bb[3], bb[0]:bb[1]] = 0
            masked_objects.append(masked_object_idx)

        return image_masked, masked_objects

    def get_imagelets(self, image, objects):
        # list option
        imagelet_batch = []

        if self.use_cuda:
            imagelet_base = torch.cuda.FloatTensor
        else:
            imagelet_base = torch.FloatTensor

        for b in range(self.opts.batch_size):
            bboxes = [obj['bbox'] for obj in objects[b]]
            imagelets = imagelet_base(len(bboxes), 3, 96, 96)
            for i, bb in enumerate(bboxes):
                imagelets[i, ...] = image[b, :, bb[2]:bb[3], bb[0]:bb[1]]
            imagelet_batch.append(imagelets)

        # # tensor_option
        # # feat dimension: Batch x Dmax X d
        # # imagelet dim: Batch x Dmax X c x h x w
        # D_max = max([len(a['objects']) for a in annotations])
        # c = 3
        # h = 32
        # w = 32
        # imagelets = torch.zeros(self.opts.batch_size,D_max, c, w, h)
        #
        # if self.use_cuda:
        #     imagelets.cuda()
        #
        # for i, image_annotation in enumerate(annotations):
        #     for j, detection in enumerate(image_annotation['objects']):
        #         bbox = detection['bbox']
        #         imagelets[i, j, ...] = image[i, :, bbox[2]:bbox[3], bbox[0]:bbox[1]]

        return imagelet_batch

    def get_loss(self):
        return self.loss.detach().item()

    def get_image_output(self):
        sv = utils.SceneVisualiser()
        sv.visualise(self.image[0], self.objects[0], self.connectivity_argmaxed[0])

        return sv.scene


if __name__ is '__main__':
    print('test mode')

    import json
    import os.path as osp
    import torchvision.transforms as T
    from PIL import Image
    import janky_trainloop


    def peek_at_im(tensor):
        Image.fromarray((np.array(tensor[0] * 255).astype('uint8')).transpose(1, 2, 0)).show()


    # open an image and its annotations
    data_root = '/Users/i517610/PycharmProjects/SynRelDetection/datasets/sr3/train'
    annotations = json.load(open(osp.join(data_root, 'scene_info.json')))
    lucky_idx = np.random.choice(len(annotations))

    im = Image.open(osp.join(data_root, 'images', str(lucky_idx) + '.jpg'))
    transform = T.Compose([T.ToTensor()])
    im_torch = transform(im)
    im_torch = im_torch.unsqueeze(0)
    lucky_annotation = annotations[lucky_idx]
    data = {'visual': im_torch,
            'objects': [lucky_annotation['objects']],
            'relationships': [lucky_annotation['relationships']]}

    opts = janky_trainloop.get_opts()
    opts.batch_size = 1
    model = SyntheticGraphLearner(opts)
    model.eval()
    model.evaluate(data)

    img = model.get_image_output()
    Image.fromarray(img).show()
