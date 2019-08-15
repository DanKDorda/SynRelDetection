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
        self.predicted_orientation = torch.empty(4, 10, 10)
        self.target = torch.LongTensor()

        # Define networks
        self.graph_proposal_net = GraphProposalNetwork(opts)
        if self.method == 'unsupervised' or self.method == 'joint':
            self.final_predictor = FinalPredictor(opts)

        if self.use_cuda:
            self.graph_proposal_net.cuda()
            if self.method == 'unsupervised' or self.method == 'joint':
                self.final_predictor.cuda()

        # Define optimizers
        param_list = list(self.graph_proposal_net.parameters())
        self.supervised_optimizer = torch.optim.Adam(param_list, lr=self.opts.train.lr)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.supervised_optimizer,
                                                         self.opts.train.schedule.lr_step_every,
                                                         gamma=self.opts.train.schedule.gamma)

        # now for the unsupervised part...
        if self.method == 'unsupervised' or self.method == 'joint':
            self.unsupervised_optimizer = torch.optim.Adam(param_list + list(self.final_predictor.parameters()),
                                                           lr=self.opts.train.lr)
        # TODO: scheduler
        # Define loss functions
        self.l1_critetion = nn.L1Loss()
        self.softmax_criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.opts.train.ce_weight).float(), reduction='sum')
        #self.bce = nn.BCEWithLogitsLoss()
        self.loss = torch.FloatTensor()

    def forward(self, input_data):
        #### PAIRWISE METHOD ==> LIST OPTION

        # get detections
        self.image, self.objects, self.relationships, self.indices_removed = input_data['visual'], input_data[
            'objects'], input_data['relationships'], input_data['indices_removed']

        d_max = max([len(obj) for obj in self.objects])
        #self.gt_adjacency_tensor = utils.rel_list_to_adjacency_tensor(self.relationships, self.opts.batch_size, d_max)

        self.gt_connectivity_matrix = utils.rel_list_to_connectivity_matrix(self.relationships, self.opts.batch_size,
                                                                            d_max)

        position_tensor, orientation_tensor = utils.get_positions_and_orients(self.objects)
        self.full_feat = torch.cat([position_tensor, orientation_tensor], dim=2)

        ### MOVE TO GPU
        if self.use_cuda:
            self.gt_adjacency_tensor.cuda()
            self.image.cuda()
            position_tensor.cuda()
            orientation_tensor.cuda()

        if self.method == 'unsupervised' or self.method == 'joint':
            # image_masked, chosen_idx = self.masker(self.image, self.objects)
            chosen_idx = np.random.randint(0, 9)
            # chosen_idx = 3

        # proposal of edges from object features and geometry
        self.raw_score, self.connectivity_matrix = self.graph_proposal_net(position_tensor, orientation_tensor)

        with torch.no_grad():
            self.connectivity_argmaxed = self.connectivity_matrix
            # self.connectivity_argmaxed = torch.zeros_like(self.connectivity_matrix)
            # self.connectivity_argmaxed.scatter_(2, torch.argmax(self.connectivity_matrix, dim=2, keepdim=True), 1)

        if self.method == 'unsupervised' or self.method == 'joint':
            # propose image for missing boy
            # proposals = self.final_predictor.generate_image_proposals(self.objects, chosen_idx)
            # proposals, self.target = utils.propose_orientations(orientation_tensor, chosen_idx)
            # proposal_feats = self.feature_net(proposals)
            self.chosen_idx = chosen_idx
            self.predicted_orientation = self.final_predictor(position_tensor, orientation_tensor, self.raw_score,
                                                   chosen_idx)

    def compute_loss(self):
        self.loss = torch.tensor([0.0], dtype=torch.float32) #self.l1_critetion(self.adjacency_tensor, self.gt_adjacency_tensor) * 0.1
        self.sup_loss = 0
        self.unsup_loss = 0

        ce_loss = 0
        self.sup_loss = ce_loss

        if self.method == 'supervised' or self.method == 'joint':
            bad_idcs = self.indices_removed
            # for b in range(self.opts.batch_size):
            #     for row, row_gt, bad_idx in zip(self.raw_score[b].split(1),
            #                                     self.gt_connectivity_matrix[b].split(1), bad_idcs[b]):
            #         if bad_idx == 0 or True:
            #             target = torch.argmax(row_gt)
            #             ce_loss += self.softmax_criterion(row, target.unsqueeze(0))

            target = self.gt_connectivity_matrix.long()
            ce_loss += self.softmax_criterion(self.raw_score.permute(0, 3, 1, 2), target)
            #ce_loss += self.bce(self.raw_score, self.gt_connectivity_matrix)

            self.loss += ce_loss
            self.sup_loss = ce_loss

        if self.method == 'unsupervised' or self.method == 'joint':
            # self.loss = self.l1_criterion(self.predicted_image, self.desired_out)
            #ce_loss_2 = self.softmax_criterion(self.predicted_orientation.squeeze(2), self.target.repeat(4))
            #self.loss += ce_loss_2
            unsup_loss = self.l1_critetion(self.predicted_orientation.squeeze(1), self.full_feat[:, self.chosen_idx, 2])
            self.loss += unsup_loss
            self.unsup_loss = unsup_loss.detach().item()

    def optimize_params(self):
        if self.method == 'supervised':
            self.supervised_optimizer.zero_grad()
            self.loss.backward()
            # utils.plot_grad_flow(self.named_parameters())
            self.supervised_optimizer.step()
        else:
            self.unsupervised_optimizer.zero_grad()
            self.loss.backward()
            # utils.plot_grad_flow(self.named_parameters())
            self.unsupervised_optimizer.step()
            #raise NotImplementedError('only supervised implemented')

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

    def get_loss(self):
        return self.loss.detach().item(), self.sup_loss.detach().item(), self.unsup_loss

    def get_image_output(self):
        sv = utils.SceneVisualiser()
        sv.visualise(self.image[0], self.objects[0], self.connectivity_argmaxed[0])

        return sv.scene

    def get_weight_norm(self):
        params = list(self.parameters())
        norm = [torch.norm(param.data) for param in params]
        return norm

    def get_grad_magnitude(self):
        params = list(self.parameters())
        norm = [torch.norm(param.grad) for param in params]
        return norm

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
