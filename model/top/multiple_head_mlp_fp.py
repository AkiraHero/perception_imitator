from model.model_base import ModelBase
import factory.model_factory as mf
import torch.nn as nn
import torch

'''
multiple heads for predict box error 
    heads including: x, y, z, w, h, l, rot
'''


class MultipleHeadMLPFp(ModelBase):
    def __init__(self, config):
        super(MultipleHeadMLPFp, self).__init__()
        self.backbone = mf.ModelFactory.get_model(config['paras']['backbone'])
        self.fp_predictor = mf.ModelFactory.get_model(config['paras']['fp_predictor'])
        self.all_num_predictor = mf.ModelFactory.get_model(config['paras']['all_num_predictor'])
        self.hard_num_predictor = mf.ModelFactory.get_model(config['paras']['hard_num_predictor'])

        # # box[x, y, z, w, h, l, rot, cls]
        # self.head_name = ['x', 'y', 'z', 'w', 'h', 'l', 'rot']
        # for name in self.head_name:
        #     self.add_module(name, mf.ModelFactory.get_model(config['paras']['box_err_predictor_heads']))
       
        # loss
        self.fp_loss_func = nn.BCELoss(reduction='none')
        self.all_num_loss_func = nn.CrossEntropyLoss(reduction='none')
        self.hard_num_loss_func = nn.CrossEntropyLoss(reduction='none')

        self.loss = 0

        # forward result
        self.data = None
        self.fp_prediction = None
        self.all_num_prediction = None
        self.hard_num_prediction = None

    def forward(self, data):
        input_ = data['gtbox_input']

        # input boxes into backbone
        x = self.backbone(input_)
        self.fp_prediction = self.fp_predictor(x)
        self.all_num_prediction = self.all_num_predictor(x)
        self.hard_num_prediction = self.hard_num_predictor(x)
        self.data = data
        return {
            "fp_prediction": self.fp_prediction,
            "all_num_prediction": self.all_num_prediction,
            "hard_num_prediction": self.hard_num_prediction
        }

    def get_loss(self):
        # loss for fp prediction: class 1: have, 0: dont have
        loss_dict = {}
        data = self.data
        target_fp = torch.eye(2)[data['have_fp'],:].cuda()
        batch_size = target_fp.shape[0]

        loss_dict['fp'] = self.fp_loss_func(self.fp_prediction, target_fp).sum() / batch_size

        detected_mask = data['have_fp']
        valid_batch_size = len((detected_mask == 1).nonzero())
        # loss for fp num prediction imitating the target model
        target_all_num = data['all_fp'].long()
        loss_dict['all_fp'] = self.all_num_loss_func(self.all_num_prediction, target_all_num).mul(
            detected_mask).sum() / valid_batch_size
        target_hard_num = data['hard_fp'].long()
        loss_dict['hard_fp'] = self.hard_num_loss_func(self.hard_num_prediction, target_hard_num).mul(
            detected_mask).sum() / valid_batch_size

        self.loss = sum(loss_dict.values())
        loss_dict['total'] = self.loss
        return loss_dict
