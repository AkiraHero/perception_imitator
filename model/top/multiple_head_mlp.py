from model.model_base import ModelBase
import factory.model_factory as mf
import torch.nn as nn
import torch
'''
multiple heads for predict box error 
    heads including: x, y, z, w, h, l, rot
'''


class MultipleHeadMLP(ModelBase):
    def __init__(self, config):
        super(MultipleHeadMLP, self).__init__()
        self.backbone = mf.ModelFactory.get_model(config['paras']['backbone'])
        self.fn_predictor = mf.ModelFactory.get_model(config['paras']['fn_predictor'])
        self.cls_predictor = mf.ModelFactory.get_model(config['paras']['cls_predictor'])

        # box[x, y, z, w, h, l, rot, cls]
        self.head_name = ['fn', 'x', 'y', 'z', 'w', 'h', 'l', 'rot']
        for name in self.head_name:
            self.add_module(name, mf.ModelFactory.get_model(config['paras']['box_err_predictor_heads']))

        # loss
        self.fn_loss_func = nn.BCELoss(reduction='none')
        # calculated from the whole dataset
        self.fn_loss_weight = 0.81916
        self.tp_loss_weight = 0.18084
        # no weight for heads for their class have uniform distribution
        self.head_loss_func = nn.CrossEntropyLoss(reduction='none')
        self.cls_loss_func = nn.CrossEntropyLoss(reduction='none')

        self.loss = 0

        # forward result
        self.data = None
        self.box_err_prediction = None
        self.fn_prediction = None
        self.cls_prediction = None

    def forward(self, data):
        input_ = data['gt_box']

        # input boxes into backbone
        x = self.backbone(input_)
        self.fn_prediction = self.fn_predictor(x)
        self.box_err_prediction = {i: self._modules[i](x) for i in self.head_name}
        self.cls_prediction = self.cls_predictor(x)
        self.data = data

    def get_loss(self):
        # loss for fn prediction: class 1: detected, 0: non-detected
        data = self.data
        target_fn = data['detected']
        batch_size = target_fn.shape[0]
        loss_weight = torch.ones(target_fn.shape, device=target_fn.device)
        detected_inx = (target_fn == 1).nonzero()
        non_detected_inx = (target_fn == 0).nonzero()
        loss_weight[detected_inx] = self.tp_loss_weight
        loss_weight[non_detected_inx] = self.fn_loss_weight
        loss_fp = self.fn_loss_func(self.fn_prediction, target_fn).mul(loss_weight).sum() / batch_size

        # loss for error prediction of box location / dimension
        target_x = data['box_diff_cls'][:, 0, :].long()
        target_y = data['box_diff_cls'][:, 1, :].long()
        target_z = data['box_diff_cls'][:, 2, :].long()
        target_w = data['box_diff_cls'][:, 3, :].long()
        target_h = data['box_diff_cls'][:, 4, :].long()
        target_l = data['box_diff_cls'][:, 5, :].long()
        target_rot = data['box_diff_cls'][:, 6, :].long()

        # loss for class prediction imitating the target model
        target_cls = data['dt_box'][:, 7].long()
        detected_mask = data['detected'].squeeze(-1)
        target_cls[(detected_mask==1).nonzero()] -= 1

        loss_x = self.head_loss_func(self.box_err_prediction['x'], target_x.reshape(-1, )).mul(detected_mask).sum() / batch_size
        loss_y = self.head_loss_func(self.box_err_prediction['y'], target_y.reshape(-1, )).mul(detected_mask).sum() / batch_size
        loss_z = self.head_loss_func(self.box_err_prediction['z'], target_z.reshape(-1, )).mul(detected_mask).sum() / batch_size
        loss_w = self.head_loss_func(self.box_err_prediction['w'], target_w.reshape(-1, )).mul(detected_mask).sum() / batch_size
        loss_h = self.head_loss_func(self.box_err_prediction['h'], target_h.reshape(-1, )).mul(detected_mask).sum() / batch_size
        loss_l = self.head_loss_func(self.box_err_prediction['l'], target_l.reshape(-1, )).mul(detected_mask).sum() / batch_size
        loss_rot = self.head_loss_func(self.box_err_prediction['rot'], target_rot.reshape(-1, )).mul(detected_mask).sum() / batch_size
        loss_cls = self.cls_loss_func(self.cls_prediction, target_cls).mul(detected_mask).sum() / batch_size
        self.loss = loss_fp + loss_x + loss_y + loss_z + loss_w + loss_h + loss_l + loss_rot + loss_cls
        return {
            'loss': self.loss,
            'loss_fp': loss_fp,
            'loss_x': loss_x,
            'loss_y': loss_y,
            'loss_z': loss_z,
            'loss_w': loss_w,
            'loss_h': loss_h,
            'loss_l': loss_l,
            'loss_rot': loss_rot,
            'loss_cls': loss_cls,
        }
