import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, config):
        super(CustomLoss, self).__init__()
        self.num_classes = config['num_classes']
        self.alpha = config['alpha']
        self.beta = config['beta']

    def focal_loss(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [BatchSize, Height, Width].
          y: (tensor) sized [BatchSize, Height, Width].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        # x = torch.sigmoid(x)
        x_t = x * (2 * y - 1) + (1 - y) # x_t = x     if label = 1
                                        # x_t = 1 -x  if label = 0

        alpha_t = torch.ones_like(x_t) * alpha
        alpha_t = alpha_t * (2 * y - 1) + (1 - y)

        loss = -alpha_t * (1-x_t)**gamma * x_t.log()

        return loss.mean()

    def cross_entropy(self, x, y):
        return F.binary_cross_entropy(input=x, target=y, reduction='mean')


    def forward(self, preds, targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          preds: (tensor)  cls_preds + reg_preds, sized[batch_size, 7, height, width]
          cls_preds: (tensor) predicted class confidences, sized [batch_size, 1, height, width].
          cls_targets: (tensor) encoded target labels, sized [batch_size, 1, height, width].
          loc_preds: (tensor) predicted target locations, sized [batch_size, 6, height, width].
          loc_targets: (tensor) encoded target locations, sized [batch_size, 6, height, width].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''

        batch_size = targets.size(0)
        image_size = targets.size(2) * targets.size(3)
        if targets.size(1) == 7:  # no_distribution
            cls_targets, loc_targets = targets.split([1, 6], dim=1)
        elif targets.size(1) == 12:  # distribution
            cls_targets, loc_targets = targets.split([1, 11], dim=1)

        if preds.size(1) == 7:  # no_distribution
            cls_preds, loc_preds = preds.split([1, 6], dim=1)
        elif preds.size(1) == 15:   # no_distribution下经过Decoder（7+8）
            cls_preds, loc_preds, _ = preds.split([1, 6, 8], dim=1)
        elif preds.size(1) == 12:   # distribution
            cls_preds, loc_preds = preds.split([1, 11], dim=1)
        elif preds.size(1) == 20:   # distribution下经过Decoder（12+8）
            cls_preds, loc_preds, _ = preds.split([1, 11, 8], dim=1)
        ################################################################
        # cls_preds = torch.clamp(cls_preds, min=1e-5, max=1-1e-5)
        # cls_loss = self.focal_loss(cls_preds, cls_targets) * self.alpha
        ################################################################
        cls_loss = self.cross_entropy(cls_preds, cls_targets) * self.alpha
        cls = cls_loss.item()
        ################################################################
        # reg_loss = SmoothL1Loss(loc_preds, loc_targets)
        ################################################################
        
        pos_pixels = cls_targets.sum()
        if pos_pixels > 0:
            loc_loss = F.smooth_l1_loss(cls_targets * loc_preds, loc_targets, reduction='sum') / pos_pixels * self.beta
            loc = loc_loss.item()
            loss = loc_loss + cls_loss
        else:
            loc = 0.0
            loss = cls_loss
        
        #print(cls, loc)
        return loss, cls, loc
