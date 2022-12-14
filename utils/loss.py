import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from IoU_calculate import *


class CustomLoss(nn.Module):
    def __init__(self, config):
        super(CustomLoss, self).__init__()
        self.num_classes = config['num_classes']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.use_iou = config['iou']
        if self.use_iou:
            self.gamma = config['gamma']
        self.geometry = [-40, 40 , 0.0, 70.4]
        self.ratio = 0.2

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

    def smooth_l1_loss(self, x, beta = 1.0):
        n = torch.abs(x)
        loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5*beta)
        return loss.mean()
    
    def IOU_loss(self, IOU, gamma=1.0):
        return - torch.log(IOU).sum() * gamma
    
    def decode_corner(self,x):
        device = torch.device('cpu')
        if x.is_cuda:
            device = x.get_device()
        cos_t, sin_t, dx, dy, log_w, log_l = torch.chunk(x, 6, dim=1)
        # theta = torch.atan2(sin_t, cos_t)
        # cos_t = torch.cos(theta)
        # sin_t = torch.sin(theta)

        x = torch.arange(self.geometry[3], self.geometry[2], -self.ratio, dtype=torch.float32, device=device)
        y = torch.arange(self.geometry[1], self.geometry[0], -self.ratio, dtype=torch.float32, device=device)
        xx, yy = torch.meshgrid([x, y])
        centre_y = yy + dy
        centre_x = xx + dx
        l = log_l.exp()
        w = log_w.exp()
        rear_left_x = centre_x - l/2 * cos_t - w/2 * sin_t
        rear_left_y = centre_y - l/2 * sin_t + w/2 * cos_t
        rear_right_x = centre_x - l/2 * cos_t + w/2 * sin_t
        rear_right_y = centre_y - l/2 * sin_t - w/2 * cos_t
        front_right_x = centre_x + l/2 * cos_t + w/2 * sin_t
        front_right_y = centre_y + l/2 * sin_t - w/2 * cos_t
        front_left_x = centre_x + l/2 * cos_t - w/2 * sin_t
        front_left_y = centre_y + l/2 * sin_t + w/2 * cos_t

        corners = torch.cat([rear_left_x, rear_left_y, rear_right_x, rear_right_y,
                                front_right_x, front_right_y, front_left_x, front_left_y], dim=1)

        return corners
    
    def decode_reg(self,x):
        device = torch.device('cpu')
        if x.is_cuda:
            device = x.get_device()
        cos_t, sin_t, dx, dy, log_w, log_l = torch.chunk(x, 6, dim=1)
        # theta = torch.atan2(sin_t, cos_t)
        # cos_t = torch.cos(theta)
        # sin_t = torch.sin(theta)

        x = torch.arange(self.geometry[3], self.geometry[2], -self.ratio, dtype=torch.float32, device=device)
        y = torch.arange(self.geometry[1], self.geometry[0], -self.ratio, dtype=torch.float32, device=device)
        xx, yy = torch.meshgrid([x, y])
        centre_y = yy + dy
        centre_x = xx + dx
        l = log_l.exp()
        w = log_w.exp()
        

        reg = torch.cat([centre_x, centre_y, l, w, cos_t, sin_t], dim=1)

        tensor_list = torch.split(reg, 1, dim=3)
        reg = torch.cat([t for t in tensor_list],dim = 0)
        tensor_list = torch.split(reg, 1, dim=2)
        reg = torch.cat([t for t in tensor_list],dim = 0)
        
        
        print(reg.size())
        return reg
            

    def forward(self, preds, targets, attention_mask=None):
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
        elif targets.size(1) == 8: 
            cls_targets, loc_targets = targets.split([2, 6], dim=1)
        elif targets.size(1) == 12:  # distribution
            cls_targets, loc_targets = targets.split([1, 11], dim=1)
        elif targets.size(1) == 13:
            cls_targets, loc_targets = targets.split([2, 11], dim=1)
        # no_distribution
        if preds.size(1) == 7:
            cls_preds, loc_preds = preds.split([1, 6], dim=1)
        if preds.size(1) == 8:
            cls_preds, loc_preds = preds.split([2, 6], dim=1)
        elif preds.size(1) == 15:   # no_distribution?????????Decoder???7+8???
            cls_preds, loc_preds, corner_preds = preds.split([1, 6, 8], dim=1)
        elif preds.size(1) == 16:
            cls_preds, loc_preds, _ = preds.split([2, 6, 8], dim=1)

        # distribution
        elif preds.size(1) == 12:   
            cls_preds, loc_preds = preds.split([1, 11], dim=1)
        elif preds.size(1) == 13:
            cls_preds, loc_preds = preds.split([2, 11], dim=1)
        elif preds.size(1) == 20:   # distribution?????????Decoder???12+8???
            cls_preds, loc_preds, _ = preds.split([1, 11, 8], dim=1)
        elif preds.size(1) == 21:
            cls_preds, loc_preds, _ = preds.split([2, 11, 8], dim=1)
        
        if cls_targets.size(1) == 1:
        ################################################################
        # cls_preds = torch.clamp(cls_preds, min=1e-5, max=1-1e-5)
        # cls_loss = self.focal_loss(cls_preds, cls_targets) * self.alpha
        ################################################################
            cls_loss = self.cross_entropy(cls_preds, cls_targets) * self.alpha
            # cls_loss = (0.9 * self.cross_entropy(attention_mask * cls_preds, cls_targets) + \
            #             0.1 * self.cross_entropy(cls_preds, cls_targets)) * \
            #             self.alpha
            cls = cls_loss.item()
            ################################################################
            # reg_loss = SmoothL1Loss(loc_preds, loc_targets)
            ################################################################
            
            pos_pixels = cls_targets.sum()
            if pos_pixels > 0:
                loc_loss = F.smooth_l1_loss(cls_targets * loc_preds, loc_targets, reduction='sum') / pos_pixels * self.beta
                loc = loc_loss.item()
                
                #################IoU loss####################
                if self.use_iou:
                    # mask = cls_targets > 0
                    # masked_loc_targets = torch.masked_select(loc_targets, mask)
                    # masked_loc_preds = torch.masked_select(loc_preds, mask)
                    # reg_targets = self.decode_reg(masked_loc_targets)
                    # reg_preds = self.decode_reg(masked_loc_preds)
                    # IOU = boxes_iou_cpu(reg_preds,reg_targets)
                    # corner_loss = self.IOU_loss(IOU)
                    corner_preds = self.decode_corner(loc_preds)
                    corner_targets = self.decode_corner(loc_targets)
                    
                    corner_dist = torch.norm(cls_targets *corner_preds - cls_targets*corner_targets, dim=1)
                    corner_loss = self.smooth_l1_loss(corner_dist) / self.gamma

                    corner = corner_loss.item()
                    loss = loc_loss + cls_loss + corner_loss
                else:
                    corner = 0.0
                    loss = loc_loss + cls_loss
            else:
                loc = 0.0
                corner = 0.0
                loss = cls_loss
        elif cls_targets.size(1) == 2:
            cls_loss = (self.cross_entropy(cls_preds[:,0,...], cls_targets[:,0,...]) + self.cross_entropy(cls_preds[:,1,...], cls_targets[:,1,...])) * self.alpha
            cls = cls_loss.item()
            
            cls_all_targets = torch.clamp(cls_targets[:,0,...] + cls_targets[:,1,...], 0.0, 1.0).unsqueeze(1)
            pos_pixels = cls_all_targets.sum()
            if pos_pixels > 0:
                loc_loss = F.smooth_l1_loss(cls_all_targets * loc_preds, loc_targets, reduction='sum') / pos_pixels * self.beta
                loc = loc_loss.item()
                loss = loc_loss + cls_loss
            else:
                loc = 0.0
                loss = cls_loss
        
        return loss, cls, loc, corner, cls_loss


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        pass

    def forward(self, preds, targets):
        total_num = preds.shape[0]   # ???????????????1????????????match_list????????????????????????
        
        if total_num > 0:
            loss = F.smooth_l1_loss(preds, targets, reduction='sum') / total_num
        else:
            loss = 0.0
        
        return loss


class GaussianMixtureFunctions:
    eps = 1e-8

    def mean_squared_error(self, real, predictions):
        """Mean Squared Error between the true and predicted outputs
            loss = (1/n)*??(real - predicted)^2

        Args:
            real: (array) corresponding array containing the true labels
            predictions: (array) corresponding array containing the predicted labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                    of all the sample losses or an array with the losses per sample
        """
        loss = (real - predictions).pow(2)
        return loss.sum(-1).mean()


    def reconstruction_loss(self, real, predicted, rec_type='mse' ):
        """Reconstruction loss between the true and predicted outputs
            mse = (1/n)*??(real - predicted)^2
            bce = (1/n) * -??(real*log(predicted) + (1 - real)*log(1 - predicted))

        Args:
            real: (array) corresponding array containing the true labels
            predictions: (array) corresponding array containing the predicted labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                    of all the sample losses or an array with the losses per sample
        """
        bs = real.shape[0]

        if rec_type == 'mse':
            loss = (real - predicted).pow(2)
        elif rec_type == 'bce':
            # loss = F.binary_cross_entropy(predicted, real, reduction='none')
            loss = F.smooth_l1_loss(predicted, real, reduction='sum') / bs  # use smmothl1 to calculate the reconstruction loss between gt fp bbox and gen fp bbox
        else:
            raise "invalid loss function... try bce or mse..."
        #   return loss.sum(-1).mean()
        return loss

    def log_normal(self, x, mu, var):
        """Logarithm of normal distribution with mean=mu and variance=var
            log(x|??, ??^2) = loss = -0.5 * ?? log(2??) + log(??^2) + ((x - ??)/??)^2

        Args:
            x: (array) corresponding array containing the input
            mu: (array) corresponding array containing the mean 
            var: (array) corresponding array containing the variance

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                    of all the sample losses or an array with the losses per sample
        """
        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.sum(
            np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)


    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        """Variational loss when using labeled data without considering reconstruction loss
            loss = log q(z|x,y) - log p(z) - log p(y)

        Args:
            z: (array) array containing the gaussian latent variable
            z_mu: (array) array containing the mean of the inference model
            z_var: (array) array containing the variance of the inference model
            z_mu_prior: (array) array containing the prior mean of the generative model
            z_var_prior: (array) array containing the prior variance of the generative mode
            
        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                    of all the sample losses or an array with the losses per sample
        """
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.mean()


    def entropy(self, logits, targets):
        """Entropy loss
            loss = (1/n) * -?? targets*log(predicted)

        Args:
            logits: (array) corresponding array containing the logits of the categorical variable
            real: (array) corresponding array containing the true labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                    of all the sample losses or an array with the losses per sample
        """
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))