import numpy as np
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
        elif preds.size(1) == 15:   # no_distribution下经过Decoder（7+8）
            cls_preds, loc_preds, _ = preds.split([1, 6, 8], dim=1)
        elif preds.size(1) == 16:
            cls_preds, loc_preds, _ = preds.split([2, 6, 8], dim=1)

        # distribution
        elif preds.size(1) == 12:   
            cls_preds, loc_preds = preds.split([1, 11], dim=1)
        elif preds.size(1) == 13:
            cls_preds, loc_preds = preds.split([2, 11], dim=1)
        elif preds.size(1) == 20:   # distribution下经过Decoder（12+8）
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
                loss = loc_loss + cls_loss
            else:
                loc = 0.0
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
        
        #print(cls, loc)
        return loss, cls, loc, cls_loss


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        pass

    def forward(self, preds, targets):
        total_num = preds.shape[0]   # 暂时设定为1，具体为match_list中匹配到的总个数
        
        if total_num > 0:
            loss = F.smooth_l1_loss(preds, targets, reduction='sum') / total_num
        else:
            loss = 0.0
        
        return loss


class GaussianMixtureFunctions:
    eps = 1e-8

    def mean_squared_error(self, real, predictions):
        """Mean Squared Error between the true and predicted outputs
            loss = (1/n)*Σ(real - predicted)^2

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
            mse = (1/n)*Σ(real - predicted)^2
            bce = (1/n) * -Σ(real*log(predicted) + (1 - real)*log(1 - predicted))

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
            log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

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
            loss = (1/n) * -Σ targets*log(predicted)

        Args:
            logits: (array) corresponding array containing the logits of the categorical variable
            real: (array) corresponding array containing the true labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                    of all the sample losses or an array with the losses per sample
        """
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))