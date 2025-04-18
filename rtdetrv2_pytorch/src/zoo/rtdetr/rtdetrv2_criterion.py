"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.distributed
import torch.nn.functional as F 
import torchvision

import copy

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ...core import register


@register()
class RTDETRCriterionv2(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher, 
        weight_dict, 
        losses, 
        alpha=0.2, 
        gamma=2.0, 
        num_classes=80, 
        boxes_weight_format=None,
        share_matched_indices=False):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            boxes_weight_format: format for boxes weight (iou, )
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.margin = 0.05  # Reduced margin to allow slight overlap
        self.x_scale = 0.5  # Scaling factor for x-alignment term

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
    
        # Define class weights
        class_weight = torch.ones(self.num_classes, device=src_logits.device)
        class_weight[1] = 2.0  # Higher weight for left arytenoid cartilage
        class_weight[5] = 2.0  # Higher weight for right arytenoid cartilage
    
        # Compute focal loss and apply class weights
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = (loss * class_weight[None, None, :]).mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()

        # Define class weights
        class_weight = torch.ones(self.num_classes, device=src_logits.device)
        class_weight[1] = 2.0  # Higher weight for left arytenoid cartilage
        class_weight[5] = 2.0  # Higher weight for right arytenoid cartilage
    
        # Adjust weight with class weights
        weight = (self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score) * class_weight[None, None, :]
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    # def loss_relational(self, outputs, targets, indices, num_boxes):
    #     """Compute a relational loss to enforce spatial relationships between vocal folds and arytenoid cartilage."""
    #     loss = 0.0
    #     device = outputs['pred_boxes'].device  # Get the device from model outputs
    #     for batch_idx, (src_indices, tgt_indices) in enumerate(indices):
    #         gt_labels = targets[batch_idx]['labels'].to(device)  # Move ground truth labels to the correct device
    #         # Left side: vocal fold (0) and arytenoid cartilage (1)
    #         left_vf = (gt_labels == 0).nonzero(as_tuple=True)[0]
    #         left_ac = (gt_labels == 1).nonzero(as_tuple=True)[0]
    #         if len(left_vf) > 0 and len(left_ac) > 0:
    #             s = left_vf[0]  # Ground truth index for left vocal fold
    #             t = left_ac[0]  # Ground truth index for left arytenoid cartilage
    #             # Move indices to the correct device
    #             src_indices = src_indices.to(device)
    #             tgt_indices = tgt_indices.to(device)
    #             s = s.to(device)  # Ensure s is on the correct device
    #             t = t.to(device)  # Ensure t is on the correct device
    #             i = src_indices[tgt_indices == s]  # Prediction index for vocal fold
    #             j = src_indices[tgt_indices == t]  # Prediction index for arytenoid cartilage
    #             if len(i) > 0 and len(j) > 0:
    #                 i = i[0]
    #                 j = j[0]
    #                 pred_box_i = outputs['pred_boxes'][batch_idx, i]  # Vocal fold box
    #                 pred_box_j = outputs['pred_boxes'][batch_idx, j]  # Arytenoid cartilage box
    #                 cx_i, cy_i, w_i, h_i = pred_box_i
    #                 cx_j, cy_j, w_j, h_j = pred_box_j
    #                 # Compute box edges
    #                 y_i_bottom = cy_i + h_i / 2
    #                 y_j_top = cy_j - h_j / 2
    #                 # Encourage x-alignment and y-position (arytenoid cartilage below vocal fold)
    #                 x_loss = self.x_scale * torch.abs(cx_j - cx_i)
    #                 y_loss = F.relu(y_j_top - y_i_bottom + self.margin)
    #                 loss += x_loss + y_loss
    # 
    #         # Right side: vocal fold (4) and arytenoid cartilage (5)
    #         right_vf = (gt_labels == 4).nonzero(as_tuple=True)[0]
    #         right_ac = (gt_labels == 5).nonzero(as_tuple=True)[0]
    #         if len(right_vf) > 0 and len(right_ac) > 0:
    #             s = right_vf[0]  # Ground truth index for right vocal fold
    #             t = right_ac[0]  # Ground truth index for right arytenoid cartilage
    #             # Move indices to the correct device
    #             src_indices = src_indices.to(device)
    #             tgt_indices = tgt_indices.to(device)
    #             s = s.to(device)  # Ensure s is on the correct device
    #             t = t.to(device)  # Ensure t is on the correct device
    #             i = src_indices[tgt_indices == s]  # Prediction index for vocal fold
    #             j = src_indices[tgt_indices == t]  # Prediction index for arytenoid cartilage
    #             if len(i) > 0 and len(j) > 0:
    #                 i = i[0]
    #                 j = j[0]
    #                 pred_box_i = outputs['pred_boxes'][batch_idx, i]
    #                 pred_box_j = outputs['pred_boxes'][batch_idx, j]
    #                 cx_i, cy_i, w_i, h_i = pred_box_i
    #                 cx_j, cy_j, w_j, h_j = pred_box_j
    #                 # Compute box edges
    #                 y_i_bottom = cy_i + h_i / 2
    #                 y_j_top = cy_j - h_j / 2
    #                 # Encourage x-alignment and y-position (arytenoid cartilage below vocal fold)
    #                 x_loss = self.x_scale * torch.abs(cx_j - cx_i)
    #                 y_loss = F.relu(y_j_top - y_i_bottom + self.margin)
    #                 loss += x_loss + y_loss
    # 
    #     return {'loss_relational': loss}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            # 'relational': self.loss_relational,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Retrieve the matching between the outputs of the last layer and the targets
        matched = self.matcher(outputs_without_aux, targets)
        indices = matched['indices']

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            meta = self.get_loss_meta_info(loss, outputs, targets, indices)            
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if not self.share_matched_indices:
                    matched = self.matcher(aux_outputs, targets)
                    indices = matched['indices']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of encoder auxiliary losses. For rtdetr v2
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                matched = self.matcher(aux_outputs, targets)
                indices = matched['indices']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
            
            if class_agnostic:
                self.num_classes = orig_num_classes

        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', ):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices
