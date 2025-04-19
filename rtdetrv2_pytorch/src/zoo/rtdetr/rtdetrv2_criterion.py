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

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

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
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
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

    def loss_spatial(self, outputs, targets, indices, num_boxes):
        """Custom loss enforcing spatial relationships between vocal folds and arytenoid cartilages"""
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        pred_probs = torch.sigmoid(pred_logits)
        batch_size, num_queries = pred_logits.shape[:2]

        # Class indices
        left_vocal_fold_idx = 0  # Adjust to your actual class index
        right_vocal_fold_idx = 4  # Adjust to your actual class index
        left_arytenoid_idx = 1  # Adjust to your actual class index
        right_arytenoid_idx = 5  # Adjust to your actual class index

        # Mapping between vocal folds and expected cartilages
        vocal_to_cartilage = {
            left_vocal_fold_idx: left_arytenoid_idx,
            right_vocal_fold_idx: right_arytenoid_idx
        }

        # Updated parameters based on the image for expected cartilage position relative to vocal fold
        position_params = {
            left_vocal_fold_idx: {
                'cx_offset': 0.15,  # Slightly medial (toward center)
                'cy_offset': 0.9,  # Below vocal fold (90% of height)
                'width_ratio': 0.5,  # 50% of vocal fold width
                'height_ratio': 0.5  # 50% of vocal fold height
            },
            right_vocal_fold_idx: {
                'cx_offset': -0.15,  # Slightly medial (toward center) - negative for right side
                'cy_offset': 0.9,  # Below vocal fold (90% of height)
                'width_ratio': 0.5,  # 50% of vocal fold width
                'height_ratio': 0.5  # 50% of vocal fold height
            }
        }

        # Initialize loss as a tensor with value 0
        total_loss = torch.tensor(0.0, device=pred_logits.device)
        num_relationships = 0

        for batch_idx in range(batch_size):
            # Get matched predictions and targets
            src_idx, tgt_idx = indices[batch_idx]
            if len(src_idx) == 0:
                continue

            # Process each vocal fold in predictions
            for i in range(num_queries):
                vocal_class_probs = pred_probs[batch_idx, i, [left_vocal_fold_idx, right_vocal_fold_idx]]
                max_vocal_prob, max_vocal_idx = torch.max(vocal_class_probs, dim=0)
                vocal_class = [left_vocal_fold_idx, right_vocal_fold_idx][max_vocal_idx]

                # If the prediction is confidently a vocal fold (confidence > threshold)
                if max_vocal_prob > 0.5:  # Confidence threshold
                    num_relationships += 1

                    # Get corresponding vocal fold box
                    vf_box = pred_boxes[batch_idx, i]
                    cx, cy, w, h = vf_box.unbind(-1)

                    # Calculate expected cartilage position
                    params = position_params[vocal_class]
                    exp_cart_cx = cx + params['cx_offset'] * w
                    exp_cart_cy = cy + params['cy_offset'] * h
                    exp_cart_w = w * params['width_ratio']
                    exp_cart_h = h * params['height_ratio']

                    expected_cart_box = torch.stack([exp_cart_cx, exp_cart_cy, exp_cart_w, exp_cart_h], dim=-1)

                    # Get corresponding cartilage class
                    cartilage_class = vocal_to_cartilage[vocal_class]

                    # Find best matching cartilage prediction
                    cart_probs = pred_probs[batch_idx, :, cartilage_class]
                    cart_boxes = pred_boxes[batch_idx]

                    # Calculate IoU between expected cartilage box and all prediction boxes
                    # Convert center-size format to corner format for IoU calculation
                    expected_xmin = exp_cart_cx - 0.5 * exp_cart_w
                    expected_ymin = exp_cart_cy - 0.5 * exp_cart_h
                    expected_xmax = exp_cart_cx + 0.5 * exp_cart_w
                    expected_ymax = exp_cart_cy + 0.5 * exp_cart_h
                    expected_corners = torch.stack([expected_xmin, expected_ymin, expected_xmax, expected_ymax], dim=-1)

                    # Convert all prediction boxes to corner format
                    pred_xmin = cart_boxes[:, 0] - 0.5 * cart_boxes[:, 2]
                    pred_ymin = cart_boxes[:, 1] - 0.5 * cart_boxes[:, 3]
                    pred_xmax = cart_boxes[:, 0] + 0.5 * cart_boxes[:, 2]
                    pred_ymax = cart_boxes[:, 1] + 0.5 * cart_boxes[:, 3]
                    pred_corners = torch.stack([pred_xmin, pred_ymin, pred_xmax, pred_ymax], dim=-1)

                    # Calculate IoU
                    ious = box_iou(expected_corners.unsqueeze(0), pred_corners)[0]

                    # Combined score: cartilage class probability * IoU
                    combined_scores = cart_probs * ious

                    # Apply soft-max to get attention weights
                    weights = torch.softmax(combined_scores * 10, dim=0)  # Temperature = 10

                    # Calculate the expected cartilage score
                    expected_cart_score = torch.sum(weights * combined_scores)

                    # Calculate loss: encourage high cartilage scores at expected positions
                    cart_loss = 1.0 - expected_cart_score
                    total_loss += cart_loss

                    # Also add loss to push cartilage predictions toward expected positions
                    for j in range(num_queries):
                        # If this prediction has high probability of being the corresponding cartilage
                        if cart_probs[j] > 0.3:
                            # Get current box
                            curr_box = cart_boxes[j]
                            # Calculate L1 loss between current and expected position
                            box_loss = F.l1_loss(curr_box, expected_cart_box)
                            # Weight by cartilage probability
                            total_loss += box_loss * cart_probs[j] * 0.5  # 0.5 is the weight for this term

                    # Add a term to enforce the presence of a cartilage when a vocal fold is detected
                    if torch.max(cart_probs) < 0.3:
                        # If no cartilage is detected with sufficient confidence
                        presence_penalty = torch.tensor(0.8, device=pred_logits.device)  # Substantial penalty as tensor
                        total_loss += presence_penalty * max_vocal_prob  # Scale by vocal fold confidence

        # Normalize loss - ensure it's a tensor
        if num_relationships > 0:
            spatial_loss = total_loss / num_relationships
        else:
            spatial_loss = torch.tensor(0.0, device=pred_logits.device)

        return {'loss_spatial': spatial_loss}

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
            'spatial': self.loss_spatial,
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
