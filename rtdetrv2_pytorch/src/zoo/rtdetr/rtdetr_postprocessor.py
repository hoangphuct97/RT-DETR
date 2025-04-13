"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision

from ...core import register


__all__ = ['RTDETRPostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class RTDETRPostProcessor(nn.Module):
    __share__ = [
        'num_classes', 
        'use_focal_loss', 
        'num_top_queries', 
        'remap_mscoco_category'
    ]
    
    def __init__(
        self, 
        num_classes=80, 
        use_focal_loss=True, 
        num_top_queries=300, 
        remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False

        self.LEFT_VOCAL_FOLD = 1
        self.LEFT_ARYTENOID_CARTILAGE = 2
        self.RIGHT_VOCAL_FOLD = 5
        self.RIGHT_ARYTENOID_CARTILAGE = 6

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    def anatomical_postprocessing(self, results):
        """
        Post-process RT-DETR predictions to enhance arytenoid cartilage detection
        based on vocal fold detections.
        
        Args:
            results: List of dictionaries containing detection results for each image
                    Each dict contains 'labels', 'boxes', and 'scores' tensors
        
        Returns:
            Enhanced detection results with added arytenoid cartilages
        """
        # Confidence thresholds
        VOCAL_FOLD_THRESHOLD = 0.5
        EXISTING_ARYTENOID_THRESHOLD = 0.3  # Lower threshold to check for existing detections
        ADDED_ARYTENOID_SCORE = 0.4  # Confidence score for added arytenoid predictions

        enhanced_results = []

        # Process each image's predictions
        for result in results:
            labels = result['labels']
            boxes = result['boxes']
            scores = result['scores']

            # Find vocal fold detections with high confidence
            left_vocal_fold_mask = (labels == self.LEFT_VOCAL_FOLD) & (scores > VOCAL_FOLD_THRESHOLD)
            right_vocal_fold_mask = (labels == self.RIGHT_VOCAL_FOLD) & (scores > VOCAL_FOLD_THRESHOLD)

            # Find existing arytenoid detections
            left_arytenoid_mask = (labels == self.LEFT_ARYTENOID) & (scores > EXISTING_ARYTENOID_THRESHOLD)
            right_arytenoid_mask = (labels == self.RIGHT_ARYTENOID) & (scores > EXISTING_ARYTENOID_THRESHOLD)

            # Extract relevant detections
            left_vocal_fold_boxes = boxes[left_vocal_fold_mask] if torch.any(left_vocal_fold_mask) else torch.tensor([], device=boxes.device)
            left_arytenoid_boxes = boxes[left_arytenoid_mask] if torch.any(left_arytenoid_mask) else torch.tensor([], device=boxes.device)

            right_vocal_fold_boxes = boxes[right_vocal_fold_mask] if torch.any(right_vocal_fold_mask) else torch.tensor([], device=boxes.device)
            right_arytenoid_boxes = boxes[right_arytenoid_mask] if torch.any(right_arytenoid_mask) else torch.tensor([], device=boxes.device)

            # Lists to store new detections
            new_boxes = []
            new_labels = []
            new_scores = []

            # For each left vocal fold, check if we need to add left arytenoid
            for vocal_box in left_vocal_fold_boxes:
                x1, y1, x2, y2 = vocal_box.tolist()

                # Check if there's already an arytenoid detection below this vocal fold
                has_arytenoid = False

                for ary_box in left_arytenoid_boxes:
                    a_x1, a_y1, a_x2, a_y2 = ary_box.tolist()

                    # Check if arytenoid is below vocal fold and has horizontal overlap
                    if (a_y1 > y2) and (max(0, min(a_x2, x2) - max(a_x1, x1)) > 0):
                        has_arytenoid = True
                        break

                # If no corresponding arytenoid, add one based on anatomical knowledge
                if not has_arytenoid:
                    width = x2 - x1
                    height = y2 - y1

                    # Predict where arytenoid should be based on the vocal fold position
                    ary_x1 = x1 + width * 0.1  # Slight adjustment based on anatomy
                    ary_x2 = x2 - width * 0.1
                    ary_y1 = y2 + 5  # Just below the vocal fold
                    ary_y2 = y2 + height * 0.8  # Estimated height based on vocal fold

                    # Add new detection
                    new_boxes.append([ary_x1, ary_y1, ary_x2, ary_y2])
                    new_labels.append(self.LEFT_ARYTENOID)
                    new_scores.append(ADDED_ARYTENOID_SCORE)

            # Handle right vocal fold and right arytenoid similarly
            for vocal_box in right_vocal_fold_boxes:
                x1, y1, x2, y2 = vocal_box.tolist()

                # Check if there's already an arytenoid detection below this vocal fold
                has_arytenoid = False

                for ary_box in right_arytenoid_boxes:
                    a_x1, a_y1, a_x2, a_y2 = ary_box.tolist()

                    # Check if arytenoid is below vocal fold and has horizontal overlap
                    if (a_y1 > y2) and (max(0, min(a_x2, x2) - max(a_x1, x1)) > 0):
                        has_arytenoid = True
                        break

                # If no corresponding arytenoid, add one based on anatomical knowledge
                if not has_arytenoid:
                    width = x2 - x1
                    height = y2 - y1

                    # Predict where arytenoid should be based on the vocal fold position
                    ary_x1 = x1 + width * 0.1  # Slight adjustment based on anatomy
                    ary_x2 = x2 - width * 0.1
                    ary_y1 = y2 + 5  # Just below the vocal fold
                    ary_y2 = y2 + height * 0.8  # Estimated height based on vocal fold

                    # Add new detection
                    new_boxes.append([ary_x1, ary_y1, ary_x2, ary_y2])
                    new_labels.append(self.RIGHT_ARYTENOID)
                    new_scores.append(ADDED_ARYTENOID_SCORE)

            # Append new detections if any were added
            if new_boxes:
                # Convert to tensors with same device as original tensors
                new_boxes_tensor = torch.tensor(new_boxes, device=boxes.device)
                new_labels_tensor = torch.tensor(new_labels, device=labels.device)
                new_scores_tensor = torch.tensor(new_scores, device=scores.device)

                # Concatenate with existing detections
                result['boxes'] = torch.cat([boxes, new_boxes_tensor], dim=0)
                result['labels'] = torch.cat([labels, new_labels_tensor], dim=0)
                result['scores'] = torch.cat([scores, new_scores_tensor], dim=0)

            enhanced_results.append(result)

        return enhanced_results
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))
        
        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        # Apply our anatomical post-processing
        enhanced_results = self.anatomical_postprocessing(results)
        return enhanced_results
        

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 
