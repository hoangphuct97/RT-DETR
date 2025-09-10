"""Enhanced visualization for detection model with CocoDetection ground truth support
Enhanced with random batch processing from directory
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import CocoDetection
from matplotlib.colors import to_rgba
import os
import json
import random
import glob
from pathlib import Path

from rtdetrv2_pytorch.references.deploy.export import export_results
from rtdetrv2_pytorch.src.core import YAMLConfig

# from src.core import YAMLConfig

# Define color palette for 6 categories - distinct colors with good visibility
CATEGORY_COLORS = {
    1: '#FF3838',  # Red
    2: '#18A558',  # Green
    3: '#4361EE',  # Blue
    4: '#FFB300',  # Amber
    5: '#9C27B0',  # Purple
    6: '#00BCD4',  # Cyan
}

# Map category indices to names (replace with your actual category names)
CATEGORY_NAMES = {
    1: 'L_Vocal Fold',
    2: 'L_Arytenoid cartilage',
    3: 'Benign lesion',
    4: 'Malignant lesion',
    5: 'R_Vocal Fold',
    6: 'R_Arytenoid cartilage',
}

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def get_image_files(directory):
    """
    Get all image files from a directory recursively
    
    Args:
        directory: Path to the directory containing images
        
    Returns:
        List of image file paths
    """
    image_files = []
    directory = Path(directory)

    for ext in IMAGE_EXTENSIONS:
        # Search recursively for images with each extension
        image_files.extend(glob.glob(str(directory / f"**/*{ext}"), recursive=True))
        image_files.extend(glob.glob(str(directory / f"**/*{ext.upper()}"), recursive=True))

    return image_files


def select_random_images(directory, num_images=2):
    """
    Randomly select a specified number of images from a directory
    
    Args:
        directory: Path to the directory containing images
        num_images: Number of images to select
        
    Returns:
        List of randomly selected image file paths
    """
    all_images = get_image_files(directory)

    if not all_images:
        raise ValueError(f"No image files found in directory: {directory}")

    print(f"Found {len(all_images)} images in directory: {directory}")

    # Select random images (without replacement if possible)
    if len(all_images) <= num_images:
        print(f"Directory contains {len(all_images)} images, using all of them.")
        return all_images
    else:
        selected = random.sample(all_images, num_images)
        print(f"Randomly selected {num_images} images from {len(all_images)} available.")
        return selected


def hex_to_rgba(hex_color, alpha=0.1):
    """Convert hex color to RGBA tuple with transparency"""
    rgba = to_rgba(hex_color, alpha)
    return tuple(int(c * 255) for c in rgba)


def draw_boxes(image, labels, boxes, scores=None, is_gt=False, score_threshold=0.7):
    """
    Draw bounding boxes on image with colored, transparent fill

    Args:
        image: PIL Image
        labels: tensor of label indices
        boxes: tensor of bounding boxes [x0, y0, x1, y1]
        scores: tensor of confidence scores (None for ground truth)
        is_gt: whether drawing ground truth (True) or predictions (False)
        score_threshold: minimum score to show for predictions

    Returns:
        PIL Image with drawn boxes
    """
    # Create a copy of the image
    result_img = image.copy()
    draw = ImageDraw.Draw(result_img, 'RGBA')

    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    # Filter by confidence score for predictions
    valid_indices = torch.arange(len(labels))
    if scores is not None and not is_gt:
        valid_indices = torch.where(scores > score_threshold)[0]

    # Draw each box
    for idx in valid_indices:
        i = idx.item()
        label = labels[i].item()
        if not is_gt:
            label += 1
        box = boxes[i].tolist()

        # Get category color and name
        color_hex = CATEGORY_COLORS.get(label, '#808080')  # Gray for unknown categories
        color_rgba = hex_to_rgba(color_hex)
        category_name = CATEGORY_NAMES.get(label, f'Class {label}')

        # Box outline (solid)
        draw.rectangle(box, outline=color_hex, width=2)

        # Transparent fill
        draw.rectangle(box, fill=color_rgba)

        # Text background and label
        text = category_name
        if scores is not None and not is_gt:
            text = f"{category_name}: {scores[i].item():.2f}"

        # Draw text background
        text_size = draw.textbbox((0, 0), text, font=font)[2:]
        text_box = [box[0], box[1] - text_size[1] - 4, box[0] + text_size[0] + 4, box[1]]
        draw.rectangle(text_box, fill=color_hex)

        # Draw text
        draw.text((box[0] + 2, box[1] - text_size[1] - 2), text, fill='white', font=font)

    return result_img


def visualize_detection(image, gt_labels=None, gt_boxes=None, pred_labels=None, pred_boxes=None, pred_scores=None,
                        score_threshold=0.7):
    """
    Create visualization of ground truth and predictions side by side
    
    Args:
        image: PIL Image
        gt_labels, gt_boxes: ground truth annotations
        pred_labels, pred_boxes, pred_scores: model predictions
        score_threshold: minimum score threshold for showing predictions
    
    Returns:
        List of PIL Images [original, ground_truth, prediction]
    """

    # Draw ground truth if provided
    gt_image = None
    pred_image = None

    if gt_labels is not None and gt_boxes is not None:
        gt_image = draw_boxes(image, gt_labels, gt_boxes, is_gt=True)

    # Draw predictions if provided
    if pred_labels is not None and pred_boxes is not None:
        pred_image = draw_boxes(image, pred_labels, pred_boxes, pred_scores, is_gt=False,
                                score_threshold=score_threshold)

    return gt_image, pred_image


def save_visualization(prediction, output_dir='test_results_2', filename_base='detection'):
    """
    Save visualization images to disk
    
    Args:
        prediction: PIL Image with predictions drawn
        output_dir: directory to save images
        filename_base: base name for output files
    """
    os.makedirs(output_dir, exist_ok=True)

    if prediction is not None:
        prediction.save(os.path.join(output_dir, f"{filename_base}_prediction.jpg"))


def get_ground_truth_from_coco(image_id, coco_dataset, coco_to_model_id=None):
    """
    Extract ground truth boxes and labels for an image from CocoDetection dataset
    
    Args:
        image_id: COCO image id or filename to find in the dataset
        coco_dataset: CocoDetection dataset object
        coco_to_model_id: Optional mapping from COCO category ids to model category ids
    
    Returns:
        gt_labels: tensor of label indices
        gt_boxes: tensor of bounding boxes [x0, y0, x1, y1]
    """
    # Find the image in the dataset
    target = None
    img_idx = None

    # If image_id is a string (filename), find it in the dataset
    if isinstance(image_id, str):
        for idx, (_, image) in enumerate(coco_dataset.coco.imgs.items()):
            file_name = image['file_name']
            if file_name.startswith("./images"):
                file_name = file_name.replace("./images/VoFo-SEG/", "")
            if os.path.basename(file_name) == os.path.basename(image_id):
                img_idx = idx
                break
    else:
        # If image_id is an integer ID
        for idx, (img_id, _) in enumerate(coco_dataset.imgs.items()):
            if img_id == image_id:
                img_idx = idx
                break

    if img_idx is not None:
        target = coco_dataset.targets[img_idx]
    else:
        return None, None  # Return None instead of raising error for missing annotations

    # Extract bounding boxes and labels
    boxes = []
    labels = []

    for ann in target:
        # Get the bounding box
        bbox = ann['bbox']  # [x, y, width, height] format in COCO
        # Convert to [x0, y0, x1, y1] format
        x0, y0, w, h = bbox
        x1, y1 = x0 + w, y0 + h
        boxes.append([x0, y0, x1, y1])

        # Get the category id
        cat_id = ann['category_id']
        # Map COCO category ID to model category ID if mapping is provided
        if coco_to_model_id is not None:
            cat_id = coco_to_model_id.get(cat_id, cat_id)
        labels.append(cat_id)

    if not boxes:
        return None, None

    return torch.tensor(labels), torch.tensor(boxes)


class CustomCocoDetection(CocoDetection):
    """
    Extended CocoDetection class with direct access to targets
    """

    def __init__(self, root, annFile, transform=None, target_transform=None):
        super().__init__(root, annFile, transform, target_transform)
        # Pre-load all targets for easier access
        self.targets = []
        for img_id in self.ids:
            self.targets.append(self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)))


def process_single_image(model, image_path, args, coco_dataset=None, coco_to_model_id=None):
    """
    Process a single image and return results
    
    Args:
        model: Loaded model
        image_path: Path to the image file
        args: Command line arguments
        coco_dataset: Optional COCO dataset for ground truth
        coco_to_model_id: Optional category mapping
        
    Returns:
        ground_truth, prediction PIL Images
    """
    try:
        # Load image
        im_pil = Image.open(image_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None].to(args.device)

        # Run inference
        output = model(im_data, orig_size)
        pred_labels, pred_boxes, pred_scores = output

        # Load ground truth if provided
        gt_labels, gt_boxes = None, None
        if coco_dataset is not None:
            # Get ground truth for the current image
            image_id = os.path.basename(image_path)
            gt_labels, gt_boxes = get_ground_truth_from_coco(image_id, coco_dataset, coco_to_model_id)

            if gt_labels is not None:
                # Ensure proper device
                gt_labels = gt_labels.to(args.device)
                gt_boxes = gt_boxes.to(args.device)

        # Create visualizations
        ground_truth, prediction = visualize_detection(
            im_pil,
            gt_labels=gt_labels,
            gt_boxes=gt_boxes,
            pred_labels=pred_labels[0],
            pred_boxes=pred_boxes[0],
            pred_scores=pred_scores[0],
            score_threshold=args.threshold
        )

        # Save results
        filename_base = Path(image_path).stem
        save_visualization(prediction, args.output_dir, filename_base)

        return ground_truth, prediction

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, None


def main(args):
    """main function with batch processing support"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    # Setup COCO dataset if provided
    coco_dataset = None
    coco_to_model_id = None
    if args.coco_root and args.coco_ann:
        coco_dataset = CustomCocoDetection(args.coco_root, args.coco_ann)

        if args.category_map:
            with open(args.category_map, 'r') as f:
                coco_to_model_id = json.load(f)

    # Process images
    if args.image_dir:
        # Batch processing mode: select random images from directory
        selected_images = select_random_images(args.image_dir, args.num_images)

        ground_truths = []
        predictions = []

        print(f"Processing {len(selected_images)} images...")
        for i, image_path in enumerate(selected_images, 1):
            print(f"Processing image {i}/{len(selected_images)}: {os.path.basename(image_path)}")

            ground_truth, prediction = process_single_image(
                model, image_path, args, coco_dataset, coco_to_model_id
            )

            if ground_truth is not None:
                ground_truths.append(ground_truth)
            if prediction is not None:
                predictions.append(prediction)

        print(f"Successfully processed {len(predictions)} images")
        print(f"Results saved to: {args.output_dir}")

        # Optionally export results to PDF
        if args.export_pdf and predictions:
            pdf_path = os.path.join(args.output_dir, "batch_results.pdf")
            export_results(ground_truths, predictions, pdf_path)
            print(f"PDF report saved to: {pdf_path}")

    elif args.im_file:
        # Single image processing mode (original behavior)
        ground_truth, prediction = process_single_image(
            model, args.im_file, args, coco_dataset, coco_to_model_id
        )
        print(f"Single image processed. Results saved to: {args.output_dir}")

    else:
        raise ValueError("Either --image-dir or --im-file must be provided")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='RT-DETR Detection with batch processing support')
    parser.add_argument('-c', '--config', type=str, required=True, help='Model config file')
    parser.add_argument('-r', '--resume', type=str, required=True, help='Checkpoint to resume from')

    # Input options - either single image or directory
    parser.add_argument('-f', '--im-file', type=str, help='Single input image file')
    parser.add_argument('--image-dir', type=str, help='Directory containing images for batch processing')
    parser.add_argument('-n', '--num-images', type=int, default=100,
                        help='Number of random images to select from directory (default: 100)')

    # Model and processing options
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('-t', '--threshold', type=float, default=0.7, help='Score threshold')
    parser.add_argument('-o', '--output-dir', type=str, default='test_results_2', help='Output directory')

    # COCO dataset arguments
    parser.add_argument('--coco-root', type=str, default=None, help='COCO dataset root directory')
    parser.add_argument('--coco-ann', type=str, default=None, help='COCO annotation JSON file')
    parser.add_argument('--category-map', type=str, default=None,
                        help='JSON file mapping COCO category IDs to model category IDs')

    # Export options
    parser.add_argument('--export-pdf', action='store_true',
                        help='Export batch results to PDF report')

    # Legacy support
    parser.add_argument('-b', '--bulk', action='store_true',
                        help='Legacy bulk processing (deprecated, use --image-dir instead)')

    args = parser.parse_args()

    # Handle legacy bulk mode
    if args.bulk:
        print("Warning: --bulk is deprecated. Please use --image-dir instead.")
        if not args.image_dir and args.coco_root:
            args.image_dir = args.coco_root
            args.num_images = 9  # Legacy behavior

    # Set random seed for reproducibility
    random.seed(8)

    main(args)
