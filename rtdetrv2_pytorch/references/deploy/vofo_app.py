import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import time

from src.core import YAMLConfig

# Page config
st.set_page_config(
    page_title="VoFo-DETR Object Detection",
    page_icon="🎯",
    layout="wide"
)

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs

@st.cache_resource
def load_model(config_path, checkpoint_path, device):
    """Load and cache the RT-DETR model"""
    cfg = YAMLConfig(config_path, resume=checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']

    cfg.model.load_state_dict(state)
    model = Model(cfg).to(device)
    model.eval()

    return model

def draw_boxes(image, labels, boxes, scores, class_names=None, threshold=0.5):
    """Draw bounding boxes on image"""
    draw = ImageDraw.Draw(image)

    # Try to use a better font
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()

    colors = [
        '#FF3838', '#18A558', '#4361EE', '#FFB300', '#9C27B0', '#00BCD4'
    ]

    for label, box, score in zip(labels, boxes, scores):
        if score < threshold:
            continue

        label_idx = int(label.item())
        score_val = float(score.item())

        # Get box coordinates
        x1, y1, x2, y2 = box.tolist()

        # Choose color
        color = colors[label_idx % len(colors)]

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Prepare label text
        if class_names and label_idx < len(class_names):
            text = f"{class_names[label_idx]}: {score_val:.2f}"
        else:
            text = f"Class {label_idx}: {score_val:.2f}"

        # Draw label background
        bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1), text, fill='white', font=font)

    return image

def run_inference(model, image, device):
    """Run inference on the uploaded image"""
    # Get original size
    w, h = image.size
    orig_size = torch.tensor([w, h])[None].to(device)

    # Prepare transforms
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    # Transform image
    im_data = transforms(image)[None].to(device)

    # Run inference with timing
    start_time = time.time()

    with torch.no_grad():
        output = model(im_data, orig_size)

    inference_time = time.time() - start_time

    pred_labels, pred_boxes, pred_scores = output

    return pred_labels[0], pred_boxes[0], pred_scores[0], inference_time

def main():
    st.title("🎯 RT-DETR Object Detection")
    st.markdown("Upload an image to detect objects using RT-DETR model")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    config_path = st.sidebar.text_input(
        "Config Path",
        value="configs/rtdetr/rtdetr_r50vd_6x_coco.yml",
        help="Path to your RT-DETR config file"
    )

    checkpoint_path = st.sidebar.text_input(
        "Checkpoint Path",
        value="checkpoints/rtdetr_r50vd_6x_coco.pth",
        help="Path to your model checkpoint"
    )

    device = st.sidebar.selectbox(
        "Device",
        ["cuda", "cpu", "mps"],
        index=0 if torch.cuda.is_available() else 1
    )

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    # Optional: COCO class names
    coco_classes = st.sidebar.checkbox("Use COCO class names", value=True)

    if coco_classes:
        class_names = [
            'L_Vocal Fold', 'L_Arytenoid cartilage', 'Benign lesion', 
            'Malignant lesion', 'R_Vocal Fold', 'R_Arytenoid cartilage'
        ]
    else:
        class_names = None

    # Load model button
    if st.sidebar.button("Load Model"):
        try:
            with st.spinner("Loading model..."):
                model = load_model(config_path, checkpoint_path, device)
                st.session_state['model'] = model
                st.session_state['device'] = device
            st.sidebar.success("Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
            return

    # Main content area
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image for object detection"
    )

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file).convert('RGB')

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, width=480)

        # Run inference if model is loaded
        if 'model' in st.session_state:
            if st.button("Run Inference", type="primary"):
                with st.spinner("Running inference..."):
                    try:
                        model = st.session_state['model']
                        device = st.session_state['device']

                        labels, boxes, scores, inference_time = run_inference(
                            model, image, device
                        )

                        # Draw boxes on image
                        result_image = image.copy()
                        result_image = draw_boxes(
                            result_image, labels, boxes, scores,
                            class_names, confidence_threshold
                        )

                        # Display results
                        with col2:
                            st.subheader("Detection Results")
                            st.image(result_image, width=480)

                        # Display metrics
                        st.success(f"✅ Inference completed!")

                        metric_col1, metric_col2, metric_col3 = st.columns(3)

                        with metric_col1:
                            st.metric("Inference Time", f"{inference_time*1000:.2f} ms")

                        with metric_col2:
                            num_detections = (scores >= confidence_threshold).sum().item()
                            st.metric("Detections", num_detections)

                        # with metric_col3:
                        #     st.metric("Image Size", f"{image.size[0]}×{image.size[1]}")

                        # Show detailed results
                        with st.expander("📊 Detailed Detection Results"):
                            filtered_indices = scores >= confidence_threshold
                            filtered_labels = labels[filtered_indices]
                            filtered_boxes = boxes[filtered_indices]
                            filtered_scores = scores[filtered_indices]

                            if len(filtered_labels) > 0:
                                for idx, (label, box, score) in enumerate(
                                        zip(filtered_labels, filtered_boxes, filtered_scores)
                                ):
                                    label_idx = int(label.item())
                                    class_name = class_names[label_idx] if class_names and label_idx < len(class_names) else f"Class {label_idx}"

                                    st.write(f"**Detection {idx+1}:**")
                                    st.write(f"- Class: {class_name}")
                                    st.write(f"- Confidence: {score.item():.4f}")
                                    st.write(f"- Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
                                    st.write("---")
                            else:
                                st.info("No detections above confidence threshold")

                    except Exception as e:
                        st.error(f"Error during inference: {str(e)}")
        else:
            st.warning("⚠️ Please load the model first using the sidebar.")

if __name__ == "__main__":
    main()