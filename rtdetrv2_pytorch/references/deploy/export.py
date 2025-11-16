from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image as PDFImage, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO


def export_results(ground_truth_images, predicted_images, image_names, output_file="output/output.pdf"):
    """
    Create a PDF with image pairs (ground truth left, prediction right) in each row.
    
    Args:
        ground_truth_images (list): List of PIL.Image.Image objects for ground truth
        predicted_images (list): List of PIL.Image.Image objects for predictions
        output_file (str): Path to save the PDF document
    """
    # Verify inputs
    if len(ground_truth_images) != len(predicted_images):
        raise ValueError("Number of ground truth and predicted images must match")

    # Create PDF document
    doc = SimpleDocTemplate(output_file, pagesize=letter)

    # Prepare elements for the PDF
    elements = []
    styles = getSampleStyleSheet()

    # Add title
    title = Paragraph("<b>Image Comparison: Ground Truth vs Predictions</b>", styles['Title'])
    elements.append(title)
    elements.append(Paragraph("<br/><br/>", styles['Normal']))  # Add some space

    # Convert PIL images to PDF-compatible format and create table data
    table_data = []
    for i, (gt_img, pred_img) in enumerate(zip(ground_truth_images, predicted_images)):
        # Convert PIL images to bytes
        def pil_to_bytes(pil_img):
            img_byte_arr = BytesIO()
            pil_img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            return img_byte_arr

        gt_bytes = pil_to_bytes(gt_img)
        pred_bytes = pil_to_bytes(pred_img)

        # Create PDF Image objects with consistent size
        img_width = 3 * inch
        gt_pdf_img = PDFImage(gt_bytes, width=img_width, height=img_width)
        pred_pdf_img = PDFImage(pred_bytes, width=img_width, height=img_width)

        # Add row with both images and their labels
        row = [
            Paragraph(f"<b>Ground Truth {i + 1} ({image_names[i]})</b>", styles['Normal']),
            Paragraph(f"<b>Prediction {i + 1}</b>", styles['Normal'])
        ]
        table_data.append(row)

        row = [gt_pdf_img, pred_pdf_img]
        table_data.append(row)

        # Add empty row for spacing between pairs
        if i < len(ground_truth_images) - 1:
            table_data.append(["", ""])

    # Create table with all images
    table = Table(table_data, colWidths=[img_width, img_width])

    # Add table style
    table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))

    elements.append(table)

    # Build PDF
    doc.build(elements)
    print(f"PDF saved as {output_file}")
