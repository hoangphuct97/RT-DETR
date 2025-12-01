from PIL import Image

def rotate_90_clockwise_and_save(input_path, output_path):
    img = Image.open(input_path)
    rotated = img.rotate(-90, expand=True)  # clockwise
    rotated.save(output_path)
    print(f"Saved rotated image to {output_path}")


if __name__ == '__main__':
    rotate_90_clockwise_and_save("../results/rt-detr/15161066_slice0240_prediction.jpg", "../results/rt-detr/15161066_slice0240_prediction.jpg")