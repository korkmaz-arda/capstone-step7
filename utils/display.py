import os
import sys
import random
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from utils.bbox import poly2bbox


def display_labels(labels, path, box_color="red", show=True, save=False, output_dir="./bbox_images"):
    img = Image.open(path)
    width, height = img.size
    
    draw = ImageDraw.Draw(img)
    for label in labels:
        bbox = poly2bbox(label['bbox'])        
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)

        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=7)
        draw.text((x1 + 8, y1 + 8), label['class'], fill="black", font=ImageFont.load_default(size=15))

    if save:
        output_path = os.path.join(output_dir, f"{os.path.basename(path)}")
        img.save(output_path)

    if show:
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        
        
def display_bbox(bbox, path, show=True, save=False, output_dir="./bbox_images"):
    img = Image.open(path)
    width, height = img.size

    x1 = int(bbox[0] * width)
    y1 = int(bbox[1] * height)
    x2 = int(bbox[2] * width)
    y2 = int(bbox[3] * height)

    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=7)

    if save:
        base_name = os.path.splitext(os.path.basename(path))[0]
        ext = os.path.splitext(path)[1]
        output_path = os.path.join(output_dir, f"{base_name}{ext}")
        counter = 2
        while os.path.exists(output_path):
            output_path = os.path.join(output_dir, f"{base_name}_{counter}{ext}")
            counter += 1
        
        img.save(output_path)

    if show:
        print(image_path) # commrnt out
        plt.imshow(img)
        plt.axis("off")
        plt.show()


def display_all(
        img_path, 
        pred_labels, 
        gt_labels, 
        pred_color="red", 
        gt_color="blue", 
        show=True, 
        save=False, 
        output_dir="./bbox_images"
    ):
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    
    draw = _annotate_img_canvas(img, draw, pred_labels, pred_color)
    draw = _annotate_img_canvas(img, draw, gt_labels, gt_color)

    if save:
        output_path = os.path.join(output_dir, f"{os.path.basename(path)}")
        img.save(output_path)
    
    if show:
        plt.imshow(img)
        plt.axis("off")
        plt.show()

def _annotate_img_canvas(img: Image, canvas: ImageDraw.ImageDraw, labels, box_color):
    width, height = img.size

    for label in labels:
        bbox = poly2bbox(label['bbox'])        
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)

        canvas.rectangle([x1, y1, x2, y2], outline=box_color, width=7)
        canvas.text((x1 + 8, y1 + 8), label['class'], fill="black", font=ImageFont.load_default(size=15))
    
    return canvas


def display_ground_truth(gt, no_images=5):
    display_count = 0
    sampled_truths = random.sample(gt, min(no_images, len(gt)))
    
    displayed_images = set()
    
    for truth in sampled_truths:
        image_path, labels = next(iter(truth.items()))
        
        if labels != [] and image_path not in displayed_images:
            display_labels(labels, image_path, show=True, save=False)
            displayed_images.add(image_path)
            display_count += 1
