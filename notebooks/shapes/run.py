import matplotlib.pyplot as plt
import numpy as np
import json
import os
import tqdm
import argparse
import cv2

# ============================
# 🔹 Define Dataset Parameters
# ============================
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
shape_attribs = {'rect': [4, 4], 'circle': [3]}  # Shape sizes auto-adjust
categories = [
    {"id": 1, "name": "circle", "supercategory": "shape"},
    {"id": 2, "name": "rect", "supercategory": "shape"}
]

# ============================
# 🔹 Argument Parser
# ============================
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, help="Dataset save directory")
parser.add_argument("--image_size", nargs=2, default=[500, 500], type=int)  # Dynamically controlled
parser.add_argument("--num_images", type=int, default=10)
parser.add_argument("--shapes", nargs='+', default=['circle', 'rect'])
parser.add_argument("--shape_color", type=str, default='blue')
parser.add_argument("--shuffle_color", type=bool, default=False)
parser.add_argument("--task_type", type=str, default='segmentation')

args = parser.parse_args()

# ============================
# 🔹 Dataset Variables
# ============================
image_w, image_h = args.image_size  # Dynamically assigns values
shapes = args.shapes
num_images = args.num_images
save_dir = args.save_dir
shape_color = args.shape_color
shuffle_color = args.shuffle_color
task_type = args.task_type

assert save_dir, "❌ Specify save directory!"
assert shape_color in colors, f"❌ Available colors: {colors}"
assert task_type in ["classification", "detection", "segmentation"], "❌ Invalid task type!"

# ============================
# 🔹 Directories
# ============================
img_path = os.path.join(save_dir, "images")
anno_path = os.path.join(save_dir, "annotations")
os.makedirs(img_path, exist_ok=True)
os.makedirs(anno_path, exist_ok=True)

# ============================
# 🔹 COCO Annotation Template
# ============================
coco_data = {"images": [], "annotations": [], "categories": categories}
annotation_id = 1

# ============================
# 🔹 Shape Generation Functions
# ============================
def make_shape(x, y, shape_type):
    """ Creates a shape (circle or rectangle) at (x, y). """
    color = shape_color if not shuffle_color else np.random.choice(colors)

    if shape_type == "rect":
        return plt.Rectangle((x, y), shape_attribs["rect"][0], shape_attribs["rect"][1], color=color)
    elif shape_type == "circle":
        return plt.Circle((x, y), shape_attribs["circle"][0], color=color)

def gen_bbox(x, y, shape_type):
    """ Generates a bounding box for the shape. """
    if shape_type == "rect":
        return [x, y, shape_attribs["rect"][0], shape_attribs["rect"][1]]
    elif shape_type == "circle":
        return [
            x - shape_attribs["circle"][0],
            y - shape_attribs["circle"][0],
            2 * shape_attribs["circle"][0],
            2 * shape_attribs["circle"][0]
        ]

# ============================
# 🔹 Generate Dataset
# ============================
for img_id in tqdm.tqdm(range(num_images), desc="Generating dataset"):
    objs = []
    obj_bboxes = []

    num_shapes = np.random.randint(1, 3)  # Small number of shapes per image

    for _ in range(num_shapes):
        shape_type = np.random.choice(shapes)
        x, y = np.random.randint(2, image_w - 2), np.random.randint(2, image_h - 2)

        objs.append(make_shape(x, y, shape_type))
        obj_bboxes.append((shape_type, gen_bbox(x, y, shape_type)))

    # Save Image
    fig, ax = plt.subplots(figsize=(image_w / 100, image_h / 100), dpi=100)  # Dynamically adapt size
    ax.set_xlim([0, image_w])
    ax.set_ylim([0, image_h])
    plt.gca().invert_yaxis()
    ax.axis("off")

    for obj in objs:
        ax.add_artist(obj)

    img_filename = f"shapes_{img_id}.png"
    img_filepath = os.path.join(img_path, img_filename)
    fig.savefig(img_filepath, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Save to COCO Format
    coco_data["images"].append({
        "id": img_id,
        "width": image_w,
        "height": image_h,
        "file_name": img_filename
    })

    for shape_type, bbox in obj_bboxes:
        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": img_id,
            "category_id": 1 if shape_type == "circle" else 2,
            "bbox": bbox,
            "segmentation": [],  # To be filled after SAM
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        })
        annotation_id += 1

# ============================
# 🔹 Save COCO Annotations
# ============================
coco_annotation_file = os.path.join(anno_path, "coco_annotations.json")
with open(coco_annotation_file, "w") as f:
    json.dump(coco_data, f, indent=4)

print(f"✅ Dataset and COCO annotations saved to {save_dir}")