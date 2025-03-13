import matplotlib.pyplot as plt
import numpy as np
import json
import os
import tqdm
import argparse

# Shape attributes: 50×50 rect, 40px radius circle
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
task_types = ['classification', 'detection', 'segmentation']
shape_attribs = {
    'rect': [50, 50],
    'circle': [40]
}

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", help="path to save dataset", type=str)
parser.add_argument("--image_size", nargs='+', default=(500, 500), type=int,
                    help="(width height)")
parser.add_argument("--num_images", type=int, default=10,
                    help="number of images")
parser.add_argument("--shapes", nargs='+', default=['circle', 'rect'],
                    help="which shapes to generate")
parser.add_argument("--shape_color", type=str, default='blue')
parser.add_argument("--shuffle_color", type=bool, default=False)
parser.add_argument("--task_type", type=str, default='detection',
                    help="classification/detection/segmentation")

args = parser.parse_args()
image_size = args.image_size
shapes = list(set(args.shapes))
num_images = args.num_images
save_dir = args.save_dir
shape_color = args.shape_color
shuffle_color = args.shuffle_color
task_type = args.task_type

assert save_dir, "specify save directory"
assert shape_color in colors, "Available colors: " + str(colors)
assert task_type in task_types, "Available task types: " + str(task_types)

image_w, image_h = image_size

print("[DEBUG] run.py arguments:")
print("  save_dir =", save_dir)
print("  image_size =", image_size)
print("  num_images =", num_images)
print("  shapes =", shapes)
print("  shape_color =", shape_color, ", shuffle_color =", shuffle_color)
print("  task_type =", task_type)

def make_shape(x, y, shape_idx):
    """Create a circle or rectangle at (x,y)."""
    if shapes[shape_idx] == 'rect':
        color = (shuffle_color * colors[np.random.randint(0, 7)]
                 + (1 - shuffle_color) * shape_color)
        w, h = shape_attribs["rect"]
        return plt.Rectangle((x, y), w, h, color=color)
    else:  # circle
        color = (shuffle_color * colors[np.random.randint(0, 7)]
                 + (1 - shuffle_color) * shape_color)
        r = shape_attribs["circle"][0]
        return plt.Circle((x, y), r, color=color)

def gen_bbox(x, y, shape_idx):
    """Generate bounding box for shape at (x,y)."""
    if shapes[shape_idx] == 'rect':
        w, h = shape_attribs["rect"]
        return {
            'object': 'rect',
            'x': x,
            'y': y,
            'w': w,
            'h': h
        }
    else:  # circle
        r = shape_attribs["circle"][0]
        return {
            'object': 'circle',
            'x': x - r,
            'y': y - r,
            'w': 2 * r,
            'h': 2 * r
        }

def detection_gen():
    def make_dirs():
        img_path = os.path.join(save_dir, "images")
        lab_path = os.path.join(save_dir, "labels_json")
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(lab_path, exist_ok=True)
        return img_path, lab_path

    img_path, lab_path = make_dirs()
    print(f"[DEBUG] detection_gen: Saving images to {img_path}")
    print(f"[DEBUG] detection_gen: Saving labels to {lab_path}")

    for n in tqdm.tqdm(range(num_images)):
        objs = []
        obj_bbox = []

        # Generate multiple shapes per image
        num_shapes_in_image = np.random.randint(3, 7)  # e.g. 3..6
        print(f"[DEBUG] Image {n}: Generating {num_shapes_in_image} shapes.")

        for _ in range(num_shapes_in_image):
            shape_idx = np.random.randint(0, len(shapes))
            if shapes[shape_idx] == 'rect':
                w, h = shape_attribs["rect"]
                x = np.random.randint(0, image_w - w)
                y = np.random.randint(0, image_h - h)
            else:  # circle
                r = shape_attribs["circle"][0]
                x = np.random.randint(r, image_w - r)
                y = np.random.randint(r, image_h - r)

            shape_obj = make_shape(x, y, shape_idx)
            bbox = gen_bbox(x, y, shape_idx)
            objs.append(shape_obj)
            obj_bbox.append(bbox)

            print(f"[DEBUG] {shapes[shape_idx]} at (x={x}, y={y}) -> bbox={bbox}")

        # Create figure
        fig, ax = plt.subplots(figsize=(image_w/100, image_h/100), dpi=100)
        ax.set_xlim([0, image_w])
        ax.set_ylim([0, image_h])
        ax.invert_yaxis()  # Make (0,0) top-left
        ax.axis("off")

        # Fill entire canvas, no margins
        ax.set_position([0, 0, 1, 1])

        for obj in objs:
            ax.add_artist(obj)

        # Save image with no bounding or padding
        img_filename = os.path.join(img_path, f"shapes_{n}.png")
        print(f"[DEBUG] Saving image: {img_filename}")
        fig.savefig(img_filename, bbox_inches=None, pad_inches=0)
        plt.close(fig)

        # Save bounding boxes
        json_filename = os.path.join(lab_path, f"shapes_{n}.json")
        print(f"[DEBUG] Saving JSON: {json_filename}")
        with open(json_filename, "w") as outfile:
            json.dump(obj_bbox, outfile)

    print("Generated dataset in", save_dir)

def classification_gen():
    pass

def segmentation_gen():
    pass

if task_type == "classification":
    classification_gen()
elif task_type == "detection":
    detection_gen()
elif task_type == "segmentation":
    segmentation_gen()
