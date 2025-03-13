import matplotlib.pyplot as plt
import numpy as np
import json
import os
import tqdm
import argparse

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
task_types = ['classification', 'detection', 'segmentation']

# The following shape_attribs values will be computed dynamically later
# shape_attribs = {
#     'rect': [50, 50],
#     'circle': [40]
# }

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", help="path to where you want to save the dataset", type=str)
parser.add_argument("--image_size", help="size of the image (width height)", nargs='+', default=(500, 500), type=int)
parser.add_argument("--num_images", help="number of images for your dataset", type=int, default=10)
parser.add_argument("--shapes", help="shapes that you require in your dataset. Available: %s" % str(task_types),
                    nargs='+', default=['circle', 'rect'])
parser.add_argument("--shape_color", help="specify a particular color for all the shapes", type=str, default='blue')
parser.add_argument("--shuffle_color", help="shuffle colors for the shapes", type=bool, default=False)
parser.add_argument("--task_type", help="specify type of task. Available: %s" % str(task_types), type=str, default='detection')

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

# --- Dynamic shape attributes ---
# Here we set the rectangle size to be 10% of the image dimensions
# and the circle radius to be 8% of the image width.
rect_size = [int(0.1 * image_w), int(0.1 * image_h)]
circle_radius = int(0.08 * image_w)
shape_attribs = {
    'rect': rect_size,
    'circle': [circle_radius]
}
print("[DEBUG] Dynamic shape attributes:")
print("  rect size =", rect_size)
print("  circle radius =", circle_radius)
# --- End dynamic shape attributes ---

print("[DEBUG] run.py arguments:")
print("  save_dir =", save_dir)
print("  image_size =", image_size)
print("  num_images =", num_images)
print("  shapes =", shapes)
print("  shape_color =", shape_color, ", shuffle_color =", shuffle_color)
print("  task_type =", task_type)

def make_shape(x, y, shape_idx):
    """Create the matplotlib shape (circle or rectangle) at (x,y)."""
    if shapes[shape_idx] == 'rect':
        color = (shuffle_color * colors[np.random.randint(0, 7)]
                 + (1 - shuffle_color) * shape_color)
        w, h = shape_attribs["rect"]
        return plt.Rectangle((x, y), w, h, color=color)
    else:  # 'circle'
        color = (shuffle_color * colors[np.random.randint(0, 7)]
                 + (1 - shuffle_color) * shape_color)
        r = shape_attribs["circle"][0]
        return plt.Circle((x, y), r, color=color)

def gen_bbox(x, y, shape_idx):
    """Generate the bounding box for the shape at (x, y)."""
    if shapes[shape_idx] == 'rect':
        w, h = shape_attribs["rect"]
        return {
            'object': 'rect',
            'x': x,
            'y': y,
            'w': w,
            'h': h
        }
    else:  # 'circle'
        r = shape_attribs["circle"][0]
        return {
            'object': 'circle',
            'x': x - r,
            'y': y - r,
            'w': 2 * r,
            'h': 2 * r
        }

def detection_gen():
    """Generate a detection dataset with multiple shapes per image."""
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

        # Generate a random number of shapes per image (e.g., 3 to 6)
        num_shapes_in_image = np.random.randint(3, 7)
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

        # Create the figure with the full image size
        fig, ax = plt.subplots(figsize=(image_w/100, image_h/100), dpi=100)
        ax.set_xlim([0, image_w])
        ax.set_ylim([0, image_h])
        ax.invert_yaxis()  # Make (0,0) top-left
        ax.axis("off")
        ax.set_position([0, 0, 1, 1])

        for obj in objs:
            ax.add_artist(obj)

        img_filename = os.path.join(img_path, f"shapes_{n}.png")
        print(f"[DEBUG] Saving image: {img_filename}")
        fig.savefig(img_filename, bbox_inches=None, pad_inches=0)
        plt.close(fig)

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
