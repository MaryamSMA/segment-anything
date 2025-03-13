import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
import argparse

def json_listing(json_data):
    label = []
    for region in json_data:
        label.append([region['x'], region['y'], region['w'], region['h']])
    return label

def bbox_plot(img, boxes):
    height, width = img.shape[0], img.shape[1]

    fig, ax = plt.subplots(figsize=(width/100, height/100))
    # Remove or comment out any invert_yaxis() call
    # plt.gca().invert_yaxis()  # DON'T do this anymore


    # Map the image so that (0,0) in the image is at top-left, (width,height) at bottom-right
    ax.imshow(
        img,
        extent=[0, width, 0, height],  # left, right, bottom, top
        origin='upper'
    )
    
    # Make sure x-axis is 0..width, y-axis is 0..height
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])

    # You can set the aspect to 'auto' so it uses the full figure space
    ax.set_aspect('auto')

    # Draw bounding boxes
    for (x, y, w, h) in boxes:
        rect = plt.Rectangle(
            (x, y), w, h,
            linewidth=1, edgecolor='g', facecolor="none"
        )
        ax.add_patch(rect)

    plt.show()


def bounding_boxes(path):
    # List all entries in the dataset dir, ignoring '.' and '..'
    all_entries = sorted(os.listdir(path))
    valid_entries = [d for d in all_entries if not d.startswith('.')]

    # We expect exactly 2 subfolders: 'images' and 'labels_json'
    # If there's anything else (like .DS_Store), we'll skip it
    # or handle an error if there are more than 2 folders
    # Filter to folders only
    valid_entries = [d for d in valid_entries 
                     if os.path.isdir(os.path.join(path, d))]

    # If you expect precisely 2 folders:
    if len(valid_entries) != 2:
        print(f"❌ Error: Expected exactly 2 folders under {path}, found: {valid_entries}")
        return

    img_dir, lab_dir = valid_entries
    img_path = os.path.join(path, img_dir)
    lab_path = os.path.join(path, lab_dir)

    # Now list the images and label JSONs
    img_list = sorted([f for f in os.listdir(img_path) if f.endswith('.png')])
    lab_list = sorted([f for f in os.listdir(lab_path) if f.endswith('.json')])

    for im, lab in zip(img_list, lab_list):
        img = plt.imread(os.path.join(img_path, im))
        print("Image file:", im)
        print("Label file:", lab)
        with open(os.path.join(lab_path, lab), 'r') as json_data_file:
            data = json.load(json_data_file)
            box_list = json_listing(data)
        bbox_plot(img, box_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", help="dataset path to be visualized")
    args = parser.parse_args()
    bounding_boxes(args.dataset_dir)