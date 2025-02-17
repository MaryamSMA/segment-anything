# **SAM-Shapes-Dataset**
An implementation of **Segment Anything Model (SAM)** with a **Shapes Dataset**, supporting different resolutions of generated shape images.

## 📌 Project Overview
This repository is based on the **Roboflow SAM tutorial**:  
🔗 [Original Roboflow Notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-anything-with-sam.ipynb)

It has been **modified** to work with the **Shapes dataset**:  
🔗 [Shapes Dataset Repository](https://github.com/cjpurackal/shapes/tree/master)

## 🔹 Modifications & Enhancements
✅ **Updated `run.py`** to generate synthetic shape images  
✅ **Added COCO-style annotations** to ensure compatibility with the SAM pipeline  
✅ **Supports multiple resolutions** (16×16, 28×28, 128×128, 256×256)  
✅ **Works with different shapes** (circles, rectangles)  
✅ **Saves outputs in a structured format:**
   - 📂 `dataset/images/` → Stores generated images  
   - 📂 `dataset/annotations/shapes_coco.json` → Stores annotations in COCO format  

This implementation allows testing how **SAM performs across different image resolutions** for segmentation tasks.

## 💡 Future Additional Features  
🔲 **Support for grayscale shape images** for testing SAM on monochrome datasets  
🔲 **Incorporation of more complex shapes** (e.g., polygons, triangles) to improve segmentation robustness  
🔲 **Enhancing SAM using adapters** to fine-tune its performance for synthetic datasets  
🔲 **Integrating a quantum layer** to explore hybrid quantum-classical segmentation models  

## 🔬 Experiment Overview

## 🧪 Experiment Configurations

| **Experiment Name**     | **Experiment Label (Ref)** | **Description** | **Dimensions (Before)** | **Dimensions (After)** | **Architecture Placement** |
|------------------------|--------------------------|---------------|----------------------|----------------------|--------------------------|
| **BasicSAMSh**         | S1                       | Run the original SAM pipeline to obtain baseline classical segmentation results. |  |  | Classical SAM |
| **SAMAdaptSh**         | S2                       | Develop an adapter specifically optimized for the shape dataset to improve segmentation accuracy. |  |  |   |
| **BasicSAMQuantum**    | SQ1                      | Integrate quantum components into the SAM pipeline to analyze performance improvements. |  |  |   |
| **SAMAdaptShQuantum**  | SQ2                      | Combine the shape-specific adapter with quantum enhancements for optimized segmentation. |  |  |  |

## 📊 Dataset Configurations

| **Dataset Label (Version)** | **Shape Type Count** | **Shape Types**         | **Generated Samples** | **Shape Complexity (Edge Blur)** | **Color – Grayscale** | **Image Size** |
|----------------------------|----------------------|------------------------|----------------------|---------------------------------|--------------------|----------------|
| **Shapes v1**              | 2                    | Circle, Rectangle      | 5                    | No                              | Default = 'Blue'   | 16×16, 28×28, 128×128, 256×256 |
| **Shapes v2**              | TBD                  | TBD                    | TBD                  | TBD                             | TBD                | TBD |
| **Shapes v3**              | TBD                  | TBD                    | TBD                  | TBD                             | TBD                | TBD |

*TBD: to be defined
### 📝 Additional Notes:
- **Keep plotting dimensions** → Print the shape before and after the mask is applied.
- **Throughout the pipeline** → Display the shape at different transformation stages.
- **Shape function** → Track where the image changes in size (print the image size at each step).

