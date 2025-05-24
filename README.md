# PhytoBeacon 🧪
Real-time detection of Phytoplankton using Deep Learning 

Here’s a detailed and well-structured `README.md` content for your Python Tkinter application using YOLO for object detection:

---

# 🦠 PhytoBeacon - Deep Learning-Powered Phytoplankton Detection GUI

Phytoscope is a desktop application built using Python and Tkinter that leverages the power of the [YOLO (You Only Look Once)](https://github.com/ultralytics/ultralytics) object detection model to process, visualize, and export object detection results. It enables users to easily annotate images, view results, and export annotations in PASCAL VOC XML format.

---

## ✨ Features

* **📁 Select Multiple Images**: Choose multiple image files (`.jpg`, `.jpeg`, `.png`) from your system using a GUI file selector.
* **🧠 YOLO Model Integration**: Automatically detects objects using a pre-trained YOLOv8 model (`best_yolo11.pt`).
* **🖼 Annotated Image Viewer**: Display annotated bounding boxes on each image with object labels and confidence scores.
* **⬅️➡️ Image Navigation**: Browse through the images using "Previous" and "Next" buttons.
* **📝 Export to XML (PASCAL VOC)**: Save detection labels in XML format for each image.
* **💾 Save Annotated Images**: Export the YOLO-detected, annotated images to any desired folder.
* **📦 Save All Results**: Save both XML label files and annotated images in bulk.
* **🔎 Label Inspection**: Click on a label to highlight the corresponding bounding box in the image.
* **📂 Organized Output**: Automatically stores annotations in `annotated_results/` and `labels/` directories.

---

## 📦 Requirements

Ensure the following Python packages are installed:

```bash
pip install ultralytics Pillow opencv-python tk
```

Or install all dependencies from a `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🚀 Getting Started

1. **Clone or Download the Repository**

   ```bash
   git clone https://github.com/your-repo/phytoscope.git
   cd phytoscope
   ```

2. **Ensure YOLO Weights Are Available**

   The app expects your YOLOv8 model weights at:

   ```
   runs/detect/train/weights/best_yolo11.pt
   ```

   Make sure this file exists and is trained for your specific task.

3. **Run the App**

   ```bash
   python phytoscope.py
   ```

---

## 🖥 How to Use

1. **Select Images**
   Click `Select Images` to choose one or more images for object detection.

2. **Process Images**
   Click `Process Images` to run YOLO detection and view annotated images with bounding boxes and labels.

3. **Navigate**
   Use `Next Image` and `Previous Image` to move through your selected images.

4. **Inspect Labels**
   Click on a label in the label list to highlight the corresponding bounding box in the image.

5. **Save Annotated Image**
   Save the currently displayed annotated image using the `Save Annotated Image` button.

6. **Save All Results**
   Click `Save Results` to export:

   * All annotated images.
   * All label files in PASCAL VOC (XML) format.

---

## 🗃 Folder Structure

Upon processing, the following folders will be created:

* **annotated\_results/** – Stores annotated versions of your input images.
* **labels/** – Stores XML label files in PASCAL VOC format.

You can also choose a custom directory when saving results.

---

## 📄 Output Format

### Annotated Images

* JPEG files with bounding boxes and labels drawn on them.

### XML Labels (PASCAL VOC)

Each XML file includes:

* Object `name`
* Bounding box coordinates: `xmin`, `ymin`, `xmax`, `ymax`
* Image filename reference

Example snippet:

```xml
<object>
  <name>plant_disease</name>
  <bndbox>
    <xmin>120</xmin>
    <ymin>60</ymin>
    <xmax>250</xmax>
    <ymax>190</ymax>
  </bndbox>
</object>
```

---

## 🧠 Model Details

This app uses the Ultralytics `YOLOv8` model. You can customize or retrain your model and update the path in the line below:

```python
self.model = YOLO('runs/detect/train/weights/best_yolo11.pt')
```

---

## 🛠 Troubleshooting

* **Model not found?**
  Make sure `best_yolo11.pt` exists at the specified path.

* **No predictions shown?**
  Ensure the model is trained correctly and is detecting the target classes in your images.

* **App not launching?**
  Ensure you have a display environment available (e.g., don't run it in a headless server without GUI).

---

## 👨‍💻 Author

Developed by **\[Your Name]**
🔗 GitHub: [@your-username]((https://github.com/Riddhiman-1098))

---

Let me know if you'd like this turned into a Markdown file or want a version formatted for a website or documentation portal.
