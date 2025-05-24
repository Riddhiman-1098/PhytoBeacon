import os
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import shutil
import logging
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def save_labels_to_xml(image_path, labels, output_folder):
    """Convert labels into an XML file and save it."""
    root = ET.Element("annotation")
    filename = ET.SubElement(root, "filename")
    filename.text = os.path.basename(image_path)

    for label, bbox in labels:
        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        name.text = str(label)
        bbox_el = ET.SubElement(obj, "bndbox")
        xmin, ymin, xmax, ymax = bbox
        ET.SubElement(bbox_el, "xmin").text = str(xmin)
        ET.SubElement(bbox_el, "ymin").text = str(ymin)
        ET.SubElement(bbox_el, "xmax").text = str(xmax)
        ET.SubElement(bbox_el, "ymax").text = str(ymax)

    tree = ET.ElementTree(root)
    output_file = output_folder / f"{Path(image_path).stem}.xml"
    tree.write(output_file)

class YOLOTkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phytoscope")
        self.model = YOLO('runs/detect/train/weights/best_yolo11.pt')

        self.image_paths = []
        self.current_image_index = 0
        self.annotated_images = []
        self.labels_data = []

        self.annotated_folder = Path('annotated_results')
        self.label_folder = Path('labels')

        for folder in [self.annotated_folder, self.label_folder]:
            folder.mkdir(parents=True, exist_ok=True)

        self.setup_ui()

    def setup_ui(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        self.select_button = tk.Button(frame, text="Select Images", command=self.select_images)
        self.select_button.pack(side=tk.LEFT, padx=10)

        self.process_button = tk.Button(frame, text="Process Images", command=self.process_images, state=tk.DISABLED)
        self.process_button.pack(side=tk.LEFT, padx=10)

        self.prev_button = tk.Button(frame, text="Previous Image", command=self.prev_image, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=10)

        self.next_button = tk.Button(frame, text="Next Image", command=self.next_image, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=10)

        self.save_button = tk.Button(frame, text="Save Results", command=self.save_results, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=10)

        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        self.labels_frame = tk.Frame(self.root)
        self.labels_frame.pack()

        self.labels_listbox = tk.Listbox(self.labels_frame, height=10, width=30)
        self.labels_listbox.pack(side=tk.LEFT, padx=10)
        self.labels_listbox.bind("<<ListboxSelect>>", self.highlight_bbox)

        self.info_label = tk.Label(self.root, text="Select images to begin", font=("Arial", 12))
        self.info_label.pack()

        self.save_annotated_button = tk.Button(self.root, text="Save Annotated Image",
                                               command=self.save_annotated_image, state=tk.DISABLED)
        self.save_annotated_button.pack(pady=10)

    def select_images(self):
        filetypes = [('Image files', '*.jpg *.jpeg *.png')]
        paths = filedialog.askopenfilenames(title='Select Images', filetypes=filetypes)
        if paths:
            self.image_paths = paths
            self.current_image_index = 0
            self.process_button.config(state=tk.NORMAL)

    def process_images(self):
        self.annotated_images = []
        self.labels_data = []

        for image_path in self.image_paths:
            results = self.model.predict(image_path, imgsz=640)
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            labels = []
            for box in results[0].boxes:
                label = results[0].names[int(box.cls)]
                bbox = [int(coord) for coord in box.xyxy[0].tolist()]
                conf = float(box.conf)  # Get the confidence score
                labels.append((label, bbox, conf))  # Store label, bbox, and confidence
                draw.rectangle(bbox, outline="red", width=3)

            annotated_path = self.annotated_folder / f"annotated_{Path(image_path).stem}.jpg"
            img.save(annotated_path)
            self.annotated_images.append(annotated_path)
            self.labels_data.append((image_path, labels))
            save_labels_to_xml(image_path, [(label, bbox) for label, bbox, conf in labels], self.label_folder)

        self.display_image_with_labels(0)
        self.save_button.config(state=tk.NORMAL)
        self.process_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.NORMAL)
        self.prev_button.config(state=tk.NORMAL)

    def save_results(self):
        try:
            output_folder = filedialog.askdirectory(title='Select Output Folder')
            if not output_folder:
                return
            output_folder = Path(output_folder)
            for img_path in self.annotated_images:
                shutil.copy(img_path, output_folder)
            for image_path, labels in self.labels_data:
                save_labels_to_xml(image_path, [(label, bbox) for label, bbox, conf in labels], output_folder)
            messagebox.showinfo("Success", "Results saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving results: {e}")

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_image_with_labels(self.current_image_index)

    def next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.display_image_with_labels(self.current_image_index)

    def highlight_bbox(self, event):
        selected_index = self.labels_listbox.curselection()
        if not selected_index:
            return

        # Get the bbox from the selected label (ignore the confidence score here)
        label, bbox, conf = self.labels_data[self.current_image_index][1][selected_index[0]]
        self.display_image_with_labels(self.current_image_index, highlight=bbox)

    def display_image_with_labels(self, index, highlight=None):
        image_path = self.image_paths[index]
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        self.labels_listbox.delete(0, tk.END)
        for label, bbox, conf in self.labels_data[index][1]:  # Unpack label, bbox, and conf
            self.labels_listbox.insert(tk.END, f"{label} ({conf:.2f})")  # Display label with confidence
            draw.rectangle(bbox, outline="red", width=3)

        if highlight:
            # Find the confidence score for the highlighted bounding box
            for label, bbox, conf in self.labels_data[index][1]:
                if bbox == highlight:
                    draw.rectangle(highlight, outline="blue", width=3)
                    font = ImageFont.load_default()
                    # Display the confidence score instead of "something123"
                    draw.text((highlight[0], highlight[1]), f"{conf:.2f}", font_size=100)
                    break

        img.thumbnail((800, 600))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk
        self.info_label.config(text=f"Displaying: {os.path.basename(image_path)}")
        self.current_image_index = index
        self.save_annotated_button.config(state=tk.NORMAL)

    def save_annotated_image(self):
        """Save the currently displayed annotated image."""
        try:
            if not self.annotated_images:
                messagebox.showerror("Error", "No annotated images available to save.")
                return

            current_image_path = self.annotated_images[self.current_image_index]
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])

            if save_path:
                shutil.copy(current_image_path, save_path)
                messagebox.showinfo("Success", "Annotated image saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving annotated image: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOTkApp(root)
    root.mainloop()