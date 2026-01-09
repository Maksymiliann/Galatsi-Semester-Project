# import cv2
# from patched_yolo_infer import MakeCropsDetectThem, CombineDetections

# # Load the image 
# img_path = "first_frame_static.png"
# img = cv2.imread(img_path)

# element_crops = MakeCropsDetectThem(
#     image=img,
#     model_path="yolo11x.pt",
#     segment=False,
#     shape_x=640,
#     shape_y=640,
#     overlap_x=25,
#     overlap_y=25,
#     conf=0.5,
#     iou=0.7,
# )
# result = CombineDetections(element_crops, nms_threshold=0.25)  

# # Final Results:
# img=result.image
# confidences=result.filtered_confidences
# boxes=result.filtered_boxes
# polygons=result.filtered_polygons
# classes_ids=result.filtered_classes_id
# classes_names=result.filtered_classes_names

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

from patched_yolo_infer import (
    MakeCropsDetectThem,
    CombineDetections,
    visualize_results,
)

"""
Patch-based YOLOv8 segmentation with detection merging.

- Load an image and a YOLOv8 segmentation model.
- Run two inference passes:
  1) Tiled inference (small overlapping crops, low conf) to catch small/edge objects.
  2) Full-image inference (single crop, higher conf) for strong global detections.
- Merge both results with NMS using CombineDetections.
- Visualize the final merged segmentation masks (optionally without boxes/labels).

Goal: improve segmentation robustness on difficult images by combining local + global passes.
"""


# Load the image
img_path = '/content/difficult.jpg'
img = cv2.imread(img_path)

plt.imshow(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB))
plt.show()


model = YOLO("yolov8m-seg.pt")

element_crops_1 = MakeCropsDetectThem(
        image=img,
        model=model,
        segment=True,
        show_crops=True,
        shape_x=450,
        shape_y=325,
        overlap_x=20,
        overlap_y=45,
        conf=0.2,
        iou=0.75,
        classes_list=[0, 1, 2, 3, 5, 7],
        memory_optimize=False,
        inference_extra_args={'retina_masks': True}
    )
element_crops_2 = MakeCropsDetectThem(
        image=img,
        model=model,
        segment=True,
        show_crops=True,
        shape_x=img.shape[1],
        shape_y=img.shape[0],
        overlap_x=0,
        overlap_y=0,
        conf=0.7,
        iou=0.5,
        classes_list=[0, 1, 2, 3, 5, 7],
        memory_optimize=False,
        inference_extra_args={'retina_masks': True}
    )

result = CombineDetections([element_crops_2, element_crops_1], nms_threshold=0.5, sorter_bins=3)


visualize_results(
    img=result.image,
    confidences=result.filtered_confidences,
    boxes=result.filtered_boxes,
    masks=result.filtered_masks,
    classes_ids=result.filtered_classes_id,
    classes_names=result.filtered_classes_names,
    thickness=5,
    show_boxes=False,
    show_class=False,
    segment=True,
    fill_mask=True,
)