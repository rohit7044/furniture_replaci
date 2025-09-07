import random
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Processor, Sam2Model, AutoProcessor, AutoModelForZeroShotObjectDetection

#CONSTANTS
WALL_IMAGE_PATH = "Test_photos/angled_wall.jpg"
TV_IMAGE_PATH = "Test_photos/TV.jpg"
detector_id = "IDEA-Research/grounding-dino-tiny"
segmenter_id = "facebook/sam2.1-hiera-tiny"

device = "cuda"

def read_img(image_path):
    img = Image.open(image_path).convert("RGB")
    return img

def detection(image, text_prompt):
    detect_processor = AutoProcessor.from_pretrained(detector_id)
    detect_model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(device)
    
    
    text_labels = [[text_prompt]]
    inputs = detect_processor(images=image, text=text_labels, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = detect_model(**inputs)

    results = detect_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.3,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    result = results[0]["boxes"].cpu().numpy()
    # print(result)
    return result

def segmentation(image):
    seg_model = Sam2Model.from_pretrained(segmenter_id).to(device)
    seg_processor = Sam2Processor.from_pretrained(segmenter_id)

    inputs = seg_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = seg_model(**inputs)

    masks = seg_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

    return masks

def visualize_masks(image, masks):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Make a copy to draw masks
    overlay = image.copy()
    print(f"Number of masks: {len(masks)}")

    # Iterate through masks
    for i, mask in enumerate(masks):
        mask = mask[0].numpy().astype(np.uint8)  # (H, W), values 0/1

        # Create a colored mask (red in this case)
        color = (0, 0, 255)  # BGR -> Red
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[mask == 1] = color

        # Blend with the original image
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

    # Show images
    # cv2.imshow("Original Image", image)
    cv2.imshow("Segmentation Masks Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawbbox(image_path, boxes, labels=None):
    image = cv2.imread(image_path)

    # Define a list of distinct colors (BGR format for OpenCV)
    colors = [
        (255, 0, 0),     # Blue
        (0, 255, 0),     # Green
        (0, 0, 255),     # Red
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Yellow
        (128, 0, 128),   # Purple
        (255, 165, 0),   # Orange
        (0, 128, 128),   # Teal
        (128, 128, 0),   # Olive
    ]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        color = colors[i % len(colors)]  # cycle colors if boxes > colors
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label if provided
        if labels is not None:
            label = str(labels[i])
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def crop_bbox(img, bbox):
    img = np.array(img)
    x1, y1, x2, y2 = map(int, bbox)
    cropped_img = img[y1:y2, x1:x2]
    return cropped_img

def map_real_to_pixels(obj_w_m, obj_h_m, wall_w_m, wall_h_m, wall_img):
    """Convert real-world object dimensions to pixel dimensions in the wall image."""
    img_h, img_w = wall_img.shape[:2]

    px_per_m_w = img_w / wall_w_m
    px_per_m_h = img_h / wall_h_m

    obj_w_px = int(obj_w_m * px_per_m_w)
    obj_h_px = int(obj_h_m * px_per_m_h)

    return obj_w_px, obj_h_px

def overlay_image(wall_img, tv_img, wall_w_m, wall_h_m, tv_w_m, tv_h_m, center=True):
    # Get TV size in pixels
    tv_w_px, tv_h_px = map_real_to_pixels(tv_w_m, tv_h_m, wall_w_m, wall_h_m, wall_img)

    # Resize TV
    tv_resized = cv2.resize(tv_img, (tv_w_px, tv_h_px), interpolation=cv2.INTER_AREA)

    # Positioning
    wall_h, wall_w = wall_img.shape[:2]
    if center:
        x_offset = (wall_w - tv_w_px) // 2
        y_offset = (wall_h - tv_h_px) // 2
    else:
        x_offset, y_offset = 50, 50  # default position

    # Place TV directly (no blending since no alpha channel)
    wall_img[y_offset:y_offset+tv_h_px, x_offset:x_offset+tv_w_px] = tv_resized

    return wall_img

def overlay_tv_perspective(wall_img, tv_img, wall_w_m, wall_h_m, tv_w_m, tv_h_m):

    wall_h, wall_w = wall_img.shape[:2]

    tv_w_px, tv_h_px = map_real_to_pixels(tv_w_m, tv_h_m, wall_w_m, wall_h_m, wall_img)
    tv_resized = cv2.resize(tv_img, (tv_w_px, tv_h_px), interpolation=cv2.INTER_AREA)
    

    src_pts = np.float32([[0, 0], [tv_w_px, 0], [tv_w_px, tv_h_px], [0, tv_h_px]])

    cx, cy = wall_w // 2, wall_h // 2
    half_w, half_h = tv_w_px // 2, tv_h_px // 2

    dst_pts = np.float32([
        [cx - half_w, cy - half_h],  # top-left
        [cx + half_w, cy - half_h],  # top-right
        [cx + half_w, cy + half_h],  # bottom-right
        [cx - half_w, cy + half_h]   # bottom-left
    ])

    H, _ = cv2.findHomography(src_pts, dst_pts)
    warped_tv = cv2.warpPerspective(tv_resized, H, (wall_w, wall_h))

    gray = cv2.cvtColor(warped_tv, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    wall_bg = cv2.bitwise_and(wall_img, wall_img, mask=mask_inv)
    final = cv2.add(wall_bg, warped_tv)

    return final

def convertBGR2RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def findcornerpoints(mask_image, orig_image=None, visualize=True):
    mask = mask_image.cpu().numpy().squeeze()
    if mask.ndim == 3:
        mask = mask[0]
    disp = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(disp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Largest contour
    cnt = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(cnt)    
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    corners = np.array(box, dtype=np.float32)
    print("Detected corners (unsorted):", corners)

    # Visualize the corners on the original image
    vis_image = orig_image.copy()

    # Draw the convex hull contour (optional)
    cv2.drawContours(vis_image, [hull], -1, (0, 255, 0), 2)  # Green contour

    # Draw the corners as red circles
    for (x, y) in corners.astype(int):
        cv2.circle(vis_image, (x, y), 8, (0, 0, 255), -1)  # Red circles
        # Optional: add text labels for each corner
        cv2.putText(vis_image, f"({x},{y})", (int(x)+10, int(y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show the result
    cv2.imshow("Detected Corners", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # return ordered


    

    


text_prompt = "a wall"
wall_img = read_img(WALL_IMAGE_PATH)
tv_img = read_img(TV_IMAGE_PATH)
r_wall_height = 2.6 # in meters
r_wall_width = 3.8 # in meters
r_tv_height = 0.7 # in meters
r_tv_width = 1.2 # in meters

detection_results = detection(wall_img, text_prompt)

d_cropped_img = crop_bbox(wall_img, detection_results[0])
d_cropped_img = convertBGR2RGB(d_cropped_img)

# drawbbox(WALL_IMAGE_PATH, detection_results)

wall_img = convertBGR2RGB(np.array(wall_img))
segmentation_results = segmentation(d_cropped_img)
# visualize_masks(d_cropped_img, segmentation_results)
findcornerpoints(segmentation_results,d_cropped_img)




# tv_img = convertBGR2RGB(np.array(tv_img))

# overlay_img = overlay_image(cropped_img, tv_img, r_wall_width, r_wall_height, r_tv_width, r_tv_height, center=True)

# overlay_img = overlay_tv_perspective(d_cropped_img, tv_img, r_wall_width, r_wall_height, r_tv_width, r_tv_height)


# cv2.imwrite("wall_with_tv.png", overlay_img)
# cv2.imshow("Overlay TV over wall", overlay_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

