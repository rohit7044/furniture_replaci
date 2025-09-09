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

def convertBGR2RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def simplify_hull_to_corners(hull, target_points=4, epsilon_factor=0.01):

    peri = cv2.arcLength(hull, True)
    epsilon = epsilon_factor * peri
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # Increase epsilon until we get <= target_points
    while len(approx) > target_points:
        epsilon *= 1.1
        approx = cv2.approxPolyDP(hull, epsilon, True)

    return approx.reshape(-1, 2).astype(np.float32)

def get_wall_corners(seg_masks, orig_image=None, visualize=True):

    seg_masks = seg_masks.cpu().numpy().squeeze()
    if seg_masks.ndim == 3:
        mask = seg_masks[0]
    mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)

    # Convex hull
    hull = cv2.convexHull(cnt)

    # Simplify to ~4 corners
    corners = simplify_hull_to_corners(hull, target_points=4, epsilon_factor=0.01)

    # If we didn't get exactly 4, fall back to minAreaRect
    if len(corners) != 4:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        corners = np.array(box, dtype=np.float32)

    # Order them consistently (tl, tr, br, bl)
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = corners[np.argmin(s)]     # top-left
    ordered[2] = corners[np.argmax(s)]     # bottom-right
    ordered[1] = corners[np.argmin(diff)]  # top-right
    ordered[3] = corners[np.argmax(diff)]  # bottom-left

    # Visualization
    if visualize and orig_image is not None:
        vis = orig_image.copy()
        cv2.drawContours(vis, [hull], -1, (0, 255, 0), 2)  # hull
        for i, (x, y) in enumerate(ordered.astype(int)):
            cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(vis, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,0,0), 2)
        cv2.imshow("Simplified Hull Corners", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Crop the image based on the corner points
    

    return ordered

def shrink_quad_toward_centroid(corners, scale=0.45):
    """
    corners: (4,2) ordered [tl,tr,br,bl] (float32)
    scale: in (0,1). 1.0 -> original quad, 0.0 -> centroid.
    Returns smaller quad centered inside original, same ordering.
    """
    corners = np.array(corners, dtype=np.float32)
    centroid = corners.mean(axis=0)
    shrunk = centroid + (corners - centroid) * scale
    return shrunk.astype(np.float32)

def center_rect_from_minarea(corners, tv_aspect=None, scale=0.45):
    """
    Build a centered oriented rectangle from minAreaRect of corners,
    then shrink it by `scale`. Returns 4 pts ordered tl,tr,br,bl.
    Use if you want TV to be perfectly rectangular and aligned with wall tilt.
    """
    rect = cv2.minAreaRect(corners.astype(np.float32))
    box = cv2.boxPoints(rect)  # 4x2
    box = np.array(box, dtype=np.float32)

    # get width/height/center from rect
    (cx, cy), (w_rect, h_rect), angle = rect
    if w_rect <= 0 or h_rect <= 0:
        return box.astype(np.float32)

    # determine TV aspect
    if tv_aspect is None:
        # fallback: use box ratio
        tv_aspect = (w_rect / h_rect) if h_rect != 0 else 1.0

    # choose smaller dims by scale while keeping aspect
    target_w = scale * w_rect
    target_h = target_w / tv_aspect

    # create local axes from box points
    box_pts = box.reshape(4,2)
    # take width direction as vector from box[0] -> box[1]
    u = (box_pts[1] - box_pts[0])
    u = u / (np.linalg.norm(u) + 1e-8)
    v = (box_pts[3] - box_pts[0])
    v = v / (np.linalg.norm(v) + 1e-8)

    half_w = target_w / 2.0
    half_h = target_h / 2.0
    center = np.array([cx, cy], dtype=np.float32)

    tl = center - u*half_w - v*half_h
    tr = center + u*half_w - v*half_h
    br = center + u*half_w + v*half_h
    bl = center - u*half_w + v*half_h

    dst = np.vstack([tl, tr, br, bl]).astype(np.float32)
    return dst

def order_points_clockwise(pts):
    """
    Order 4 points as [tl, tr, br, bl] (float32).
    """
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def overlay_tv_fixed_size_on_wall(wall_img, tv_img, wall_w_m, wall_h_m, tv_w_m, tv_h_m, corner_points):
    """
    Place TV of fixed pixel size (from real-world dims) centered/oriented on wall.
    Ensures proper axis ordering so TV stays horizontal (width along u).
    """
    wall_h, wall_w = wall_img.shape[:2]

    # 1) compute tv pixel size from real dimensions and wall mapping
    tv_w_px, tv_h_px = map_real_to_pixels(tv_w_m, tv_h_m, wall_w_m, wall_h_m, wall_img)
    if tv_w_px <= 0 or tv_h_px <= 0:
        raise ValueError("Computed TV pixel dimensions are invalid")

    # 2) oriented rectangle for wall using minAreaRect
    corners = np.array(corner_points, dtype=np.float32)
    rect = cv2.minAreaRect(corners)           # ((cx,cy),(w_rect,h_rect),angle)
    (cx, cy), (w_rect, h_rect), angle = rect

    # fallback to axis bbox if weird
    w_rect = float(abs(w_rect))
    h_rect = float(abs(h_rect))
    if w_rect < 1 or h_rect < 1:
        x_min, y_min = int(np.min(corners[:,0])), int(np.min(corners[:,1]))
        x_max, y_max = int(np.max(corners[:,0])), int(np.max(corners[:,1]))
        w_rect = x_max - x_min
        h_rect = y_max - y_min
        cx = x_min + w_rect/2.0
        cy = y_min + h_rect/2.0

    # 3) scale down if needed to fit inside wall oriented rect
    fit_scale_w = w_rect / tv_w_px
    fit_scale_h = h_rect / tv_h_px
    fit_scale = min(fit_scale_w, fit_scale_h, 1.0) * 0.95  # a tiny margin

    use_w = tv_w_px * fit_scale
    use_h = tv_h_px * fit_scale

    # 4) compute ordered box points for the wall rect to derive axes
    box = cv2.boxPoints(rect).astype(np.float32)  # shape (4,2), arbitrary order
    box_ord = order_points_clockwise(box)         # now [tl, tr, br, bl]

    tl, tr, br, bl = box_ord

    # u: width direction (top edge), v: height direction (left edge)
    u_vec = tr - tl
    v_vec = bl - tl
    u_norm = u_vec / (np.linalg.norm(u_vec) + 1e-8)
    v_norm = v_vec / (np.linalg.norm(v_vec) + 1e-8)

    # 5) construct destination rectangle centered at (cx,cy) with oriented axes
    center = np.array([cx, cy], dtype=np.float32)
    half_w = use_w / 2.0
    half_h = use_h / 2.0

    dst_tl = center - u_norm * half_w - v_norm * half_h
    dst_tr = center + u_norm * half_w - v_norm * half_h
    dst_br = center + u_norm * half_w + v_norm * half_h
    dst_bl = center - u_norm * half_w + v_norm * half_h

    dst_pts = np.vstack([dst_tl, dst_tr, dst_br, dst_bl]).astype(np.float32)

    # Clamp to image bounds
    dst_pts[:,0] = np.clip(dst_pts[:,0], 0, wall_w-1)
    dst_pts[:,1] = np.clip(dst_pts[:,1], 0, wall_h-1)

    # 6) prepare TV image resized to target (width, height)  (cv2.resize expects (w,h))
    target_w_int = max(1, int(round(use_w)))
    target_h_int = max(1, int(round(use_h)))

    # Important: cv2.resize uses (width, height) as (cols,rows) = (target_w, target_h)
    tv_resized = cv2.resize(tv_img, (target_w_int, target_h_int), interpolation=cv2.INTER_AREA)

    # 7) compute homography and warp
    sh, sw = tv_resized.shape[0], tv_resized.shape[1]
    src_pts = np.float32([[0,0],[sw-1,0],[sw-1,sh-1],[0,sh-1]])

    H, _ = cv2.findHomography(src_pts, dst_pts)
    warped_tv = cv2.warpPerspective(tv_resized, H, (wall_w, wall_h), flags=cv2.INTER_LINEAR)

    # 8) composite
    gray = cv2.cvtColor(warped_tv, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    # mask_inv = cv2.bitwise_not(mask)

    # wall_bg = cv2.bitwise_and(orig_wall_img, orig_wall_img, mask=mask_inv)
    # final = cv2.add(wall_bg, warped_tv)

    return warped_tv, (mask > 0).astype(np.uint8)

def paste_warped_into_original(orig_img, warped_tv, mask, bbox):
    """
    Paste warped_tv (Hc,Wc,3) into orig_img at bbox = (x1,y1,x2,y2) using binary mask (Hc,Wc).
    Returns a new image (copy).
    """
    x1, y1, x2, y2 = map(int, bbox)
    h_roi = y2 - y1
    w_roi = x2 - x1

    # Ensure shapes match
    if warped_tv.shape[0] != h_roi or warped_tv.shape[1] != w_roi:
        # If warped_tv is same size as crop, this should be equal.
        # If not, resize warped_tv and mask to ROI shape:
        warped_tv = cv2.resize(warped_tv, (w_roi, h_roi), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.uint8)*255, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)

    # Prepare uint8 mask 0/255
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    mask_inv = cv2.bitwise_not(mask_u8)

    # ROI from original
    roi = orig_img[y1:y2, x1:x2].copy()

    # Background where TV will go
    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Foreground: warped_tv masked
    fg = cv2.bitwise_and(warped_tv, warped_tv, mask=mask_u8)

    # Composite
    comp = cv2.add(bg, fg)

    # Paste back
    out = orig_img.copy()
    out[y1:y2, x1:x2] = comp

    return out


# Main execution
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

wall_corners = get_wall_corners(segmentation_results, d_cropped_img, visualize=False)
tv_img = convertBGR2RGB(np.array(tv_img))


warped_tv, mask = overlay_tv_fixed_size_on_wall(
    d_cropped_img, tv_img, r_wall_width, r_wall_height, r_tv_width, r_tv_height, wall_corners
)


orig_with_tv = paste_warped_into_original(wall_img, warped_tv, mask, detection_results[0])

cv2.imshow("Final on original", orig_with_tv)
cv2.waitKey(0)
cv2.destroyAllWindows()