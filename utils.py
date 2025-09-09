# utils.py
import cv2
import numpy as np
from PIL import Image

def read_img(image_path):
    """Return PIL RGB image (unchanged from original)."""
    img = Image.open(image_path).convert("RGB")
    return img

def convertBGR2RGB(image):
    """Convert OpenCV BGR numpy image to RGB numpy image."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def drawbbox(image, boxes, labels=None):
    """
    Draw boxes on image_path and show with cv2.imshow.
    Note: in Streamlit environment you will generally use st.image instead.
    """
    colors = [
        (255, 0, 0), (0,255,0), (0,0,255), (255,255,0),
        (255,0,255), (0,255,255), (128,0,128), (255,165,0),
        (0,128,128), (128,128,0),
    ]
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        color = colors[i % len(colors)]
        cv2.rectangle(image, (x1,y1),(x2,y2), color, 2)
        if labels is not None:
            label = str(labels[i])
            cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Detected Objects", image)
    # cv2.waitKey(0); cv2.destroyAllWindows()
    return image  # return image for downstream display

def crop_bbox(img, bbox):
    """Crop a PIL or numpy image by bbox (x1,y1,x2,y2). Returns numpy RGB image."""
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    x1, y1, x2, y2 = map(int, bbox)
    cropped_img = img[y1:y2, x1:x2]
    return cropped_img

def map_real_to_pixels(obj_w_m, obj_h_m, wall_w_m, wall_h_m, wall_img):
    """Convert real-world dimensions (meters) to pixel dims for wall image (numpy array)."""
    img_h, img_w = wall_img.shape[:2]
    px_per_m_w = img_w / wall_w_m
    px_per_m_h = img_h / wall_h_m
    obj_w_px = int(obj_w_m * px_per_m_w)
    obj_h_px = int(obj_h_m * px_per_m_h)
    return obj_w_px, obj_h_px

def simplify_hull_to_corners(hull, target_points=4, epsilon_factor=0.01):
    peri = cv2.arcLength(hull, True)
    epsilon = epsilon_factor * peri
    approx = cv2.approxPolyDP(hull, epsilon, True)
    while len(approx) > target_points:
        epsilon *= 1.1
        approx = cv2.approxPolyDP(hull, epsilon, True)
    return approx.reshape(-1, 2).astype(np.float32)

def order_points_clockwise(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def get_wall_corners(seg_masks, orig_image=None):
    """
    Given seg_masks (as returned by segmentation() -> post_process_masks()[0]),
    extract hull, simplify to corner quad and return ordered corners [tl,tr,br,bl].
    """
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

    # Optional visualization for debugging   
    # vis = orig_image.copy()
    # cv2.drawContours(vis, [hull], -1, (0,255,0), 2)
    # for i, (x, y) in enumerate(ordered.astype(int)):
    #     cv2.circle(vis, (x,y), 6, (0,0,255), -1)
    #     cv2.putText(vis, str(i), (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    # cv2.imshow("Simplified Hull Corners", vis); cv2.waitKey(0); cv2.destroyAllWindows()
    return ordered

def shrink_quad_toward_centroid(corners, scale=0.45):
    corners = np.array(corners, dtype=np.float32)
    centroid = corners.mean(axis=0)
    shrunk = centroid + (corners - centroid) * scale
    return shrunk.astype(np.float32)

def center_rect_from_minarea(corners, tv_aspect=None, scale=0.45):
    rect = cv2.minAreaRect(corners.astype(np.float32))
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)
    (cx, cy), (w_rect, h_rect), angle = rect
    if w_rect <= 0 or h_rect <= 0:
        return box.astype(np.float32)
    if tv_aspect is None:
        tv_aspect = (w_rect / h_rect) if h_rect != 0 else 1.0
    target_w = scale * w_rect
    target_h = target_w / tv_aspect
    box_pts = box.reshape(4,2)
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

def overlay_tv_fixed_size_on_wall(wall_img, tv_img, wall_w_m, wall_h_m, tv_w_m, tv_h_m, corner_points):
    """
    Place TV of fixed pixel size (from real-world dims) centered/oriented on wall.
    Returns warped_tv and binary mask as (H,W,3) and (H,W) respectively.
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
    mask_inv = cv2.bitwise_not(mask)

    wall_bg = cv2.bitwise_and(wall_img, wall_img, mask=mask_inv)
    croppedwall_withTV = cv2.add(wall_bg, warped_tv)

    return warped_tv,croppedwall_withTV, (mask > 0).astype(np.uint8)

def paste_warped_into_original(orig_img, warped_tv, mask, bbox):
    """
    Paste warped_tv into orig_img at bbox using binary mask.
    bbox: (x1,y1,x2,y2) in ints.
    Returns composited image (numpy array).
    """
    x1, y1, x2, y2 = map(int, bbox)
    h_roi = y2 - y1
    w_roi = x2 - x1
    if warped_tv.shape[0] != h_roi or warped_tv.shape[1] != w_roi:
        warped_tv = cv2.resize(warped_tv, (w_roi, h_roi), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.uint8)*255, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    mask_inv = cv2.bitwise_not(mask_u8)
    roi = orig_img[y1:y2, x1:x2].copy()
    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg = cv2.bitwise_and(warped_tv, warped_tv, mask=mask_u8)
    comp = cv2.add(bg, fg)
    out = orig_img.copy()
    out[y1:y2, x1:x2] = comp
    return out

def visualize_masks(image, masks):
    """
    Blend and return a visualization numpy image with masks overlaid (red).
    Does not show windows (useful for Streamlit).
    """
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
    # cv2.imshow("Segmentation Masks Overlay", overlay)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return overlay
