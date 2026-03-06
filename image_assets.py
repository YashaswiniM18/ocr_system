import cv2
import os
import numpy as np
from typing import Optional, Tuple
try:
    import pytesseract
except ImportError:
    pytesseract = None



from image_preprocessing import load_image

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

import threading
_FACE_LOCK = threading.Lock()
_PAN_ROT_CACHE = {}
_PAN_ROT_LOCK = threading.Lock()

from ocr_engine import get_ocr


# ---------------- COMMON FACE DETECTOR ---------------- #

def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _expand_and_square_bbox(x, y, w, h, img_w, img_h):
    # Add tighter padding to focus on just the passport photo.

    # Strict Padding for "Exact Photo" crop
    pad_left = int(0.25 * w)
    pad_right = int(0.25 * w)
    pad_top = int(0.55 * h)
    # Aggressively reduced bottom padding
    pad_bottom = int(0.25 * h)

    x1 = _clamp(x - pad_left, 0, img_w - 1)
    y1 = _clamp(y - pad_top, 0, img_h - 1)
    x2 = _clamp(x + w + pad_right, 0, img_w - 1)
    y2 = _clamp(y + h + pad_bottom, 0, img_h - 1)

    # Adjust to a 3:4 (w:h) passport-ish ratio by expanding within bounds.
    box_w = x2 - x1
    box_h = y2 - y1
    target_ratio = 3 / 4  # width / height

    if box_w / box_h > target_ratio:
        # Too wide: expand height
        target_h = int(box_w / target_ratio)
        extra = target_h - box_h
        y1 = _clamp(y1 - extra // 2, 0, img_h - 1)
        y2 = _clamp(y2 + extra - extra // 2, 0, img_h - 1)
    else:
        # Too tall: expand width
        target_w = int(box_h * target_ratio)
        extra = target_w - box_w
        x1 = _clamp(x1 - extra // 2, 0, img_w - 1)
        x2 = _clamp(x2 + extra - extra // 2, 0, img_w - 1)

    return x1, y1, x2, y2


def _trim_white_border(img, threshold=235, min_border=4):
    # Trim uniform white-ish borders around the cropped photo.
    # If no strong border detected, return original.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < threshold

    coords = cv2.findNonZero(mask.astype("uint8"))
    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)
    # Avoid over-trimming tiny margins.
    if x <= min_border and y <= min_border:
        return img[y:y + h, x:x + w]
    return img[y:y + h, x:x + w]


def _tight_crop_by_nonwhite(img, threshold=228, pad=4):
    # Aggressively remove white/near-white background.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < threshold
    coords = cv2.findNonZero(mask.astype("uint8"))
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    x1 = _clamp(x - pad, 0, img.shape[1] - 1)
    y1 = _clamp(y - pad, 0, img.shape[0] - 1)
    x2 = _clamp(x + w + pad, 0, img.shape[1] - 1)
    y2 = _clamp(y + h + pad, 0, img.shape[0] - 1)
    return img[y1:y2, x1:x2]


def _trim_uniform_top_border(img, threshold=228, max_scan=0.40, pad=2):
    # Trim uniform light top band (common in PAN photo).
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scan_h = int(max_scan * h)
    top = 0
    # Move down while row is mostly light.
    for y in range(scan_h):
        row = gray[y]
        light_ratio = (row > threshold).sum() / w
        if light_ratio < 0.82:
            break
        top = y + 1
    if top == 0:
        return img
    top = _clamp(top - pad, 0, h - 1)
    return img[top:h, 0:w]


def _trim_top_if_light(img, fraction=0.06, threshold=225):
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cut = int(fraction * h)
    if cut <= 0:
        return img
    top_mean = gray[:cut].mean()
    if top_mean >= threshold:
        return img[cut:h, 0:w]
    return img


def _rotate_image(img, angle):
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _haar_detect_bbox(img, scale_factor=1.1, min_neighbors=5, min_size=(40, 40)) -> Optional[Tuple[int, int, int, int, float]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    with _FACE_LOCK:
        # Use detectMultiScale3 to get confidence weights
        faces, rejectLevels, levelWeights = FACE_CASCADE.detectMultiScale3(
            gray, scale_factor, min_neighbors, minSize=min_size, outputRejectLevels=True
        )
    
    if len(faces) == 0:
        return None
        
    # Zip faces with their weights
    weighted_faces = []
    for (x, y, w, h), weight in zip(faces, levelWeights):
        # detection weight is usually in index 0 of the weight array or just a float
        w_val = weight[0] if isinstance(weight, (list, np.ndarray)) else weight
        weighted_faces.append(((x, y, w, h), w_val))
    
    # Sort by weight (descending) -> Best face first
    weighted_faces.sort(key=lambda x: x[1], reverse=True)
    
    # Return (x,y,w,h, weight) of the best face
    best_face, best_weight = weighted_faces[0]
    return (*best_face, best_weight)


def _find_photo_rect(img, face_bbox):
    # Try to detect the rectangular photo region that contains the face.
    if face_bbox is None:
        return None

    fx, fy, fw, fh = face_bbox
    fh_img, fw_img, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Relaxed Canny thresholds to detect faint photo edges
    edges = cv2.Canny(blur, 30, 100)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_rect = None
    best_score = 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if w <= 0 or h <= 0:
            continue

        # Must contain the face bbox.
        if x > fx or y > fy or x + w < fx + fw or y + h < fy + fh:
            continue

        area = w * h
        if area < 0.03 * fw_img * fh_img or area > 0.60 * fw_img * fh_img:
            continue

        ratio = w / h
        if ratio < 0.60 or ratio > 1.0:
            continue

        # Prefer rects with face near center.
        cx = x + w / 2
        cy = y + h / 2
        fcx = fx + fw / 2
        fcy = fy + fh / 2
        center_dist = abs(cx - fcx) + abs(cy - fcy)
        score = area - center_dist * 4

        if score > best_score:
            best_score = score
            best_rect = (x, y, x + w, y + h)

    return best_rect


def _crop_face_passport(img, pad_top_ratio=0.55, **kwargs):
    result = _haar_detect_bbox(img, **kwargs)
    if result is None:
        return None, 0.0

    x, y, w, h, weight = result
    
    # 1. OPTION A: Try to find the EXACT photo rectangle (edges of the printed photo).
    # This is the "accurate" way to get exactly what's on the card.
    # Relaxed params to find distinct photo box more easily.
    photo_rect = _find_photo_rect(img, (x, y, w, h))
    if photo_rect is not None:
        px1, py1, px2, py2 = photo_rect
        face = img[py1:py2, px1:px2]
        
        # Even if we found a "photo rect", it might include a logo header if the logic was fooled.
        # Apply the detached header trim here too!
        # Face Y relative to this crop:
        face_y_in_rect = y - py1
        face = _trim_disconnected_header(face, face_y_in_rect, h)
        
        # Optional: Smart trim might be risky here if exact rect is perfect, 
        # but let's do it gently to clean up edges? 
        # Actually, let's just stick to header trim for now.
        return face, weight

    # 2. OPTION B: Fallback to Geometric Expansion (Smart Crop)
    # If we can't find the photo edges, we estimate based on face position.
    
    img_h, img_w, _ = img.shape
    
    # NEW STRATEGY: 
    # 1. Take a slightly larger "Search Crop" (0.40 padding) to ensure we get the full photo box.
    # 2. Run Contour Detection on this crop to find the EXACT photo rectangle.
    # 3. If found, use it (Perfect Crop).
    # 4. If NOT found, revert to the "Strict" padding (0.25) as fallback.
    
    # Step 1: Search Crop (0.35 padding)
    s_pad_left = int(0.35 * w)
    s_pad_right = int(0.35 * w)
    s_pad_top = int(pad_top_ratio * h)
    s_pad_bottom = int(0.50 * h)
    
    sx1 = _clamp(x - s_pad_left, 0, img_w - 1)
    sy1 = _clamp(y - s_pad_top, 0, img_h - 1)
    sx2 = _clamp(x + w + s_pad_right, 0, img_w - 1)
    sy2 = _clamp(y + h + s_pad_bottom, 0, img_h - 1)
    
    search_crop = img[sy1:sy2, sx1:sx2]
    
    # Face relative to search crop for validation
    face_rect_in_search = (x - sx1, y - sy1, w, h)
    
    # Step 2: Try Perfect Contour Detection
    perfect_crop = _refine_crop_by_contour(search_crop, face_rect_in_search)
    
    if perfect_crop is not None:
        return perfect_crop, weight
        
    # Step 3: Fallback to Strict Padding (0.25)
    # Re-calculate strict coordinates from original image
    # Strict Padding for "Exact Photo" crop
    pad_left = int(0.25 * w)
    pad_right = int(0.25 * w)
    pad_top = int(pad_top_ratio * h)
    # Aggressively reduced bottom padding
    pad_bottom = int(0.25 * h)

    x1 = _clamp(x - pad_left, 0, img_w - 1)
    y1 = _clamp(y - pad_top, 0, img_h - 1)
    x2 = _clamp(x + w + pad_right, 0, img_w - 1)
    y2 = _clamp(y + h + pad_bottom, 0, img_h - 1)
    
    face = img[y1:y2, x1:x2]
    
    # Calculate where the original Face Box sits inside this new Crop
    # Original face y is at `y`. Crop starts at `y1`.
    # So inside 'face', the Haar box starts at `y - y1`.
    face_y_in_crop = y - y1
    
    # 0. Clean top background (remove white space above head)
    face = _trim_top_bg(face)
    
    # Re-calculate offsets if size changed? _trim_top_bg changes the image content/size.
    # It might be safer to do this last or re-detect. 
    # But let's just do it. The subsequent steps rely on content, not absolute coordinates.
    
    # 1. Remove detached headers (logos above head)
    face = _trim_disconnected_header(face, face_y_in_crop, h)
    
    # 2. Trim borders using smart detection
    # REMOVED: Smart trim was removing the padding found/added by _expand_and_square_bbox,
    # causing the "zoomed in" effect the user disliked.
    pass # face = _smart_trim_to_content(face)
    
    # 3. Explicitly check for text footer (e.g. "Name") at the bottom
    face = _trim_bottom_text(face)
    
    # 4. Trim sides to remove background artifacts
    face = _trim_sides(face)

    return face, weight


def _trim_disconnected_header(img, face_y, face_h):
    # Scanning UPWARDS from face to find floating header.
    # Uses Row Variance to distinguish detailed content (Text/Hair) from flat background.
    if img is None or img.size == 0:
        return img
        
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Start Scan from clearly inside the forehead area
    # Start at 20% down the face (eyebrows/forehead).
    start_y = int(face_y + 0.20 * face_h)
    start_y = _clamp(start_y, 0, h - 1)
    
    # Config
    # Variance Threshold: Text/Features have high variance. Flat background has low.
    # We use standard deviation here.
    # A flat scan usually has std < 5.0. Text has std > 10.0.
    # We'll use a conservative threshold of 7.0.
    std_threshold = 7.0 
    min_gap_px = int(0.02 * face_h)
    
    # STATE MACHINE
    # 0 = On Face (expect high variance)
    # 1 = In Gap (expect low variance)
    # 2 = Hit Header (high variance after gap)
    state = 0 
    
    current_gap_height = 0
    
    for y in range(start_y, -1, -1):
        row = gray[y]
        row_std = np.std(row)
        
        is_flat = row_std < std_threshold
        
        if state == 0: # On Face
            if is_flat:
                # Potential start of gap
                state = 1
                current_gap_height = 1
        
        elif state == 1: # In Gap
            if is_flat:
                current_gap_height += 1
            else:
                # Hit something detailed (High Variance)
                # Is the gap big enough?
                if current_gap_height >= min_gap_px:
                     # YES! We crossed a flat gap and hit detailed text/logo.
                     # Cut at the BOTTOM of the gap (y + gap_size).
                     # Actually, users usually prefer keeping the gap if it's clean.
                     # But current user complaint is "same image", implying tight crop is expected?
                     # Let's cut at y + 2 (just inside the gap).
                     cut_y = y + 2
                     return img[cut_y:h, 0:w]
                else:
                    # No, gap was too small. Reset.
                    state = 0
                    current_gap_height = 0
            
    return img


def _trim_bottom_text(img):
    # Scan UPWARDS from bottom to find detailed text (Name label etc)
    # and crop above it.
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Only scan the bottom 25%
    scan_start = h - 1
    scan_end = int(0.75 * h)
    
    # "Text" usually has high variance. Background is flat.
    std_threshold = 8.0
    
    # State:
    # 0: In potential footer (high variance text)
    # 1: In gap (low variance)
    # If we find a gap ABOVE high variance stuff, we cut there.
    
    # Heuristic: Find first "flat" row scanning from bottom.
    # The moment we hit a flat row after seeing high variance, that's our cut.
    
    seen_high_variance = False
    cut_y = h # Default: no cut
    
    for y in range(scan_start, scan_end, -1):
        row = gray[y]
        row_std = np.std(row)
        
        if row_std > std_threshold:
            seen_high_variance = True
        else:
            # Low variance (flat)
            if seen_high_variance:
                # We saw noise/text at bottom, now we hit a flat gap.
                # Use this as cut point.
                cut_y = y
                break
                
    if cut_y < h:
        # Buffer: move cut down a few pixels to include the "gap"
        cut_y = min(cut_y + 2, h)
        return img[0:cut_y, 0:w]
        
    return img


def _trim_top_bg(img):
    # Trim uniform top background
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Scan down to find first "content"
    # Content = high variance OR significantly darker than background
    
    # Assume top corner is background
    bg_sample = gray[0:5, 0:w].mean()
    
    cut_y = 0
    scan_h = int(0.20 * h)
    
    for y in range(scan_h):
        row = gray[y]
        row_mean = row.mean()
        row_std = np.std(row)
        
        # If row is very different (darker) or has texture
        if abs(row_mean - bg_sample) > 20.0 or row_std > 8.0:
            cut_y = y
            break
            
    if cut_y > 0:
        return img[cut_y:h, 0:w]
    return img


def _refine_crop_by_contour(img, face_rect):
    # Try to find the single largest rectangle that:
    # 1. Contains the face center
    # 2. Has a passport-like aspect ratio (0.7 - 0.9 range, maybe up to 1.0)
    # 3. Covers a significant portion of the image (it shouldn't be tiny)
    
    if img is None or img.size == 0:
        return None
        
    h, w = img.shape[:2]
    fx, fy, fw, fh = face_rect
    fcx = fx + fw / 2
    fcy = fy + fh / 2
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance edges
    # Blur slightly
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny Edge Detection
    edges = cv2.Canny(blur, 30, 100)
    
    # Dilate to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    best_crop = None
    best_score = 0
    
    for cnt in contours:
        # Approximate contour to polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        
        # We look for 4-sided objects (rectangles)
        if len(approx) == 4:
            x, y, cw, ch = cv2.boundingRect(approx)
            
            # Constraints
            
            # 1. Must contain face center
            if not (x < fcx < x + cw and y < fcy < y + ch):
                continue
                
            # 2. Minimum Area constraint
            area = cw * ch
            if area < (fw * fh) * 1.5: # Must be at least 1.5x the face area
                continue
                
            # 3. Aspect Ratio (w/h)
            # Passport photo usually 0.75 - 0.85
            ratio = cw / ch
            if ratio < 0.65 or ratio > 0.95:
                continue
                
            # Score: larger area is better, closer to aspect ratio 0.8 is better
            score = area
            
            if score > best_score:
                best_score = score
                best_crop = img[y:y+ch, x:x+cw]
    
    return best_crop


def _remove_text_lines(img):
    # Remove horizontal lines of text that might be captured (e.g. Name label)
    if img is None or img.size == 0:
        return img
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Morphological op to find horizontal structures (text lines)
    # Use a wide kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    
    # Threshold inverted
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    
    # Detect horizontal lines
    lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours of these lines
    cnts, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts:
        return img
        
    # Create mask to paint over these lines? 
    # Or cleaner: if a line spans most of the width, it's likely text/underline.
    # The signature itself is flowy and won't match a (25,1) rect well usually.
    # But straight signatures might.
    
    # Heuristic: If we find a solid text-like block at the TOP or BOTTOM, crop it out.
    
    # Check Top 20%
    cut_top = 0
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > 0.4 * w: # Long enough to be text line
            if y < 0.2 * h: # At top
                cut_top = max(cut_top, y + ch)
                
    if cut_top > 0:
        img = img[cut_top:h, 0:w]
        
    return img


def _trim_sides(img):
    # Scan INWARDS from sides to find vertical lines or background changes
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Scan logic similar to trim_bottom but for columns
    # We look for high variance (text/lines) or very dark/light vertical bands
    
    std_threshold = 8.0
    scan_w = int(0.20 * w) # Check first/last 20%
    
    # Left Side
    cut_x1 = 0
    for x in range(scan_w):
        col = gray[:, x]
        if np.std(col) < std_threshold:
            # Flat background/border
            cut_x1 = x + 1
        # else: high variance (photo detail starts) -> Stop
    
    # Right Side
    cut_x2 = w
    for x in range(w - 1, w - scan_w, -1):
        col = gray[:, x]
        if np.std(col) < std_threshold:
            cut_x2 = x
        # else: high variance -> Stop
        
    if cut_x1 >= cut_x2:
        return img
        
    return img[:, cut_x1:cut_x2]


def _smart_trim_to_content(img, sensitivity=35):
    # Auto-detect background from corners and trim inwards.
    if img is None or img.size == 0:
        return img
        
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Sample 4 corners to find background intensity
    # (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
    # We take a small 5x5 patch from each corner.
    corners = [
        gray[0:5, 0:5],
        gray[0:5, w-5:w],
        gray[h-5:h, 0:5],
        gray[h-5:h, w-5:w]
    ]
    
    # Calculate median background brightness
    bg_vals = np.concatenate([c.flatten() for c in corners])
    bg_median = np.median(bg_vals)
    
    # scan_limit: don't crop more than 40% from any side to avoid losing face
    limit_h = int(h * 0.40)
    limit_w = int(w * 0.40)
    
    # Helper to find crop edge
    def find_edge(axis_data, limit):
        # axis_data: array of line averages or min/max?
        # Better: check if ROW is "similar" to background.
        # A row is background if its mean is close to bg_median OR 
        # (for dark backgrounds) if it's uniform.
        # Let's use simple diff from median.
        cutoff = 0
        for i in range(limit):
            row = axis_data[i]
            # Check if majority of pixels in this row are close to bg_median
            # OR check row mean.
            diff = abs(row.mean() - bg_median)
            if diff > sensitivity:
                return i
            cutoff = i
        return cutoff

    # Top
    y1 = find_edge(gray, limit_h)
    # Bottom (scan inverse)
    y2 = h - find_edge(gray[::-1], limit_h)
    
    # Left
    x1 = find_edge(gray.T, limit_w)
    # Right
    x2 = w - find_edge(gray.T[::-1], limit_w)
    
    # Safety Check: don't crop to empty
    if x2 <= x1 or y2 <= y1:
        return img
        
    return img[y1:y2, x1:x2]


def _crop_face_best_rotation(img, pad_top_ratio=0.55):
    
    # helper for running detection on all 4 rotations
    def _scan_rotations(image_to_scan, scale_factor, min_neighbors, min_size):
        best_face = None
        best_weight = -1.0
        best_angle = None
        
        for angle in (0, 90, 180, 270):
            rotated = _rotate_image(image_to_scan, angle)
            # Pass strict/relaxed params
            face, weight = _crop_face_passport(
                rotated, 
                pad_top_ratio=pad_top_ratio,
                scale_factor=scale_factor, 
                min_neighbors=min_neighbors, 
                min_size=min_size
            )
            
            if face is None:
                continue

            # BIAS:
            # 1. Penalize 180 (upside down) significantly. It's the most common false positive (flipping upright photos).
            # 2. Slight boost to 0 (upright) to win tie-breaks.
            # 3. Do NOT penalize 90/270 (sideways) because valid scans are often sideways.
            
            if angle == 180:
                weight -= 3.0
            elif angle == 0 and weight > 0:
                weight += 0.5
                
            if weight > best_weight:
                best_weight = weight
                best_face = face
                best_angle = angle
            
            # EARLY EXIT: If we find a very high confidence face, stop scanning other rotations.
            if weight > 40.0:
                break
        
        return best_face, best_angle, best_weight

    # OPTIMIZATION: Use small image for detection to speed up the 4-way rotation scan.
    h_orig, w_orig = img.shape[:2]
    detection_img = img
    scale = 1.0
    MAX_DIM = 800
    
    if max(h_orig, w_orig) > MAX_DIM:
        scale = MAX_DIM / max(h_orig, w_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        detection_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 1. Strict Pass (default params)
    # Note: min_size needs to be scaled down if we scaled the image!
    strict_min_size = (int(80 * scale), int(80 * scale))
    face_small, angle, weight = _scan_rotations(detection_img, 1.1, 5, strict_min_size)
    
    final_face_found = False
    
    if face_small is None:
        # 2. Relaxed Pass (if strict failed)
        relaxed_min_size = (int(40 * scale), int(40 * scale))
        face_small, angle, weight = _scan_rotations(detection_img, 1.05, 3, relaxed_min_size)

    if angle is not None:
        # We found the best angle using the small image.
        # Now apply that SINGLE rotation to the full-res image.
        img_rotated = _rotate_image(img, angle)
        
        # Run detection ONCE on the correctly oriented full-res image to get high quality crop.
        # We use strict params first, then relaxed if needed (though unlikely if small found it).
        face, _ = _crop_face_passport(img_rotated, pad_top_ratio=pad_top_ratio, scale_factor=1.1, min_neighbors=5, min_size=(80, 80))
        
        if face is None:
             # Retry relaxed on full res
             face, _ = _crop_face_passport(img_rotated, pad_top_ratio=pad_top_ratio, scale_factor=1.05, min_neighbors=3, min_size=(40, 40))
             
        if face is not None:
            return face, angle, weight

    return None, None, 0.0


def _choose_pan_rotation(img):
    # Use id(img) as a per-run cache key. Since main.py passes the same
    # ndarray object to both face and signature tasks, this prevents
    # redundant heavy face detection.
    img_id = id(img)
    with _PAN_ROT_LOCK:
        if img_id in _PAN_ROT_CACHE:
            return _PAN_ROT_CACHE[img_id]

    # ...
    best_angle = 0
    best_score = -100.0
    h_img, w_img = img.shape[:2]

    for angle in (0, 90, 180, 270):
        rotated = _rotate_image(img, angle)
        bbox_result = _haar_detect_bbox(rotated, scale_factor=1.05, min_neighbors=3, min_size=(40, 40))
        score = -10.0
        if bbox_result is not None:
            _, _, _, _, weight = bbox_result
            score = weight
        if angle == 0:
            score += 0.1
        if score > best_score:
            best_score = score
            best_angle = angle

    with _PAN_ROT_LOCK:
        # Simple cleanup if cache grows too large
        if len(_PAN_ROT_CACHE) > 50:
            _PAN_ROT_CACHE.clear()
        _PAN_ROT_CACHE[img_id] = best_angle

    return best_angle




def _signature_crop_by_keyword(img):
    ocr = get_ocr()
    if ocr is None:
        return None

    h, w, _ = img.shape
    max_w = 800
    scale = 1.0
    ocr_img = img
    if w > max_w:
        scale = max_w / w
        ocr_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    result = ocr.ocr(ocr_img)
    lines = result[0] if result and isinstance(result, list) and len(result) > 0 else result
    if not lines:
        return None

    items = []
    if isinstance(lines, dict) and 'rec_texts' in lines and 'rec_boxes' in lines:
        for i in range(len(lines['rec_texts'])):
            items.append((lines['rec_texts'][i], lines['rec_boxes'][i]))
    elif isinstance(lines, list):
        for line in lines:
            if isinstance(line, list) and len(line) >= 2:
                try:
                    items.append((line[1][0], line[0]))
                except:
                    pass

    keywords = ("signature", "sign", "holder", "sd")
    for text, box in items:
        text = text.lower().strip()
        if any(k in text for k in keywords):
            # Skip issuing authority signatures if they contain "issuing" or "authority"
            if "issuing" in text or "authority" in text:
                continue

            if len(box) == 4 and isinstance(box[0], (int, float, np.integer, np.floating)):
                # Flat array: [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = [v / scale for v in box]
            else:
                # Nested list: [[x,y], [x,y], [x,y], [x,y]]
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                x_min, x_max = min(xs) / scale, max(xs) / scale
                y_min, y_max = min(ys) / scale, max(ys) / scale

            # Keep a wider band around the signature area to avoid over-cropping.
            # NEW LOGIC: The keyword "Signature" is BELOW the actual signature.
            # So we want the crop to END at y_min (top of the word "Signature")
            # and start some amount above it.
            
            # y_min is the top of "Signature" word.
            signature_bottom = int(_clamp(y_min, 0, h - 1))
            
            # Estimate signature height. Usually about 15-20% of card height?
            # Or just use a fixed context? Let's use 15% of image height as safe bet.
            sig_height = int(0.12 * h) 
            signature_top = int(_clamp(signature_bottom - sig_height, 0, h - 1))
            
            # Left/Right: Use the keyword center? No, signature is usually CENTERED above it 
            # or sometimes shifted left.
            # Let's take a generous width centered on the keyword, but biased left?
            # Actually, the previous logic of expanding band_left/right was okay, 
            # just need to check overlapping text.
            
            band_left = int(_clamp(x_min - 0.15 * w, 0, w - 1))
            band_right = int(_clamp(x_max + 0.35 * w, 0, w - 1))
            
            # If QR is present, keep crop left of it.
            qr_bbox = _find_qr_bbox(img)
            if qr_bbox is not None:
                qx1, _, _, _ = qr_bbox
                band_right = min(band_right, _clamp(qx1 - int(0.02 * w), 0, w - 1))

            signature = img[signature_top:signature_bottom, band_left:band_right]
            
            # Check if this crop actually contains any ink
            test_sig = _tight_crop_signature(signature)
            if test_sig is not None and test_sig.size > 0:
                # If we found ink, proceed with the crop
                signature = _trim_white_border(signature, threshold=225)
                signature = _remove_text_lines(signature)
                return signature
            
            # If no ink found immediately above, try a slightly larger search area above keyword
            signature_top = int(_clamp(signature_bottom - int(0.18 * h), 0, h - 1))
            signature = img[signature_top:signature_bottom, band_left:band_right]
            test_sig = _tight_crop_signature(signature)
            if test_sig is not None and test_sig.size > 0:
                signature = _trim_white_border(signature, threshold=225)
                signature = _remove_text_lines(signature)
                return signature

    return None


def _find_qr_bbox(img):
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(img)
    if points is None or len(points) == 0:
        return None
    pts = points[0] if len(points.shape) == 3 else points
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    return x_min, y_min, x_max, y_max


def _signature_crop_left_of_qr(img, qr_bbox):
    h, w, _ = img.shape
    x_min, y_min, x_max, y_max = qr_bbox

    # Focus on the lower band and stay left of the QR block.
    y1 = int(0.58 * h)
    y2 = int(0.88 * h)
    x1 = int(0.06 * w)
    x2 = _clamp(x_min - int(0.02 * w), 0, w - 1)

    if x2 - x1 < int(0.20 * w):
        return None

    signature = img[y1:y2, x1:x2]
    signature = _trim_white_border(signature, threshold=225)
    signature = _tight_crop_signature(signature)
    return signature


def _tight_crop_signature(img):
    # Keep only signature strokes; remove background and printed text as much as possible.
    if img is None or img.size == 0:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Pre-process to highlight ink
    # Invert so ink is white, background is black
    # Use Adaptive Thresholding to handle uneven lighting on PAN cards
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )
    
    # 2. Filter Noise and Connect Signature Strokes
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_connect, iterations=1)

    # 3. SMART TEXT FILTERING: Group printed text lines into solid horizontal blocks
    kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    mask_text = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_text, iterations=1)

    # 4. Connected Components on the wide text blocks to find blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_text, connectivity=8)
    if num_labels <= 1:
        return img # No blobs found

    h_img, w_img = mask_text.shape
    
    candidate_indices = []
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        # Filter tiny noise
        if area < 50: continue
        
        # Filter things that span entire width (borders)
        if w > 0.95 * w_img: continue
        
        # Filter pure horizontal lines or huge text blocks
        aspect_ratio = w / float(h) if h > 0 else 0
        extent = area / float(w * h)
        
        # SMART FILTER: If the block is very solid after morphological closing, it's printed text!
        # Printed addresses and labels group into solid rectangles (extent > 0.80).
        # Signatures are sparse and cursive, so they remain jagged.
        if extent > 0.65: 
            continue
            
        # Ignore things taking up huge portions of the image
        if area > h_img * w_img * 0.4:
             continue
             
        # Ignore extremely long sentences (aspect ratio > 12)
        if aspect_ratio > 10.0:
             continue
             
        # For DL fallback: if it's on the left half, it's probably "Class of vehicle" text
        # Signatures in this fallback ROI are heavily right biased. We can enforce it.
        # But this is a generic tight crop so be careful... Let's just use aspect ratio.
             
        candidate_indices.append(i)
        
    if not candidate_indices:
        return img
        
    # Create mask of valid components based on the ORIGINALLY connected fine-strokes
    final_mask = np.zeros_like(mask_text)
    for i in candidate_indices:
        final_mask[labels == i] = 255
        
    # AND operation to ensure we only get the original ink inside these valid blocks
    ink_mask = cv2.bitwise_and(mask, final_mask)
        
    # 5. Find bounding box of all valid signature ink combined
    coords = cv2.findNonZero(ink_mask)
    if coords is None:
        return img

    x, y, bw, bh = cv2.boundingRect(coords)
    
    # Add comfortable padding
    pad_y = 2
    pad_x = 2
    
    # Expand horizontally to achieve a more standard signature width (e.g., 3.5:1 ratio)
    target_aspect = 3.5
    current_aspect = bw / float(bh) if bh > 0 else 0
    
    if current_aspect < target_aspect:
        target_bw = int(bh * target_aspect)
        extra_w = target_bw - bw
        pad_x_left = extra_w // 2
        pad_x_right = extra_w - pad_x_left
    else:
        pad_x_left = pad_x
        pad_x_right = pad_x
        
    x1 = _clamp(x - pad_x_left, 0, w_img - 1)
    y1 = _clamp(y - pad_y, 0, h_img - 1)
    x2 = _clamp(x + bw + pad_x_right, 0, w_img - 1)
    y2 = _clamp(y + bh + pad_y, 0, h_img - 1)
    
    return img[y1:y2, x1:x2]


def haar_face(img, output_path):
    face, _ = _crop_face_passport(img)
    if face is None:
        return None
    cv2.imwrite(output_path, face)
    return output_path


# ---------------- AADHAAR FACE ---------------- #

def extract_aadhaar_face(img, output_path):
    # img can be path or ndarray
    if isinstance(img, str):
        img = load_image(img)

    # 1. Try to find a face in any orientation (0, 90, 180, 270)
    face, angle, weight = _crop_face_best_rotation(img)
    
    if face is not None:
        face = cv2.resize(face, (300, 400), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, face)
        return output_path

    # Fallback: Assume upright
    h, w, _ = img.shape
    fallback = img[int(0.18*h):int(0.50*h), int(0.05*w):int(0.35*w)]
    cv2.imwrite(output_path, fallback)
    return output_path


# ---------------- PAN FACE ---------------- #

def extract_pan_face(img, output_path):
    if isinstance(img, str):
        img = load_image(img)

    pan_angle = _choose_pan_rotation(img)
    img = _rotate_image(img, pan_angle)

    face_result = _haar_detect_bbox(img)
    photo_rect = None
    
    if face_result is not None:
         fx, fy, fw, fh, _ = face_result
         photo_rect = _find_photo_rect(img, (fx, fy, fw, fh))
    
    final_face = None
    if photo_rect is not None:
        x1, y1, x2, y2 = photo_rect
        final_face = img[y1:y2, x1:x2]
        if face_result:
             fx, fy, fw, fh, _ = face_result
             face_y_in_crop = fy - y1
             final_face = _trim_disconnected_header(final_face, face_y_in_crop, fh)
    else:
        final_face, _ = _crop_face_passport(img)

    if final_face is None:
        h, w, _ = img.shape
        final_face = img[int(0.14*h):int(0.50*h), int(0.06*w):int(0.36*w)]
        final_face = _trim_white_border(final_face)

    if final_face is not None and final_face.size > 0:
        final_face = cv2.resize(final_face, (300, 400), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, final_face)
        return output_path
        
    return None


# ---------------- PAN SIGNATURE ---------------- #

def extract_pan_signature(img, output_path):
    if isinstance(img, str):
        img = load_image(img)
    pan_angle = _choose_pan_rotation(img)
    img = _rotate_image(img, pan_angle)
    h, w, _ = img.shape

    # 1. Primary: Keyword-based (Signature/Sign)
    signature = _signature_crop_by_keyword(img)
    if signature is not None:
        # Final tight crop to isolate ink strokes
        tight_sig = _tight_crop_signature(signature)
        if tight_sig is not None and tight_sig.size > 0:
            cv2.imwrite(output_path, tight_sig)
        else:
            cv2.imwrite(output_path, signature)
        return output_path

    # 2. QR-based fallback
    qr_bbox = _find_qr_bbox(img)
    if qr_bbox is not None:
        signature = _signature_crop_left_of_qr(img, qr_bbox)
        if signature is not None:
            tight_sig = _tight_crop_signature(signature)
            if tight_sig is not None and tight_sig.size > 0:
                cv2.imwrite(output_path, tight_sig)
            else:
                cv2.imwrite(output_path, signature)
            return output_path

    # 3. Geometric fallback (Specific to standard PAN layout)
    # The signature is usually in the bottom-middle-left area.
    y1, y2 = int(0.55 * h), int(0.90 * h)
    x1, x2 = int(0.05 * w), int(0.70 * w)
    signature_roi = img[y1:y2, x1:x2]
    
    # Use ink stroke detection on this fallback ROI
    signature = _tight_crop_signature(signature_roi)
    
    if signature is None or signature.size == 0:
        # Last resort: non-tight geometric crop
        signature = _trim_white_border(signature_roi, threshold=225)

    cv2.imwrite(output_path, signature)
    return output_path


# ---------------- DL FACE ---------------- #

# =====================================================
# DL FACE EXTRACTION
# =====================================================

















def _check_saturation(img, threshold=60):
    """
    Checks if the image has significant color saturation.
    Signatures should be grayscale (low saturation).
    Footers/headers are often colored (high saturation).
    Returns True if saturation is HIGH (Reject), False if Low (Accept).
    """
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        mean_sat = np.mean(sat)
        return mean_sat > threshold
    except:
        return False

def _signature_crop_above_label(img):
    """
    Finds text like "Signature", "Sign", "Holder", "SD" and returns the crop strictly ABOVE it.
    """
    ocr = get_ocr()
    if ocr is None:
        return None
        
    try:
        h, w = img.shape[:2]
        roi_y = int(0.5 * h)
        roi = img[roi_y:h, 0:w]

        # Cap resolution for speed — keyword detection doesn't need full resolution
        max_w_ocr = 800
        scale = 1.0
        ocr_roi = roi
        if w > max_w_ocr:
            scale = max_w_ocr / w
            ocr_roi = cv2.resize(roi, (int(w * scale), int(roi.shape[0] * scale)),
                                 interpolation=cv2.INTER_AREA)

        result = ocr.ocr(ocr_roi)
        lines = result[0] if result and isinstance(result, list) and len(result) > 0 else result
        
        if not lines:
            return None
            
        keywords = ["sign", "holder", "sd", "signature", "licence"]
        
        items = []
        if isinstance(lines, dict) and 'rec_texts' in lines and 'rec_boxes' in lines:
            for i in range(len(lines['rec_texts'])):
                items.append((lines['rec_texts'][i], lines['rec_boxes'][i]))
        elif isinstance(lines, list):
            for line in lines:
                if isinstance(line, list) and len(line) >= 2:
                    try:
                        items.append((line[1][0], line[0]))
                    except:
                        pass
        
        for text, bbox in items:
            text = text.lower().strip()
            
            if any(k in text for k in keywords):
                if "issuing" in text or "authority" in text:
                    continue

                if len(bbox) == 4 and isinstance(bbox[0], (int, float, np.integer, np.floating)):
                    y_min = bbox[1] / scale
                    text_center_x = (bbox[0] + bbox[2]) / 2.0 / scale
                else:
                    ys = [pt[1] for pt in bbox]
                    y_min = min(ys) / scale
                    xs = [pt[0] for pt in bbox]
                    text_center_x = sum(xs) / len(xs) / scale

                # Driver signature is generally on the right side of the layout.
                # Skip labels on the left half (which are usually issuing authority)
                if text_center_x < 0.45 * w:
                    continue

                top_y = y_min
                global_top_y = roi_y + top_y
                
                sig_height = int(0.15 * h)
                y2 = int(global_top_y)
                y1 = max(0, y2 - sig_height)
                
                half_w = int(0.35 * w)
                
                x1 = int(max(0, text_center_x - half_w))
                x2 = int(min(w, text_center_x + half_w))
                
                return img[y1:y2, x1:x2]
                
    except Exception as e:
        return None
    return None

def extract_dl_signature(img, output_path):
    if isinstance(img, str):
        img = load_image(img)
    h, w = img.shape[:2]
    
    raw_signature_crop = None

    # 1. NEW PRIMARY: Try "Above Label" extraction
    signature_above = _signature_crop_above_label(img)
    if signature_above is not None:
         # VALIDATE SATURATION AND REJECT PRINTED TEXT
         if not _check_saturation(signature_above):
             test_crop = _tight_crop_signature(signature_above)
             if test_crop is not None and test_crop.size > 0:
                 raw_signature_crop = signature_above
    
    # 2. Fallback: Keyword-based extraction
    if raw_signature_crop is None:
        signature = _signature_crop_by_keyword(img)
        if signature is not None:
            # VALIDATE SATURATION AND REJECT PRINTED TEXT
            if not _check_saturation(signature):
                test_crop = _tight_crop_signature(signature)
                if test_crop is not None and test_crop.size > 0:
                    raw_signature_crop = signature

    # 3. Fallback: Geometric Search (IMPROVED + SATURATION CHECK)
    if raw_signature_crop is None:
        # Signature is usually on the far right side, low down.
        # Strict Search Area: 70% to 90% height, 60% to 100% width
        search_area = img[int(0.70*h):int(0.90*h), int(0.60*w):w]
        
        signature_fallback = _tight_crop_signature(search_area)
        
        if signature_fallback is not None and signature_fallback.size > 0:
             mean_val = np.mean(signature_fallback)
             is_footer = (mean_val < 150) # Brightness check (loosened from 180)
             is_saturated = _check_saturation(signature_fallback) # NEW checking
             
             is_too_big = (signature_fallback.shape[0] > search_area.shape[0] * 0.95)
             
             if not is_footer and not is_saturated and not is_too_big:
                  raw_signature_crop = signature_fallback
        
    # 4. Last Resort: Fixed Region (IMPROVED)
    if raw_signature_crop is None:
         # Target the white space specifically.
         # 72% to 85% height 
         # 30% to 90% width
         fixed_crop = img[int(0.72*h):int(0.85*h), int(0.30*w):int(0.90*w)]
         # Even fixed crop must pass saturation check, else we might return nothing (better than wrong)
         if not _check_saturation(fixed_crop):
             raw_signature_crop = fixed_crop

    # FINAL STEP: Strict Cleanup
    if raw_signature_crop is not None and raw_signature_crop.size > 0:
        try:
            final_sig = _tight_crop_signature(raw_signature_crop)
            if final_sig is None or final_sig.size == 0:
                final_sig = raw_signature_crop
        except:
             final_sig = raw_signature_crop
            
        cv2.imwrite(output_path, final_sig)
        return output_path
        
    return None



def _snap_to_border(img, threshold=210, pad=0):
    """
    Snaps the crop to the actual content borders, removing white/gray padding.
    Uses strict fixed thresholding (210) to treat scanned paper noise as background.
    """
    if img is None or img.size == 0:
        return img
        
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Fixed Strict Threshold
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
        
    # 3. Find largest bounding box
    max_area = 0
    best_rect = (0, 0, w, h)
    
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area > h * w * 0.03:
            if area > max_area:
                max_area = area
                best_rect = (x, y, bw, bh)
                
    x, y, bw, bh = best_rect
    
    if bw > w * 0.95 and bh > h * 0.95:
        return img
        
    # 4. Crop
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)
    
    return img[y1:y2, x1:x2]

def _aggressive_bottom_clean(img):
    """
    Aggressively removes text residue from the bottom of the image.
    Uses gradient detection (Sobel) to find horizontal edges.
    """
    if img is None or img.size == 0:
        return img
        
    h, w = img.shape[:2]
    scan_h = int(0.2 * h)
    roi_y = max(0, h - scan_h)
    roi = img[roi_y:h, 0:w]
    
    if roi.size == 0:
        return img
        
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel = np.absolute(sobel_y)
    grad = np.uint8(255 * abs_sobel / np.max(abs_sobel)) if np.max(abs_sobel) > 0 else np.uint8(abs_sobel)
    
    _, edges = cv2.threshold(grad, 50, 255, cv2.THRESH_BINARY)
    proj = np.sum(edges, axis=1)
    
    width_thresh = 0.3 * w * 255
    cut_relative = scan_h
    found_edge = False
    
    for y in range(scan_h):
        val = proj[y]
        if val > width_thresh:
            cut_relative = y
            found_edge = True
            break
            
    if found_edge:
        cut_y = roi_y + cut_relative
        return img[0:cut_y, 0:w]
        
    return img

def extract_dl_face(img, output_path):
    if isinstance(img, str):
        img = load_image(img)
    h, w = img.shape[:2]

    # Focus only on upper region where face ALWAYS exists
    roi_h = int(h * 0.65)
    roi_w = w  # Search full width to account for new layouts
    roi = img[0:roi_h, 0:roi_w]

    # 1. First, try "Exact Photo Detection" (Box Detection) on the ROI
    face_result = _haar_detect_bbox(roi)
    
    final_face = None
    
    if face_result is not None:
        fx, fy, fw, fh, _ = face_result
        
        # Try to find the exact photo rectangle *around* the face
        photo_rect = _find_photo_rect(roi, (fx, fy, fw, fh))
        
        if photo_rect is not None:
            px1, py1, px2, py2 = photo_rect
            
            # Remove strict clamp to let photo_rect work or provide generous headroom
            limit_y2 = fy + int(2.5 * fh)
            py2 = min(py2, limit_y2)
            
            final_face = roi[py1:py2, px1:px2]
            
            # Trim headers inside the box
            face_y_in_crop = fy - py1
            final_face = _trim_disconnected_header(final_face, face_y_in_crop, fh)
            
            # Combined Aggressive Cleanup
            final_face = _trim_bottom_text(final_face)
            final_face = _aggressive_bottom_clean(final_face)
            final_face = _snap_to_border(final_face) # NEW: Remove side padding
            
            if final_face is not None:
                final_face = _tight_crop_by_nonwhite(final_face, threshold=240, pad=0)
    
    if final_face is None:
        # 2. Fallback: Smart Crop on ROI
        
        if face_result:
             fx, fy, fw, fh, _ = face_result
             
             # MINIMAL PADDING: 10%
             pad_top = int(0.20 * fh)  
             pad_bot = int(0.60 * fh)  
             pad_side = int(0.25 * fw)
             
             c_y1 = max(0, fy - pad_top)
             c_y2 = min(roi_h, fy + fh + pad_bot)
             
             c_y2 = min(c_y2, fy + int(2.5 * fh))
             
             c_x1 = max(0, fx - pad_side)
             c_x2 = min(roi_w, fx + fw + pad_side)
             
             final_face = roi[c_y1:c_y2, c_x1:c_x2]
             
             # Aspect Ratio Guard
             if final_face is not None and final_face.size > 0:
                 fh_crop, fw_crop = final_face.shape[:2]
                 ratio = fh_crop / fw_crop
                 if ratio > 1.35: 
                     new_h = int(fw_crop * 1.3)
                     final_face = final_face[:new_h, :]

             final_face = _trim_disconnected_header(final_face, fy - c_y1, fh)
             final_face = _trim_bottom_text(final_face)
             final_face = _aggressive_bottom_clean(final_face)
             final_face = _snap_to_border(final_face) # NEW: Remove side padding
        else:
             pass

    # 3. Last Resort: Fixed Crop
    if final_face is None:
        final_face = roi[int(0.20*roi_h):int(0.50*roi_h), int(0.15*roi_w):int(0.45*roi_w)]

    # GLOBAL CLEANUP (For all paths)
    if final_face is not None and final_face.size > 0:
        # trim bottom text again just in case
        final_face = _trim_bottom_text(final_face)
        final_face = _aggressive_bottom_clean(final_face)
        final_face = _snap_to_border(final_face, threshold=210)
        
        if final_face is not None and final_face.size > 0:
             final_face = _tight_crop_by_nonwhite(final_face, threshold=240, pad=0)

    if final_face is not None and final_face.size > 0:
        final_face = cv2.resize(final_face, (300, 400), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, final_face)
        return output_path
        
    return None



# ---------------- MARKSHEET FACE ---------------- #

def extract_marksheet_face(img, output_path):
    if isinstance(img, str):
        img = load_image(img)
    if img is None:
        return None
        
    h, w, _ = img.shape

    # Relaxed search region to accommodate photos that might be located lower.
    search_region = img[0:int(0.50*h), 0:w]
    
    # 1. Best Rotation Search (Standard)
    face_img, angle, weight = _crop_face_best_rotation(search_region, pad_top_ratio=0.30)
    
    # 2. FALLBACK: Upper Right Quadrant Focus (More focused search)
    if (face_img is None or weight < 12.0):
        # Slightly wider right header search
        right_header = img[0:int(0.50*h), int(0.50*w):w]
        face_img_r, angle_r, weight_r = _crop_face_best_rotation(right_header, pad_top_ratio=0.30)
        if face_img_r is not None and weight_r > 8.0:
            face_img, weight = face_img_r, weight_r

    # 3. ULTRA-RELAXED FALLBACK: Full Header sweep (min_neighbors=2)
    # Reduced min_neighbors to 2 to improve recall on grainy or low-res marksheets
    if (face_img is None or weight < 8.0):
        # Header search failed, try FULL image search (accommodates bottom-placed photos)
        full_img_search = img[0:h, 0:w]
        face_img_u, weight_u = _crop_face_passport(
            full_img_search,
            pad_top_ratio=0.30,
            scale_factor=1.05,
            min_neighbors=2, 
            min_size=(30, 30)
        )
        if face_img_u is not None and weight_u > 5.0:
            face_img, weight = face_img_u, weight_u

    # VALIDATION & SAVE
    if face_img is not None and face_img.size > 0 and weight > 3.0:
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray_face)
        
        # Check aspect ratio to reject long text blocks masquerading as faces
        fh, fw = face_img.shape[:2]
        aspect_ratio = fw / fh
        
        # Widen range to 1-130 (from 2-110) and aspect ratio (0.4 to 1.5)
        # Relaxed std_dev check for grainy/low-res photos
        if 1.0 < std_dev < 130 and 0.4 <= aspect_ratio <= 1.5:
            cv2.imwrite(output_path, face_img)
            return output_path
            
    return None
