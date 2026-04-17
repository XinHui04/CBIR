import cv2
import numpy as np
from skimage.feature import hog

from preprocess import preprocess_image


def _contour_touches_border(contour, width, height, border=1):
    x, y, w, h = cv2.boundingRect(contour)
    return x <= border or y <= border or (x + w) >= (width - border) or (y + h) >= (height - border)


def _largest_valid_contour(binary):
    height, width = binary.shape[:2]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, False

    # Prefer contours not touching image borders to avoid selecting background/frame shapes.
    valid = [c for c in contours if not _contour_touches_border(c, width, height)]
    if valid:
        return max(valid, key=cv2.contourArea), True
    return max(contours, key=cv2.contourArea), False


def _build_object_mask(gray, use_largest_contour=True):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    candidates = [
        cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2),
        cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel, iterations=2),
    ]

    if not use_largest_contour:
        # Use the candidate with a more balanced foreground ratio.
        ratios = [np.count_nonzero(c) / c.size for c in candidates]
        best_idx = min(range(len(ratios)), key=lambda i: abs(ratios[i] - 0.35))
        return candidates[best_idx]

    best_mask = None
    best_area = -1.0
    best_is_valid = False
    for candidate in candidates:
        contour, is_valid = _largest_valid_contour(candidate)
        if contour is None:
            continue

        area = cv2.contourArea(contour)

        # Prefer a real (non-border-touching) contour over a border-touching
        # fallback; among contours of equal validity, prefer larger area.
        # Without this, Otsu's "background frame" contour almost always wins
        # on area and collapses the mask to the full image.
        if is_valid and not best_is_valid:
            better = True
        elif is_valid == best_is_valid:
            better = area > best_area
        else:
            better = False

        if better:
            mask = np.zeros_like(candidate)
            cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
            best_mask = mask
            best_area = area
            best_is_valid = is_valid

    # Reject masks that cover almost the whole frame — those are the
    # "background frame" contour, not an object. Fall through to the
    # ratio-based fallback instead of returning a degenerate mask.
    if best_mask is not None and (np.count_nonzero(best_mask) / best_mask.size) <= 0.9:
        return best_mask

    # Final fallback if contour extraction fails or only produced degenerate
    # full-frame masks on both threshold variants.
    ratios = [np.count_nonzero(c) / c.size for c in candidates]
    best_idx = min(range(len(ratios)), key=lambda i: abs(ratios[i] - 0.35))
    return candidates[best_idx]


def extract_hu_moments(image, apply_log_transform=True, use_largest_contour=True):
    """Extract 7 Hu Moments from a BGR image."""

    if image is None:
        raise ValueError("Input image is None")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary = _build_object_mask(gray, use_largest_contour=use_largest_contour)

    # Compute moments
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten().astype(np.float32)

    # Log transform
    if apply_log_transform:
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    # Normalize to keep shape values stable when combined with other feature types.
    norm = np.linalg.norm(hu)
    if norm > 0:
        hu /= norm

    return hu


def extract_hog_descriptor(image, size=(64, 64)):
    """HOG descriptor on a grayscale, downsized view of the input BGR image."""

    if image is None:
        raise ValueError("Input image is None")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    descriptor = hog(
        gray_small,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
        channel_axis=None,
    ).astype(np.float32)

    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor /= norm

    return descriptor


def extract_shape_feature(image_path, size=(256, 256), apply_log_transform=True):
    """Load an image and extract a hybrid shape vector (Hu + HOG)."""
    image = preprocess_image(image_path, size=size)
    hu = extract_hu_moments(image, apply_log_transform=apply_log_transform)
    hog_vec = extract_hog_descriptor(image)
    return np.concatenate([hu, hog_vec]).astype(np.float32)


# (FOR UI / DEBUG)
def visualize_binary(image_path):
    image = preprocess_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow("Binary", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TEST
if __name__ == "__main__":
    sample_path = "Query/sample.jpg"

    try:
        feature = extract_shape_feature(sample_path)

        print("Shape feature length:", len(feature))  # should be 331 (7 Hu + 324 HOG)
        print("Feature values:", feature)

    except Exception as exc:
        print("Shape feature extraction failed:", exc)