import cv2
import numpy as np

from preprocess import preprocess_image


def extract_hu_moments(image, apply_log_transform=True, use_largest_contour=True):
    """Extract 7 Hu Moments from a BGR image."""

    if image is None:
        raise ValueError("Input image is None")

    # ✅ Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 🚀 Improve thresholding (reduce noise)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 🚀 Morphological cleanup (VERY IMPORTANT)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 🚀 Focus on main object only
    if use_largest_contour:
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)

            mask = np.zeros_like(binary)
            cv2.drawContours(mask, [largest], -1, 255, thickness=-1)

            binary = mask  # replace with clean object mask

    # ✅ Compute moments
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten().astype(np.float32)

    # ✅ Log transform (you already did correctly ✔)
    if apply_log_transform:
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    # 🚀 Normalize (important when combining features)
    norm = np.linalg.norm(hu)
    if norm > 0:
        hu /= norm

    return hu


def extract_shape_feature(image_path, size=(256, 256), apply_log_transform=True):
    """Load an image and extract its Hu Moments shape feature vector."""
    image = preprocess_image(image_path, size=size)
    return extract_hu_moments(image, apply_log_transform=apply_log_transform)


# 🚀 OPTIONAL (FOR UI / DEBUG)
def visualize_binary(image_path):
    image = preprocess_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow("Binary", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 🚀 TEST
if __name__ == "__main__":
    sample_path = "Query/sample.jpg"

    try:
        feature = extract_shape_feature(sample_path)

        print("Hu Moments feature length:", len(feature))  # should be 7
        print("Feature values:", feature)

    except Exception as exc:
        print("Shape feature extraction failed:", exc)