import cv2
import numpy as np

from preprocess import preprocess_image


def extract_hsv_histogram(image, bins=(8, 8, 8), normalize=True, use_mask=True):
    """Extract a flattened HSV color histogram feature vector."""

    if image is None:
        raise ValueError("Input image is None")

    if not isinstance(bins, tuple) or len(bins) != 3:
        raise ValueError("bins must be a tuple of 3 integers (h_bins, s_bins, v_bins)")

    if any(b <= 0 for b in bins):
        raise ValueError("All bin values must be positive")

    # ✅ Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 🚀 NEW: Background Masking (important for your dataset)
    mask = None
    if use_mask:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # simple threshold to remove bright background
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # ✅ Compute histogram
    hist = cv2.calcHist(
        [hsv],
        channels=[0, 1, 2],
        mask=mask,
        histSize=[bins[0], bins[1], bins[2]],
        ranges=[0, 180, 0, 256, 0, 256],
    )

    # ✅ Normalize using OpenCV (better stability)
    if normalize:
        hist = cv2.normalize(hist, hist).flatten()
    else:
        hist = hist.flatten()

    return hist.astype(np.float32)


def extract_color_feature(image_path, size=(256, 256), bins=(8, 8, 8), normalize=True):
    """Load an image and extract its HSV histogram feature vector."""
    image = preprocess_image(image_path, size=size)
    return extract_hsv_histogram(image, bins=bins, normalize=normalize)


# 🚀 OPTIONAL (FOR UI / PRESENTATION)
def visualize_histogram(hist, bins=(8, 8, 8)):
    """Simple visualization helper (flattened histogram)."""
    import matplotlib.pyplot as plt

    plt.plot(hist)
    plt.title("HSV Histogram Feature")
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.show()


# 🚀 TEST
if __name__ == "__main__":
    sample_path = "Query/sample.jpg"

    try:
        feature = extract_color_feature(sample_path)

        print("HSV feature length:", len(feature))  # should be 512
        print("First 10 values:", feature[:10])

        # visualize_histogram(feature)  # optional

    except Exception as exc:
        print("Color feature extraction failed:", exc)