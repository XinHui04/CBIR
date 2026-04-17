import cv2
import numpy as np

from preprocess import preprocess_image


def extract_hsv_histogram(image, bins=(8, 8, 8), normalize=True, use_mask=True):

    if image is None:
        raise ValueError("Input image is None")

    if not isinstance(bins, tuple) or len(bins) != 3:
        raise ValueError("bins must be a tuple of 3 integers (h_bins, s_bins, v_bins)")

    if any(b <= 0 for b in bins):
        raise ValueError("All bin values must be positive")

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = None
    if use_mask:
        # Masking (combine HSV + grayscale idea)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # remove bright background
        _, mask1 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # remove low-saturation areas (background)
        lower = np.array([0, 30, 30])
        upper = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower, upper)

        # combine masks
        mask = cv2.bitwise_and(mask1, mask2)

    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        mask,
        bins,
        [0, 180, 0, 256, 0, 256]
    )

    if normalize:
        hist = cv2.normalize(hist, hist).flatten()
    else:
        hist = hist.flatten()

    return hist.astype(np.float32)

def extract_color_feature(image_path, size=(256, 256), bins=(8, 8, 8), normalize=True):
    """Load an image and extract its HSV histogram feature vector."""
    image = preprocess_image(image_path, size=size)
    return extract_hsv_histogram(image, bins=bins, normalize=normalize)


def visualize_histogram(hist, bins=(8, 8, 8)):
    """Simple visualization helper (flattened histogram)."""
    import matplotlib.pyplot as plt

    plt.plot(hist)
    plt.title("HSV Histogram Feature")
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    sample_path = "Query/sample.jpg"

    try:
        feature = extract_color_feature(sample_path)

        print("HSV feature length:", len(feature))  # should be 512
        print("First 10 values:", feature[:10])

        # visualize_histogram(feature)  # optional

    except Exception as exc:
        print("Color feature extraction failed:", exc)