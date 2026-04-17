import cv2
import numpy as np

from preprocess import preprocess_image


def _compute_lbp(gray_image):
    """Compute basic 8-neighbor LBP codes."""
    h, w = gray_image.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

    center = gray_image[1:-1, 1:-1]

    neighbors = [
        gray_image[0:-2, 0:-2],
        gray_image[0:-2, 1:-1],
        gray_image[0:-2, 2:],
        gray_image[1:-1, 2:],
        gray_image[2:, 2:],
        gray_image[2:, 1:-1],
        gray_image[2:, 0:-2],
        gray_image[1:-1, 0:-2],
    ]

    for bit, neighbor in enumerate(neighbors):
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    return lbp


# Uniform pattern mapping
def _get_uniform_lbp_mapping():
    """Create mapping for uniform LBP (59 bins)."""
    mapping = np.zeros(256, dtype=np.uint8)
    index = 0

    for i in range(256):
        binary = np.binary_repr(i, width=8)
        transitions = sum((binary[j] != binary[(j + 1) % 8]) for j in range(8))

        if transitions <= 2:
            mapping[i] = index
            index += 1
        else:
            mapping[i] = 58  # non-uniform bin

    return mapping


UNIFORM_MAPPING = _get_uniform_lbp_mapping()


def extract_lbp_feature(image, normalize=True, use_uniform=True):
    """Extract LBP / ULBP texture feature."""

    if image is None:
        raise ValueError("Input image is None")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    lbp = _compute_lbp(gray)

    # Apply uniform mapping
    if use_uniform:
        lbp = UNIFORM_MAPPING[lbp]
        bins = 59
    else:
        bins = 256

    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    hist = hist.astype(np.float32)

    # L2 normalization
    if normalize:
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm

    return hist


def extract_texture_feature(image_path, size=(256, 256), normalize=True):
    """Load an image and extract its LBP texture feature."""
    image = preprocess_image(image_path, size=size)
    return extract_lbp_feature(image, normalize=normalize)


def visualize_lbp(image_path):
    image = preprocess_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = _compute_lbp(gray)

    cv2.imshow("LBP", lbp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sample_path = "Query/sample.jpg"

    try:
        feature = extract_texture_feature(sample_path)

        print("LBP feature length:", len(feature))  # should be 59
        print("First 10 values:", feature[:10])

    except Exception as exc:
        print("Texture feature extraction failed:", exc)