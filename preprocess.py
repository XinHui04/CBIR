import cv2
import os


# ✅ Define allowed furniture categories
ALLOWED_CATEGORIES = {
    "chair",
    "sofa",
    "table",
    "bed",
    "cabinet",
    "bench",
    "wardrobe",
    "rack"
}


def preprocess_image(image_path, size=(256, 256)):
    """Load and preprocess an image for feature extraction."""

    if not isinstance(size, tuple) or len(size) != 2:
        raise ValueError("size must be a tuple of (width, height)")

    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Resize width and height must be positive integers")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image from path: {image_path}")

    # ✅ Resize
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    return image


# 🚀 NEW: Load dataset with category filtering
def load_dataset(dataset_path):
    """
    Load dataset while filtering only allowed furniture categories.

    Returns:
        list of (image_path, category)
    """

    data = []

    for category in os.listdir(dataset_path):

        category_path = os.path.join(dataset_path, category)

        # ❌ Skip non-folder
        if not os.path.isdir(category_path):
            continue

        # ❌ Skip unwanted categories
        if category.lower() not in ALLOWED_CATEGORIES:
            print(f"Skipping category: {category}")
            continue

        print(f"Loading category: {category}")

        for file in os.listdir(category_path):

            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(category_path, file)
                data.append((image_path, category))

    return data


# 🚀 OPTIONAL: Convert formats (for later use)
def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 🚀 TEST
if __name__ == "__main__":

    dataset_path = "dataset/"   # 🔁 change to your dataset folder

    try:
        dataset = load_dataset(dataset_path)
        print(f"Total filtered images: {len(dataset)}")

        # Test one image
        sample_path, label = dataset[0]
        processed = preprocess_image(sample_path)

        print("Sample category:", label)
        print("Preprocessed image shape:", processed.shape)

    except Exception as exc:
        print("Error:", exc)