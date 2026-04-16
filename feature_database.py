import os
from pathlib import Path

import numpy as np

from color_feature import extract_color_feature
from shape_feature import extract_shape_feature
from texture_feature import extract_texture_feature


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(dataset_dir):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    image_paths = []
    for path in dataset_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)

    image_paths.sort()
    return image_paths


def extract_feature_parts(
    image_path,
    size=(256, 256),
    color_bins=(8, 12, 3),
    normalize_color=True,
    normalize_texture=True,
    log_shape=True,
):
    return {
        "color": extract_color_feature(
            str(image_path),
            size=size,
            bins=color_bins,
            normalize=normalize_color,
        ).astype(np.float32),
        "texture": extract_texture_feature(
            str(image_path),
            size=size,
            normalize=normalize_texture,
        ).astype(np.float32),
        "shape": extract_shape_feature(
            str(image_path),
            size=size,
            apply_log_transform=log_shape,
        ).astype(np.float32),
    }


def extract_combined_feature(
    image_path,
    size=(256, 256),
    color_bins=(8, 12, 3),
    use_color=True,
    use_texture=True,
    use_shape=True,
    normalize_color=True,
    normalize_texture=True,
    log_shape=True,
):
    parts = extract_feature_parts(
        image_path=image_path,
        size=size,
        color_bins=color_bins,
        normalize_color=normalize_color,
        normalize_texture=normalize_texture,
        log_shape=log_shape,
    )

    features = []
    if use_color:
        features.append(parts["color"])
    if use_texture:
        features.append(parts["texture"])
    if use_shape:
        features.append(parts["shape"])

    if not features:
        raise ValueError("At least one feature must be selected")

    return np.concatenate(features).astype(np.float32)


def build_feature_database(
    dataset_dir="Dataset",
    output_path="feature_database.npz",
    size=(256, 256),
    color_bins=(8, 12, 3),
    use_color=True,
    use_texture=True,
    use_shape=True,
):
    if not (use_color or use_texture or use_shape):
        raise ValueError("At least one feature must be selected")

    image_paths = list_images(dataset_dir)
    if not image_paths:
        raise ValueError(f"No images found in dataset directory: {dataset_dir}")

    color_features = []
    texture_features = []
    shape_features = []
    labels = []
    kept_paths = []

    print("Building feature database...")

    for img_path in image_paths:
        try:
            parts = extract_feature_parts(
                img_path,
                size=size,
                color_bins=color_bins,
            )

            color_features.append(parts["color"])
            texture_features.append(parts["texture"])
            shape_features.append(parts["shape"])
            kept_paths.append(str(img_path))
            labels.append(img_path.parent.name)
        except Exception as exc:
            print(f"Skipping {img_path}: {exc}")

    if not kept_paths:
        raise ValueError("No valid features extracted")

    color_features = np.vstack(color_features).astype(np.float32)
    texture_features = np.vstack(texture_features).astype(np.float32)
    shape_features = np.vstack(shape_features).astype(np.float32)
    kept_paths = np.array(kept_paths, dtype=object)
    labels = np.array(labels, dtype=object)

    combined_features = []
    if use_color:
        combined_features.append(color_features)
    if use_texture:
        combined_features.append(texture_features)
    if use_shape:
        combined_features.append(shape_features)
    features = np.concatenate(combined_features, axis=1).astype(np.float32)

    np.savez_compressed(
        output_path,
        features=features,
        color_features=color_features,
        texture_features=texture_features,
        shape_features=shape_features,
        image_paths=kept_paths,
        labels=labels,
    )

    print("Database saved:", output_path)
    return features, kept_paths, labels


def load_feature_database(npz_path="feature_database.npz"):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Feature database not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        if {
            "color_features",
            "texture_features",
            "shape_features",
            "image_paths",
            "labels",
        }.issubset(data.files):
            return {
                "features": data["features"],
                "color_features": data["color_features"],
                "texture_features": data["texture_features"],
                "shape_features": data["shape_features"],
                "image_paths": data["image_paths"],
                "labels": data["labels"],
            }

        return {
            "features": data["features"],
            "color_features": None,
            "texture_features": None,
            "shape_features": None,
            "image_paths": data["image_paths"],
            "labels": data["labels"],
        }


if __name__ == "__main__":
    try:
        features, image_paths, labels = build_feature_database(
            dataset_dir="Dataset",
            output_path="feature_database.npz",
            use_color=True,
            use_texture=True,
            use_shape=True,
        )

        print("\nFeature database built successfully")
        print("Total images:", len(image_paths))
        print("Feature shape:", features.shape)
        print("Example label:", labels[0])
        print("Example path:", image_paths[0])
    except Exception as exc:
        print("Failed:", exc)
