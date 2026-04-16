import csv
import os
from pathlib import Path

import numpy as np

from color_feature import extract_color_feature
from shape_feature import extract_shape_feature
from texture_feature import extract_texture_feature


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXCLUDED_CATEGORIES = {"human_being", "fire_extinguisher"}
METADATA_FILENAME = "dataset_metadata.csv"


def list_images(dataset_dir, excluded_categories=None):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    normalized_excluded = {
        name.strip().lower() for name in (excluded_categories or set()) if name.strip()
    }

    image_paths = []
    for path in dataset_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if path.parent.name.strip().lower() in normalized_excluded:
            continue

        image_paths.append(path)

    image_paths.sort()
    return image_paths


def _normalize_relpath(path_value):
    return str(path_value).replace("\\", "/").strip().lower()


def load_dataset_metadata(dataset_dir):
    dataset_path = Path(dataset_dir)
    metadata_path = dataset_path / METADATA_FILENAME
    if not metadata_path.exists():
        return {}

    metadata = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "instance_id", "view_label"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = ", ".join(sorted(required - set(reader.fieldnames or [])))
            raise ValueError(f"Metadata file is missing required columns: {missing}")

        for row in reader:
            image_path = (row.get("image_path") or "").strip()
            instance_id = (row.get("instance_id") or "").strip()
            view_label = (row.get("view_label") or "").strip().lower()

            if not image_path or not instance_id or not view_label:
                continue

            metadata[_normalize_relpath(image_path)] = {
                "instance_id": instance_id,
                "view_label": view_label,
            }

    return metadata


def _build_metadata_arrays(image_paths, dataset_dir, metadata):
    dataset_path = Path(dataset_dir)
    instance_ids = []
    view_labels = []
    annotated_count = 0

    for image_path in image_paths:
        rel_path = _normalize_relpath(image_path.relative_to(dataset_path))
        record = metadata.get(rel_path)
        if record is None:
            instance_ids.append("")
            view_labels.append("")
            continue

        instance_ids.append(record["instance_id"])
        view_labels.append(record["view_label"])
        annotated_count += 1

    return (
        np.array(instance_ids, dtype=object),
        np.array(view_labels, dtype=object),
        annotated_count > 0,
    )


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

    image_paths = list_images(dataset_dir, excluded_categories=EXCLUDED_CATEGORIES)
    if not image_paths:
        raise ValueError(f"No images found in dataset directory: {dataset_dir}")

    metadata = load_dataset_metadata(dataset_dir)

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

    kept_paths_array = [Path(path) for path in kept_paths]
    instance_ids, view_labels, has_complete_metadata = _build_metadata_arrays(
        kept_paths_array,
        dataset_dir,
        metadata,
    )

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
        instance_ids=instance_ids,
        view_labels=view_labels,
        has_complete_metadata=np.array([has_complete_metadata], dtype=bool),
    )

    print("Database saved:", output_path)
    return features, kept_paths, labels


def load_feature_database(npz_path="feature_database.npz"):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Feature database not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        result = {
            "features": data["features"],
            "color_features": data["color_features"] if "color_features" in data.files else None,
            "texture_features": data["texture_features"] if "texture_features" in data.files else None,
            "shape_features": data["shape_features"] if "shape_features" in data.files else None,
            "image_paths": data["image_paths"],
            "labels": data["labels"],
            "instance_ids": data["instance_ids"] if "instance_ids" in data.files else None,
            "view_labels": data["view_labels"] if "view_labels" in data.files else None,
            "has_complete_metadata": bool(data["has_complete_metadata"][0]) if "has_complete_metadata" in data.files else False,
        }
        return result


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
