import numpy as np

from feature_database import extract_feature_parts, load_feature_database


def _validate_top_k(top_k, total_items):
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return min(top_k, total_items)


def _select_features(database, use_color, use_texture, use_shape):
    if not (use_color or use_texture or use_shape):
        raise ValueError("At least one feature type must be selected")

    if (
        database["color_features"] is None
        or database["texture_features"] is None
        or database["shape_features"] is None
    ):
        raise ValueError(
            "Feature database is outdated. Rebuild it so category and feature filters can be used."
        )

    selected = []
    if use_color:
        selected.append(database["color_features"])
    if use_texture:
        selected.append(database["texture_features"])
    if use_shape:
        selected.append(database["shape_features"])
    return np.concatenate(selected, axis=1).astype(np.float32)


def _build_query_feature(query_image_path, size, color_bins, use_color, use_texture, use_shape):
    parts = extract_feature_parts(
        query_image_path,
        size=size,
        color_bins=color_bins,
    )

    selected = []
    if use_color:
        selected.append(parts["color"])
    if use_texture:
        selected.append(parts["texture"])
    if use_shape:
        selected.append(parts["shape"])

    if not selected:
        raise ValueError("At least one feature type must be selected")

    return np.concatenate(selected).astype(np.float32)


def _apply_category_filter(features, image_paths, labels, category):
    if not category:
        return features, image_paths, labels

    mask = np.array(labels) == category
    filtered_features = features[mask]
    filtered_paths = image_paths[mask]
    filtered_labels = labels[mask]

    if len(filtered_labels) == 0:
        raise ValueError(f"No images found for category '{category}'")

    return filtered_features, filtered_paths, filtered_labels


def _build_results(indices, scores, image_paths, labels, score_key):
    results = []
    for idx, score in zip(indices, scores):
        results.append(
            {
                "image_path": str(image_paths[idx]),
                "label": str(labels[idx]),
                score_key: float(score),
            }
        )
    return results


def search_with_cosine(
    query_image_path,
    database_path="feature_database.npz",
    top_k=5,
    size=(256, 256),
    color_bins=(8, 12, 3),
    category=None,
    use_color=True,
    use_texture=True,
    use_shape=True,
):
    database = load_feature_database(database_path)
    db_features = _select_features(database, use_color, use_texture, use_shape)
    image_paths = database["image_paths"]
    labels = database["labels"]

    db_features, image_paths, labels = _apply_category_filter(
        db_features,
        image_paths,
        labels,
        category,
    )

    query_feature = _build_query_feature(
        query_image_path,
        size,
        color_bins,
        use_color,
        use_texture,
        use_shape,
    )

    query_norm = np.linalg.norm(query_feature)
    db_norms = np.linalg.norm(db_features, axis=1)
    cosine_scores = (db_features @ query_feature) / ((query_norm * db_norms) + 1e-12)

    k = _validate_top_k(top_k, len(cosine_scores))
    sorted_indices = np.argsort(cosine_scores)[::-1][:k]
    return _build_results(
        sorted_indices,
        cosine_scores[sorted_indices],
        image_paths,
        labels,
        "cosine_similarity",
    )


def search_with_euclidean(
    query_image_path,
    database_path="feature_database.npz",
    top_k=5,
    size=(256, 256),
    color_bins=(8, 12, 3),
    category=None,
    use_color=True,
    use_texture=True,
    use_shape=True,
):
    database = load_feature_database(database_path)
    db_features = _select_features(database, use_color, use_texture, use_shape)
    image_paths = database["image_paths"]
    labels = database["labels"]

    db_features, image_paths, labels = _apply_category_filter(
        db_features,
        image_paths,
        labels,
        category,
    )

    query_feature = _build_query_feature(
        query_image_path,
        size,
        color_bins,
        use_color,
        use_texture,
        use_shape,
    )

    distances = np.linalg.norm(db_features - query_feature, axis=1)
    k = _validate_top_k(top_k, len(distances))
    sorted_indices = np.argsort(distances)[:k]
    return _build_results(
        sorted_indices,
        distances[sorted_indices],
        image_paths,
        labels,
        "euclidean_distance",
    )


def search_images(
    query_image_path,
    metric="cosine",
    database_path="feature_database.npz",
    top_k=5,
    size=(256, 256),
    color_bins=(8, 12, 3),
    category=None,
    use_color=True,
    use_texture=True,
    use_shape=True,
):
    metric_name = metric.strip().lower()

    if metric_name == "cosine":
        return search_with_cosine(
            query_image_path,
            database_path,
            top_k,
            size,
            color_bins,
            category,
            use_color,
            use_texture,
            use_shape,
        )

    if metric_name == "euclidean":
        return search_with_euclidean(
            query_image_path,
            database_path,
            top_k,
            size,
            color_bins,
            category,
            use_color,
            use_texture,
            use_shape,
        )

    raise ValueError("metric must be either 'cosine' or 'euclidean'")
