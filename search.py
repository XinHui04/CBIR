import numpy as np

from feature_database import extract_feature_parts, load_feature_database

EXCLUDED_LABELS = {"human_being", "fire_extinguisher"}
FRONT_VIEW_LABELS = {"front", "canonical", "best_front"}


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


def _filter_database(database, use_color, use_texture, use_shape, category):
    features = _select_features(database, use_color, use_texture, use_shape)
    image_paths = database["image_paths"]
    labels = database["labels"]
    instance_ids = database["instance_ids"]
    view_labels = database["view_labels"]

    normalized_labels = np.array([str(label).strip().lower() for label in labels], dtype=object)
    exclusion_mask = ~np.isin(normalized_labels, list(EXCLUDED_LABELS))

    features = features[exclusion_mask]
    image_paths = image_paths[exclusion_mask]
    labels = labels[exclusion_mask]
    instance_ids = instance_ids[exclusion_mask] if instance_ids is not None else None
    view_labels = view_labels[exclusion_mask] if view_labels is not None else None

    if len(labels) == 0:
        raise ValueError("No searchable images available after category exclusion")

    if category:
        category_mask = np.array(labels) == category
        features = features[category_mask]
        image_paths = image_paths[category_mask]
        labels = labels[category_mask]
        instance_ids = instance_ids[category_mask] if instance_ids is not None else None
        view_labels = view_labels[category_mask] if view_labels is not None else None

        if len(labels) == 0:
            raise ValueError(f"No images found for category '{category}'")

    return {
        "features": features,
        "image_paths": image_paths,
        "labels": labels,
        "instance_ids": instance_ids,
        "view_labels": view_labels,
    }


def _build_results(indices, scores, image_paths, labels, score_key, extra_fields=None):
    results = []
    for idx, score in zip(indices, scores):
        item = {
            "image_path": str(image_paths[idx]),
            "label": str(labels[idx]),
            score_key: float(score),
        }
        if extra_fields:
            item.update({key: values[idx] for key, values in extra_fields.items()})
        results.append(item)
    return results


def _compute_scores(metric, db_features, query_feature):
    if metric == "cosine":
        query_norm = np.linalg.norm(query_feature)
        db_norms = np.linalg.norm(db_features, axis=1)
        scores = (db_features @ query_feature) / ((query_norm * db_norms) + 1e-12)
        return scores

    if metric == "euclidean":
        return np.linalg.norm(db_features - query_feature, axis=1)

    raise ValueError("metric must be either 'cosine' or 'euclidean'")


def _rank_indices(metric, scores, top_k):
    k = _validate_top_k(top_k, len(scores))
    if metric == "cosine":
        return np.argsort(scores)[::-1][:k]
    return np.argsort(scores)[:k]


def _best_match_index(metric, scores):
    if metric == "cosine":
        return int(np.argmax(scores))
    return int(np.argmin(scores))


def _find_front_candidates(instance_ids, view_labels, matched_instance_id):
    if instance_ids is None or view_labels is None:
        return np.array([], dtype=int)

    instance_mask = np.array(instance_ids, dtype=object) == matched_instance_id
    front_mask = np.isin(
        np.array([str(label).strip().lower() for label in view_labels], dtype=object),
        list(FRONT_VIEW_LABELS),
    )
    return np.where(instance_mask & front_mask)[0]


def _search_similarity(
    query_image_path,
    metric,
    database_path,
    top_k,
    size,
    color_bins,
    category,
    use_color,
    use_texture,
    use_shape,
):
    database = load_feature_database(database_path)
    filtered = _filter_database(database, use_color, use_texture, use_shape, category)
    query_feature = _build_query_feature(
        query_image_path,
        size,
        color_bins,
        use_color,
        use_texture,
        use_shape,
    )
    scores = _compute_scores(metric, filtered["features"], query_feature)
    ranked = _rank_indices(metric, scores, top_k)
    score_key = "cosine_similarity" if metric == "cosine" else "euclidean_distance"
    return _build_results(
        ranked,
        scores[ranked],
        filtered["image_paths"],
        filtered["labels"],
        score_key,
    )


def _search_canonical_front(
    query_image_path,
    metric,
    database_path,
    size,
    color_bins,
    category,
    use_color,
    use_texture,
    use_shape,
):
    database = load_feature_database(database_path)
    if not database["has_complete_metadata"]:
        raise ValueError(
            "Canonical front-view mode needs Dataset/dataset_metadata.csv with image_path, instance_id, and view_label columns."
        )

    filtered = _filter_database(database, use_color, use_texture, use_shape, category)
    annotated_mask = np.array(
        [
            bool(str(instance_id).strip()) and bool(str(view_label).strip())
            for instance_id, view_label in zip(filtered["instance_ids"], filtered["view_labels"])
        ],
        dtype=bool,
    )
    if not np.any(annotated_mask):
        raise ValueError(
            "Canonical front-view mode needs annotated images in Dataset/dataset_metadata.csv for the categories you want to search."
        )

    filtered = {
        "features": filtered["features"][annotated_mask],
        "image_paths": filtered["image_paths"][annotated_mask],
        "labels": filtered["labels"][annotated_mask],
        "instance_ids": filtered["instance_ids"][annotated_mask],
        "view_labels": filtered["view_labels"][annotated_mask],
    }

    query_feature = _build_query_feature(
        query_image_path,
        size,
        color_bins,
        use_color,
        use_texture,
        use_shape,
    )
    scores = _compute_scores(metric, filtered["features"], query_feature)
    matched_index = _best_match_index(metric, scores)
    matched_instance_id = str(filtered["instance_ids"][matched_index])
    front_indices = _find_front_candidates(
        filtered["instance_ids"],
        filtered["view_labels"],
        matched_instance_id,
    )

    if len(front_indices) == 0:
        raise ValueError(
            f"Matched instance '{matched_instance_id}' has no front view in dataset_metadata.csv."
        )

    front_scores = scores[front_indices]
    ranked_front_indices = front_indices[_rank_indices(metric, front_scores, len(front_indices))]
    score_key = "cosine_similarity" if metric == "cosine" else "euclidean_distance"
    return _build_results(
        ranked_front_indices,
        scores[ranked_front_indices],
        filtered["image_paths"],
        filtered["labels"],
        score_key,
        extra_fields={
            "instance_id": filtered["instance_ids"],
            "view_label": filtered["view_labels"],
            "matched_instance_id": np.array(
                [matched_instance_id] * len(filtered["image_paths"]),
                dtype=object,
            ),
        },
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
    retrieval_mode="similarity",
):
    metric_name = metric.strip().lower()
    retrieval_mode = retrieval_mode.strip().lower()

    if retrieval_mode == "canonical_front":
        return _search_canonical_front(
            query_image_path,
            metric_name,
            database_path,
            size,
            color_bins,
            category,
            use_color,
            use_texture,
            use_shape,
        )

    return _search_similarity(
        query_image_path,
        metric_name,
        database_path,
        top_k,
        size,
        color_bins,
        category,
        use_color,
        use_texture,
        use_shape,
    )
