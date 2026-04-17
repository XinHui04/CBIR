from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from feature_database import (
    build_feature_database,
    extract_feature_parts,
    load_feature_database,
    list_images,
)
from search import search_images

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATASET_DIR = BASE_DIR / "Dataset"
DB_PATH = BASE_DIR / "feature_database.npz"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

app = Flask(__name__)
app.config["SECRET_KEY"] = "cbir-demo-secret"
UPLOAD_DIR.mkdir(exist_ok=True)


def is_allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def get_available_categories():
    labels = sorted({path.parent.name for path in list_images(str(DATASET_DIR))})
    return labels


def ensure_feature_database() -> None:
    needs_rebuild = not DB_PATH.exists()

    if not needs_rebuild:
        try:
            database = load_feature_database(str(DB_PATH))
            needs_rebuild = any(
                database[key] is None
                for key in ("color_features", "texture_features", "shape_features")
            )
            if not needs_rebuild:
                dataset_images = list_images(str(DATASET_DIR))
                if dataset_images:
                    sample_parts = extract_feature_parts(str(dataset_images[0]))
                    expected_dims = {
                        "color_features": sample_parts["color"].shape[0],
                        "texture_features": sample_parts["texture"].shape[0],
                        "shape_features": sample_parts["shape"].shape[0],
                    }
                    for key, expected_dim in expected_dims.items():
                        current = database[key]
                        if current is None or current.ndim != 2 or current.shape[1] != expected_dim:
                            needs_rebuild = True
                            break
        except Exception:
            needs_rebuild = True

    if needs_rebuild:
        build_feature_database(dataset_dir=str(DATASET_DIR), output_path=str(DB_PATH))


def build_image_url(image_path: str) -> str:
    target = Path(image_path)
    if not target.is_absolute():
        target = (BASE_DIR / target).resolve()

    if not target.exists():
        return ""

    try:
        target.relative_to(BASE_DIR)
    except ValueError:
        return ""

    relative = str(target.relative_to(BASE_DIR)).replace("\\", "/")
    return url_for("serve_file", relative_path=relative)


@app.route("/file/<path:relative_path>")
def serve_file(relative_path: str):
    safe_target = (BASE_DIR / relative_path).resolve()

    try:
        safe_target.relative_to(BASE_DIR)
    except ValueError:
        return "Not Found", 404

    if not safe_target.exists() or not safe_target.is_file():
        return "Not Found", 404

    return send_file(safe_target)


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query_image_url = ""
    selected_metric = "cosine"
    selected_top_k = 8
    selected_category = "all"
    selected_features = {
        "color": True,
        "texture": True,
        "shape": True,
    }
    available_categories = get_available_categories()

    if request.method == "POST":
        selected_metric = request.form.get("metric", "cosine").strip().lower()
        selected_category = request.form.get("category", "all").strip()

        try:
            selected_top_k = int(request.form.get("top_k", 8))
        except ValueError:
            selected_top_k = 8

        selected_features = {
            "color": request.form.get("use_color") == "on",
            "texture": request.form.get("use_texture") == "on",
            "shape": request.form.get("use_shape") == "on",
        }

        if not any(selected_features.values()):
            flash("Select at least one feature type: color, texture, or shape.")
            return redirect(url_for("index"))

        query_file = request.files.get("query_image")
        if not query_file or query_file.filename == "":
            flash("Please upload a query image.")
            return redirect(url_for("index"))

        if not is_allowed_file(query_file.filename):
            flash("Unsupported file type. Please use JPG, PNG, BMP, TIFF, or WEBP.")
            return redirect(url_for("index"))

        filename = secure_filename(query_file.filename)
        saved_path = UPLOAD_DIR / filename
        query_file.save(saved_path)

        query_image_url = url_for("serve_file", relative_path=f"uploads/{filename}")

        try:
            ensure_feature_database()
            raw_results = search_images(
                query_image_path=str(saved_path),
                metric=selected_metric,
                database_path=str(DB_PATH),
                top_k=selected_top_k,
                category=None if selected_category == "all" else selected_category,
                use_color=selected_features["color"],
                use_texture=selected_features["texture"],
                use_shape=selected_features["shape"],
            )

            for rank, item in enumerate(raw_results, start=1):
                score = item.get("cosine_similarity", item.get("euclidean_distance", 0.0))
                results.append(
                    {
                        "rank": rank,
                        "label": item["label"],
                        "image_url": build_image_url(item["image_path"]),
                        "score": score,
                        "is_cosine": "cosine_similarity" in item,
                    }
                )
        except Exception as exc:
            flash(f"Search failed: {exc}")

    return render_template(
        "index.html",
        results=results,
        query_image_url=query_image_url,
        selected_metric=selected_metric,
        selected_top_k=selected_top_k,
        selected_category=selected_category,
        selected_features=selected_features,
        available_categories=available_categories,
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)