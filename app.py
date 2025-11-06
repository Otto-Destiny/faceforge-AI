"""
Flask web application for facial recognition image processing.
Routes provided:
- GET  /            -> serves upload.html via templates
- POST /recognize   -> accept form/file upload named "image", returns annotated JPEG
- POST /validate    -> quick validation endpoint (returns JSON)
- GET  /health      -> service health JSON
Error handlers for 413, 404, 500 are included.

Dependencies: Flask, Pillow (PIL), matplotlib, io, datetime
It imports add_labels_to_image from face_recognition (keeps same contract).
"""

import io
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional

from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image

# Import face recognition implementation
from face_recognition import add_labels_to_image

# -------------------------
# Application configuration
# -------------------------
SERVICE_NAME = "facial-recognition"
MAX_BYTES = 16 * 1024 * 1024  # 16 MB
ALLOWED_SUFFIXES = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

app = Flask(__name__)

# Apply upload size limit
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES


# -------------------------
# Helper utilities
# -------------------------
def _extension_ok(filename: str) -> bool:
    """Return True when filename has an allowed extension."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_SUFFIXES


def _is_image_mime(mimetype: Optional[str]) -> bool:
    """Simple check that the upload surface-level mime type looks like an image."""
    if not mimetype:
        return False
    return mimetype.split("/")[0].lower() == "image"


def _extract_image_meta(img: Image.Image) -> Dict:
    """Collect basic metadata about a PIL image (keeps different key names)."""
    try:
        return {
            "format": img.format,
            "mode": img.mode,
            "dimensions": {"width": img.width, "height": img.height},
            "size": img.size,
        }
    except Exception:
        return {}


def _figure_to_bytes(fig, fmt: str = "jpeg") -> io.BytesIO:
    """
    Convert a Matplotlib figure to an in-memory bytes buffer.
    Closes the figure afterwards to avoid memory accumulation.
    """
    buf = io.BytesIO()
    # use JPEG format for browser compatibility
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    try:
        # close the figure to free resources
        import matplotlib.pyplot as _plt

        _plt.close(fig)
    except Exception:
        pass
    return buf


# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def index():
    """Render the upload page (templates/upload.html expected)."""
    return render_template("upload.html")


@app.route("/recognize", methods=["POST"])
def recognize():
    """
    Main image processing endpoint.
    Expects form-file under key "image".
    Returns the annotated image as image/jpeg.
    """
    if "image" not in request.files:
        return "No image provided", 400

    upload = request.files["image"]

    if not upload or upload.filename == "":
        return "No file selected", 400

    # Basic extension check
    if not _extension_ok(upload.filename):
        return "File type not allowed. Please upload an image file.", 400

    # Basic mimetype check
    if not _is_image_mime(upload.mimetype):
        return "Image format not recognized", 400

    try:
        # Open the uploaded stream with PIL (this will validate the image)
        pil_img = Image.open(upload.stream).convert("RGB")

        # Run the face recognition/annotation pipeline (from face_recognition)
        annotated_fig = add_labels_to_image(pil_img)

        # Convert Matplotlib figure to bytes and send as response
        img_buffer = _figure_to_bytes(annotated_fig, fmt="jpeg")

        # Create response using Flask's send_file
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"recognized_{timestamp}.jpg"

        resp = send_file(
            img_buffer,
            mimetype="image/jpeg",
            as_attachment=False,
            download_name=filename,
        )

        # Recommended cache headers for dynamic images
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"

        return resp

    except Exception as exc:
        # Print server-side error for diagnostics (stdout or host logs)
        print(f"[ERROR] processing upload: {exc}")
        return f"Error processing image: {str(exc)}", 500


@app.route("/validate", methods=["POST"])
def validate():
    """
    Lightweight validation endpoint.
    Returns JSON describing whether the provided file looks like a valid image.
    """
    if "image" not in request.files:
        return jsonify({"valid": False, "error": "No image provided"}), 400

    f = request.files["image"]

    if not f or f.filename == "":
        return jsonify({"valid": False, "error": "No file selected"}), 400

    if not _extension_ok(f.filename):
        return jsonify({"valid": False, "error": "File type not allowed"}), 400

    try:
        img = Image.open(f.stream)
        meta = _extract_image_meta(img)
        return jsonify({"valid": True, "info": meta}), 200
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    """Simple health check for orchestration or uptime checks."""
    return (
        jsonify(
            {
                "service": SERVICE_NAME,
                "status": "healthy",
                "time": datetime.now(timezone.utc).isoformat(),
            }
        ),
        200,
    )


# -------------------------
# Error handlers
# -------------------------
@app.errorhandler(413)
def handle_too_large(_err):
    return "File too large. Maximum size is 16MB.", 413


@app.errorhandler(404)
def handle_not_found(_err):
    return "Page not found", 404


@app.errorhandler(500)
def handle_server_error(_err):
    return "Internal server error. Please try again later.", 500


# -------------------------
# Run the app (development)
# -------------------------
if __name__ == "__main__":
    # Note: in production use a WSGI server (gunicorn/uwsgi).
    app.run(host="127.0.0.1", port=5000, debug=True)

'''
© 2025 Destiny Otto
'''
