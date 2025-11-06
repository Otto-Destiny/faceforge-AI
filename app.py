# By Destiny Otto.

import io
import os
from datetime import datetime
from werkzeug.utils import secure_filename

from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image

# Import our face recognition code
from face_recognition import add_labels_to_image

# Starts Flask
app = Flask(__name__)

# Configuration for file uploads
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_image_info(image):
    """Extract metadata from image"""
    try:
        info = {
            'format': image.format,
            'mode': image.mode,
            'size': image.size,
            'width': image.width,
            'height': image.height
        }
        return info
    except Exception as e:
        return None


# Set the route to "/"
@app.route('/')
def home():
    return render_template("upload.html")


@app.route("/recognize", methods=["POST"])
def process_image():
    # Display an error message if no image found
    if "image" not in request.files:
        return "No image provided", 400

    # Get the file sent along with the request
    file = request.files["image"]

    # Check if file was actually selected
    if file.filename == '':
        return "No file selected", 400

    # Validate file extension
    if not allowed_file(file.filename):
        return "File type not allowed. Please upload an image file.", 400

    # Video also shows up as an image
    # we want to reject those as well
    if not file.mimetype.startswith("image/"):
        return "Image format not recognized", 400

    try:
        image_data = file.stream

        # Run our face recognition code!
        img_out = run_face_recognition(image_data)

        if img_out == Ellipsis:
            return "Image processing not enabled", 200
        else:
            # Our function returns something from matplotlib,
            # convert it to a web-friendly form and return it
            out_stream = matplotlib_to_bytes(img_out)
            
            # Add headers for better file handling
            response = send_file(
                out_stream, 
                mimetype="image/jpeg",
                as_attachment=False,
                download_name=f"recognized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response

    except Exception as e:
        # Log the error for debugging
        print(f"Error processing image: {str(e)}")
        return f"Error processing image: {str(e)}", 500


def run_face_recognition(image_data):
    # Open image_data with PIL
    input_image = Image.open(image_data)

    # Run our function on the PIL image
    img_out = add_labels_to_image(input_image)

    return img_out


def matplotlib_to_bytes(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="jpg", bbox_inches="tight")
    buffer.seek(0)
    return buffer


# Additional API endpoint for image validation (optional, for future enhancements)
@app.route("/validate", methods=["POST"])
def validate_image():
    """Validate image before processing - useful for client-side checks"""
    if "image" not in request.files:
        return jsonify({"valid": False, "error": "No image provided"}), 400
    
    file = request.files["image"]
    
    if file.filename == '':
        return jsonify({"valid": False, "error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"valid": False, "error": "File type not allowed"}), 400
    
    try:
        img = Image.open(file.stream)
        info = get_image_info(img)
        
        return jsonify({
            "valid": True,
            "info": info
        }), 200
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)}), 400


# Health check endpoint
@app.route("/health")
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "facial-recognition"
    }), 200


# Error handlers for better user experience
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size exceeded error"""
    return "File too large. Maximum size is 16MB.", 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors"""
    return "Internal server error. Please try again later.", 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return "Page not found", 404


if __name__ == "__main__":
    app.run(debug=True)
