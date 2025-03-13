from flask import Blueprint, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

# Define the Blueprint
bp = Blueprint('main', __name__)

# Define upload and result folders
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads')
RESULT_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'static', 'results')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Ensure a file was uploaded
        if 'video' not in request.files:
            return redirect(request.url)
        
        video = request.files['video']
        
        # Check for an empty filename
        if video.filename == '':
            return redirect(request.url)
        
        # Save the uploaded file
        if video:
            filename = secure_filename(video.filename)
            video_path = os.path.join(UPLOAD_FOLDER, filename)
            video.save(video_path)

            # Placeholder for prediction result (modify if needed)
            result_video_path = os.path.join(RESULT_FOLDER, 'result.mp4')

            return render_template('index.html', result_video='results/result.mp4')

    return render_template('index.html', result_video=None)
