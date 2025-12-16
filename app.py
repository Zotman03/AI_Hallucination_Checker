from flask import Flask, render_template, request, jsonify, send_file
import os
import subprocess
import sys
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
ALLOWED_EXTENSIONS = {'pdf'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('paper_txt', exist_ok=True)
os.makedirs('reference_messy_txt', exist_ok=True)
os.makedirs('reference_clean_txt', exist_ok=True)
os.makedirs('html_report', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Run txt_checker.py using the same Python interpreter
            # Ensure we pass the current environment and run from the correct directory
            print(f"Using Python: {sys.executable}")
            print(f"Environment: {os.environ.get('CONDA_DEFAULT_ENV', 'none')}")
            result = subprocess.run(
                [sys.executable, 'txt_checker.py', filepath],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env=os.environ.copy()
            )
            
            # Log output for debugging
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            print("Return code:", result.returncode)
            
            if result.returncode != 0:
                return jsonify({
                    'error': 'Processing failed',
                    'details': result.stderr,
                    'stdout': result.stdout
                }), 500

            file_name = filename.rsplit('.', 1)[0]
            report_path = f'html_report/{file_name}_report.html'
            
            if not os.path.exists(report_path):
                return jsonify({
                    'error': 'Report generation failed',
                    'details': 'HTML report was not created',
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }), 500
            
            return jsonify({
                'success': True,
                'report_url': f'/report/{file_name}',
                'filename': file_name,
                'message': 'Processing complete!'
            })
            
        except subprocess.TimeoutExpired:
            return jsonify({'error': 'Processing timeout'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/report/<filename>')
def view_report(filename):
    report_path = f'html_report/{filename}_report.html'
    if os.path.exists(report_path):
        return send_file(report_path)
    return "Report not found", 404

@app.route('/download/<filename>')
def download_report(filename):
    report_path = f'html_report/{filename}_report.html'
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True, download_name=f'{filename}_validation_report.html')
    return "Report not found", 404

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up all generated files"""
    try:
        folders = ['paper_txt', 'reference_messy_txt', 'reference_clean_txt', 'html_report', 'uploads']
        for folder in folders:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    file_path = os.path.join(folder, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        return jsonify({'success': True, 'message': 'Cleanup complete'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

