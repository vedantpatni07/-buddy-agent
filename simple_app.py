#!/usr/bin/env python3
"""
Buddy Agent - Simple Web Interface
Basic localhost interface for document Q&A
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
import pdfplumber
from docx import Document

# Add the buddy_agent directory to the Python path
current_dir = Path(__file__).parent
buddy_agent_dir = current_dir / "buddy_agent"
sys.path.insert(0, str(buddy_agent_dir))

# Import the search engine
from buddy_agent.sub_agents.rag_retriever.better_search import BetterSearchEngine

# Initialize Flask app
app = Flask(__name__)

# Initialize search engine
search_engine = BetterSearchEngine(collection_name="documents")

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Buddy Agent - Document Q&A</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .upload-section { margin: 20px 0; padding: 20px; border: 2px dashed #ccc; border-radius: 5px; }
        .qa-section { margin: 20px 0; }
        input[type="file"] { margin: 10px 0; }
        input[type="text"] { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .answer { background: #e9f7ef; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #28a745; }
        .error { background: #f8d7da; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #dc3545; }
        .status { background: #d1ecf1; padding: 10px; margin: 10px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Buddy Agent - Document Q&A</h1>
        
        <div class="upload-section">
            <h3>📄 Upload Document</h3>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="file" name="file" accept=".pdf,.docx,.txt" required>
                <button type="submit">Upload & Process</button>
            </form>
            <div id="uploadStatus"></div>
        </div>
        
        <div class="qa-section">
            <h3>❓ Ask Questions</h3>
            <input type="text" id="question" placeholder="Type your question here..." />
            <button onclick="askQuestion()">Ask</button>
            <div id="answer"></div>
        </div>
        
        <div class="status">
            <strong>Status:</strong> <span id="status">Ready</span> | 
            <strong>Documents:</strong> <span id="docCount">0</span>
        </div>
    </div>

    <script>
        // Upload file
        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData();
            const file = document.getElementById('file').files[0];
            formData.append('file', file);
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('uploadStatus').innerHTML = '<div class="answer">✅ ' + data.message + '</div>';
                    updateStatus();
                } else {
                    document.getElementById('uploadStatus').innerHTML = '<div class="error">❌ ' + data.message + '</div>';
                }
            });
        });
        
        // Ask question
        function askQuestion() {
            const question = document.getElementById('question').value;
            if (!question) return;
            
            fetch('/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: question})
            })
            .then(response => response.json())
            .then(data => {
                if (data.answer) {
                    document.getElementById('answer').innerHTML = '<div class="answer"><strong>Answer:</strong><br>' + data.answer + '</div>';
                } else {
                    document.getElementById('answer').innerHTML = '<div class="error">❌ ' + data.error + '</div>';
                }
            });
        }
        
        // Update status
        function updateStatus() {
            fetch('/status')
            .then(response => response.json())
            .then(data => {
                document.getElementById('docCount').textContent = data.documents;
                document.getElementById('status').textContent = 'Ready';
            });
        }
        
        // Allow Enter key for questions
        document.getElementById('question').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                askQuestion();
            }
        });
        
        // Initial status update
        updateStatus();
    </script>
</body>
</html>
"""

def extract_text(file_path, file_type):
    """Extract text from file."""
    try:
        if file_type == 'pdf':
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        elif file_type == 'docx':
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        elif file_type == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

@app.route('/')
def index():
    """Main page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file selected'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        # Save file temporarily
        filename = file.filename
        file_path = f"temp_{filename}"
        file.save(file_path)
        
        # Extract text
        file_type = filename.split('.')[-1].lower()
        text = extract_text(file_path, file_type)
        
        # Clean up temp file
        os.remove(file_path)
        
        if not text.strip():
            return jsonify({'success': False, 'message': 'Could not extract text from file'})
        
        # Add to search engine
        success = search_engine.add_document(
            document_id=f"doc_{filename}",
            text=text,
            metadata={"filename": filename}
        )
        
        if success:
            return jsonify({'success': True, 'message': f'Document "{filename}" processed successfully!'})
        else:
            return jsonify({'success': False, 'message': 'Failed to process document'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/ask', methods=['POST'])
def ask():
    """Handle questions."""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'Please enter a question'})
        
        # Search for answer
        results = search_engine.search_similar(question, n_results=1, threshold=0.01)
        
        if not results:
            return jsonify({'answer': 'No relevant information found in the documents.'})
        
        answer = results[0]['chunk_text']
        return jsonify({'answer': answer})
    
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'})

@app.route('/status')
def status():
    """Get status."""
    try:
        stats = search_engine.get_collection_stats()
        return jsonify({
            'documents': stats.get('total_documents', 0),
            'status': 'ready'
        })
    except Exception as e:
        return jsonify({'documents': 0, 'status': 'error'})

if __name__ == '__main__':
    print("🚀 Starting Buddy Agent...")
    print("📱 Open: http://localhost:5000")
    print("📄 Upload a document and start asking questions!")
    app.run(debug=True, host='0.0.0.0', port=5000)
