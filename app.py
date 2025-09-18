#!/usr/bin/env python3
"""
Buddy Agent - Professional Web Interface
A clean, professional web application for document Q&A
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import asyncio
from datetime import datetime

# Add the buddy_agent directory to the Python path
current_dir = Path(__file__).parent
buddy_agent_dir = current_dir / "buddy_agent"
sys.path.insert(0, str(buddy_agent_dir))

# Import the Buddy Agent
from buddy_agent.sub_agents.rag_retriever.better_search import BetterSearchEngine

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'buddy_agent_secret_key_2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize search engine
search_engine = BetterSearchEngine(collection_name="web_documents")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path, file_type):
    """Extract text from uploaded file."""
    try:
        if file_type == 'pdf':
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n{page_text}\n"
                return text
        
        elif file_type == 'docx':
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        elif file_type == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        return ""
    
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return ""

@app.route('/')
def index():
    """Main page - document upload and Q&A interface."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    try:
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            # Extract text from file
            file_type = filename.rsplit('.', 1)[1].lower()
            text = extract_text_from_file(file_path, file_type)
            
            if not text.strip():
                flash('Could not extract text from file', 'error')
                return redirect(url_for('index'))
            
            # Add document to search engine
            document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            success = search_engine.add_document(
                document_id=document_id,
                text=text,
                metadata={
                    "filename": filename,
                    "file_type": file_type,
                    "upload_time": datetime.now().isoformat(),
                    "file_size": len(text)
                }
            )
            
            if success:
                flash(f'Document "{filename}" uploaded and processed successfully!', 'success')
                # Clean up uploaded file
                os.remove(file_path)
            else:
                flash('Failed to process document', 'error')
                return redirect(url_for('index'))
        
        else:
            flash('Invalid file type. Please upload PDF, DOCX, or TXT files.', 'error')
            return redirect(url_for('index'))
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        flash('An error occurred while processing the file', 'error')
        return redirect(url_for('index'))
    
    return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle question asking."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please enter a question'}), 400
        
        # Search for relevant content
        results = search_engine.search_similar(question, n_results=3, threshold=0.01)
        
        if not results:
            return jsonify({
                'answer': 'I couldn\'t find relevant information in the uploaded documents to answer your question.',
                'sources': [],
                'confidence': 0
            })
        
        # Generate answer from top result
        top_result = results[0]
        answer = top_result['chunk_text']
        confidence = min(top_result['similarity_score'] * 100, 100)
        
        # Prepare sources
        sources = []
        for result in results:
            sources.append({
                'text': result['chunk_text'][:200] + '...' if len(result['chunk_text']) > 200 else result['chunk_text'],
                'score': round(result['similarity_score'], 3),
                'document': result['metadata'].get('filename', 'Unknown')
            })
        
        return jsonify({
            'answer': answer,
            'sources': sources,
            'confidence': round(confidence, 1)
        })
    
    except Exception as e:
        logger.error(f"Question error: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your question'}), 500

@app.route('/status')
def status():
    """Get system status and document count."""
    try:
        stats = search_engine.get_collection_stats()
        return jsonify({
            'documents': stats.get('total_documents', 0),
            'chunks': stats.get('total_chunks', 0),
            'status': 'ready'
        })
    except Exception as e:
        logger.error(f"Status error: {str(e)}")
        return jsonify({'error': 'Failed to get status'}), 500

@app.route('/clear', methods=['POST'])
def clear_documents():
    """Clear all documents."""
    try:
        search_engine.clear_collection()
        flash('All documents cleared successfully!', 'success')
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Clear error: {str(e)}")
        return jsonify({'error': 'Failed to clear documents'}), 500

if __name__ == '__main__':
    print("🚀 Starting Buddy Agent Web Interface...")
    print("=" * 50)
    print("📱 Open your browser and go to: http://localhost:5000")
    print("📚 Upload documents and start asking questions!")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
