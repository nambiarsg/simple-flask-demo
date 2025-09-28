from flask import Flask, jsonify, request, send_from_directory
from datetime import datetime
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

@app.route('/')
def root():
    return send_from_directory(app.static_folder, 'index.html')

@app.get('/api/time')
def api_time():
    return jsonify({"now": datetime.utcnow().isoformat() + "Z"})

@app.post('/api/echo')
def api_echo():
    data = request.get_json(silent=True) or {}
    return jsonify({"echo": data})

@app.get('/health')
def health():
    return jsonify({"ok": True})

if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    app.run(host='0.0.0.0', port=port)
