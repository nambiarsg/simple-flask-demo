# Simple Flask Demo

A minimal Flask API + static frontend to test local runs and exposing via Cloudflare Tunnel.

## Endpoints
- `GET /` — serves the static frontend
- `GET /api/time` — returns server UTC time (ISO 8601)
- `POST /api/echo` — echos back posted JSON
- `GET /health` — health check for uptime checks

## Run locally

### 1) Create and activate a virtual env (optional but recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Start the server (dev)
```bash
python app.py
```
Visit http://localhost:5000

### (Optional) Run with Gunicorn (prod-like)
```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers=2 --threads=4 --timeout=120
```

## Expose publicly (quick demo)
### Cloudflare Tunnel (no account quick mode)
```bash
npx cloudflared tunnel --url http://localhost:5000
# or if cloudflared is installed:
# cloudflared tunnel --url http://localhost:5000
```
You'll receive a public HTTPS URL you can share.

## Deploy later (Render / Railway / Cloud Run)
- `requirements.txt` and `Procfile` are included for Heroku-style platforms.
- Add a `Dockerfile` if you plan to use Cloud Run/Fly.io.

## Notes
- The app binds to `0.0.0.0` and reads `PORT` if provided by your platform.
- CORS is enabled for quick testing; tighten it before production.
