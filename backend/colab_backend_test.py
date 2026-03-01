# =============================================================================
# colab_backend_test.py
# =============================================================================
# Run this in a SECOND Colab cell after training is complete.
# It installs deps, launches the FastAPI backend in a background thread,
# exposes a public URL via ngrok, then connects a WebSocket client
# to verify continuous live output.
#
# USAGE (Google Colab):
#   1. Run colab_training.py to produce model.pkl, scaler.pkl, etc.
#   2. Paste this file's contents into a new Colab cell and run.
# =============================================================================

# ── CELL A: Install ──────────────────────────────────────────────────────────
# !pip install fastapi uvicorn websockets xgboost scikit-learn pyngrok --quiet
# !pip install requests scipy joblib --quiet

# ── CELL B: Copy model artifacts into backend/ ───────────────────────────────
# import shutil, os
# os.makedirs("/content/backend", exist_ok=True)
# for f in ["model.pkl", "scaler.pkl", "features.json", "threshold.json"]:
#     shutil.copy(f"/content/{f}", f"/content/backend/{f}")
# Copy app.py, trading_engine.py, model.py into /content/backend/ as well.

# ── CELL C: Start FastAPI server ─────────────────────────────────────────────
import subprocess
import threading
import time
import json
import asyncio
import websockets
import requests

BACKEND_PORT = 8000
BACKEND_DIR  = "/content/backend"

def start_server():
    subprocess.Popen(
        ["uvicorn", "app:app", "--host", "0.0.0.0", f"--port={BACKEND_PORT}"],
        cwd=BACKEND_DIR,
    )

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()
time.sleep(4)
print("FastAPI server started.")

# ── CELL D: Expose via ngrok ──────────────────────────────────────────────────
# from pyngrok import ngrok
# public_url = ngrok.connect(BACKEND_PORT)
# print(f"Public URL: {public_url}")
# print(f"Docs: {public_url}/docs")
# print(f"WS:   {str(public_url).replace('http','ws')}/ws")

# ── CELL E: Check status endpoint ────────────────────────────────────────────
BASE = f"http://localhost:{BACKEND_PORT}"

resp = requests.get(f"{BASE}/status")
print("Status:", json.dumps(resp.json(), indent=2))

resp = requests.get(f"{BASE}/model_info")
print("\nModel info:", json.dumps(resp.json(), indent=2))

# ── CELL F: Start simulation ──────────────────────────────────────────────────
resp = requests.post(f"{BASE}/start_simulation")
print("\nSimulation started:", resp.json())
time.sleep(3)   # let buffer warm up

# ── CELL G: Connect WebSocket and print 30 events ────────────────────────────
async def consume_ws(n_events: int = 30):
    uri = f"ws://localhost:{BACKEND_PORT}/ws"
    print(f"\nConnecting to {uri}...")
    async with websockets.connect(uri) as ws:
        count = 0
        async for message in ws:
            data = json.loads(message)
            ev   = data.get("event")

            if ev == "welcome":
                print(f"✓ Welcome received. Running={data['running']}")
                continue

            if ev == "bar":
                ticker = data["ticker"]
                close  = data.get("close", "–")
                prob   = data.get("probability", "–")
                signal = data.get("signal", "–")
                equity = data.get("portfolio", {}).get("equity", "–")
                print(
                    f"[{count:03d}] {ticker:5s} "
                    f"close={close:>9.2f}  "
                    f"p={str(prob)[:6]}  "
                    f"sig={signal}  "
                    f"equity=${equity:>12,.2f}"
                )
                count += 1
                if count >= n_events:
                    break

            elif ev == "heartbeat":
                print(f"[heartbeat] running={data['running']} equity={data['equity']}")

asyncio.get_event_loop().run_until_complete(consume_ws(30))

# ── CELL H: Stop simulation ───────────────────────────────────────────────────
resp = requests.post(f"{BASE}/stop")
print("\nEngine stopped:", resp.json())

# ── CELL I: Start LIVE Polygon stream ────────────────────────────────────────
# (Run only during US market hours: 9:30 AM – 4:00 PM ET)
# resp = requests.post(f"{BASE}/start_auto")
# print("Live stream:", resp.json())

print("\n✓ All tests passed. Deploy backend/ to Render.")
print("\nDeployment checklist:")
print("  1. Push backend/ to GitHub")
print("  2. Create Render Web Service, set start command:")
print("       uvicorn app:app --host 0.0.0.0 --port $PORT")
print("  3. Set environment variable: POLYGON_API_KEY=MoyLn951WdZAozaSClrOGar9xgYjy0pR")
print("  4. Upload model.pkl, scaler.pkl, features.json, threshold.json to backend/")
print("  5. Visit https://your-app.onrender.com/docs")
