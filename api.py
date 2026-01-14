from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)  # allow Streamlit / Pi access

# -----------------------------
# TEMPORARY STORAGE (RAM)
# -----------------------------
latest_data = {
    "voltage": None,
    "current": None,
    "battery_temp": None,
    "speed": None,
    "road": None,
    "timestamp": None
}

# -----------------------------
# Raspberry Pi sends data here
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload_data():
    global latest_data

    data = request.get_json()

    latest_data["voltage"] = data.get("voltage")
    latest_data["current"] = data.get("current")
    latest_data["battery_temp"] = data.get("battery_temp")
    latest_data["speed"] = data.get("speed")
    latest_data["road"] = data.get("road")
    latest_data["timestamp"] = datetime.utcnow().isoformat()

    return jsonify({
        "status": "success",
        "message": "Data received successfully"
    })

# -----------------------------
# Streamlit fetches data here
# -----------------------------
@app.route("/latest", methods=["GET"])
def get_latest_data():
    if latest_data["voltage"] is None:
        return jsonify({"status": "no_data"}), 404

    return jsonify(latest_data)

# -----------------------------
# Health check
# -----------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "API running"})

# -----------------------------
# Run locally
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
