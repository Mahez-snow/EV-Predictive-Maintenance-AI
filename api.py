from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# TEMP STORAGE (RAM)
latest_data = {
    "voltage": None,
    "current": None,
    "battery_temp": 35,        # Constant for demo
    "speed": None,
    "road": None,
    "target_distance": None,
    "charge_cycles": None,
    "load_cycles": None,
    "timestamp": None
}

# Raspberry Pi → Cloud
@app.route("/upload", methods=["POST"])
def upload_data():
    global latest_data
    data = request.get_json()

    latest_data["voltage"] = data.get("voltage")
    latest_data["current"] = data.get("current")
    latest_data["speed"] = data.get("speed")
    latest_data["road"] = data.get("road")

    latest_data["target_distance"] = data.get("target_distance")
    latest_data["charge_cycles"] = data.get("charge_cycles")
    latest_data["load_cycles"] = data.get("load_cycles")
    weight = data.get("load_cycles")  # hardware sends weight as load_cycles
    # Fixed battery temperature for stability
    latest_data["battery_temp"] = 35

    latest_data["timestamp"] = datetime.utcnow().isoformat()

    return jsonify({
        "status": "success",
        "message": "Hardware data received successfully"
    })

# Streamlit → Fetch latest
@app.route("/latest", methods=["GET"])
def get_latest_data():
    if latest_data["voltage"] is None:
        return jsonify({"status": "no_data"}), 404
    return jsonify(latest_data)

# Health check
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "API running"})

# Run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
