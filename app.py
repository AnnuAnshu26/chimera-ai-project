import os
import joblib
import pandas as pd
import json
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from haversine import haversine, Unit

app = Flask(__name__, static_folder='frontend')
CORS(app)

try:
    model = joblib.load('chimera_model.pkl')
    all_assets_df = pd.read_csv('master_assets.csv')
    with open('model_columns.json', 'r') as f:
        MODEL_COLUMNS = json.load(f)
    print("✅ Scikit-learn model and asset database loaded.")
except FileNotFoundError:
    print("❌ ERROR: Missing model files. Did you download all 3 files from Colab?")
    model = None
    all_assets_df = pd.DataFrame()
    MODEL_COLUMNS = []

def get_dummy_action_plan(direct_failures, cascade_failures, clicked_point=None, radius_km=None):
    origin_text = ""
    if clicked_point:
        origin_text = f" (origin: {clicked_point[0]:.4f}, {clicked_point[1]:.4f})"
    radius_text = f" radius={radius_km:.1f} km" if radius_km else ""
    plan = f"""
AI Simulation Complete{origin_text}{radius_text}. {len(direct_failures) + len(cascade_failures)} assets at risk.

• PRIORITY 1: Dispatch emergency generators and mobile units to all {len(cascade_failures)} predicted hospital failures.
• PRIORITY 2: Secure the {len(direct_failures)} failing substations to prevent further damage.
• PRIORITY 3: Reroute all non-emergency traffic away from affected zones.
"""
    return plan

@app.route("/api/predict-failure", methods=['POST'])
def predict_failure():
    """
    Expects optional JSON body: { "lat": <float>, "lng": <float>, "radius": <float> }
    radius is in kilometers (Model A radial spread). Defaults to 3.0 km if not provided.
    """
    try:
        req = request.get_json(force=True) or {}
    except Exception:
        req = {}

    clicked_lat = req.get('lat', None)
    clicked_lng = req.get('lng', None)
    radius_km = float(req.get('radius', 3.0)) if req.get('radius', None) is not None else 3.0
    clicked_point = (clicked_lat, clicked_lng) if (clicked_lat is not None and clicked_lng is not None) else None

    if model is None and all_assets_df.empty:
        return jsonify({"error": "Model or assets not loaded. Check server logs."}), 500

    # PART A: Model predicted failures (if model present)
    if model is not None and MODEL_COLUMNS:
        try:
            assets_to_predict_df = all_assets_df[MODEL_COLUMNS]
            predictions = model.predict(assets_to_predict_df)
            all_failing_assets_df = all_assets_df[predictions == 1].copy()
        except Exception as e:
            print("Model prediction error:", e)
            all_failing_assets_df = pd.DataFrame()
    else:
        all_failing_assets_df = pd.DataFrame()

    # PART A.2: If clicked point provided, force-substations within radius_km to fail
    if clicked_point:
        force_radius_km = float(radius_km)
        substations_df = all_assets_df[all_assets_df['asset_type'] == 1].copy()
        forced_idxs = []
        for idx, row in substations_df.iterrows():
            try:
                dist = haversine((clicked_lat, clicked_lng), (row['latitude'], row['longitude']), unit=Unit.KILOMETERS)
            except Exception:
                # if coordinates are missing or invalid, skip
                continue
            if dist <= force_radius_km:
                forced_idxs.append(idx)
        if forced_idxs:
            forced_df = substations_df.loc[forced_idxs]
            all_failing_assets_df = pd.concat([all_failing_assets_df, forced_df]).drop_duplicates().reset_index(drop=True)

    # Identify direct failures (substations)
    direct_failures = all_failing_assets_df[all_failing_assets_df['asset_type'] == 1].copy()

    # PART B: Cascading failures for hospitals
    all_hospitals = all_assets_df[all_assets_df['asset_type'] == 0].copy()
    failing_substation_coords = [
        (row['latitude'], row['longitude']) for index, row in direct_failures.iterrows()
    ]
    cascade_failures_list = []
    if failing_substation_coords:
        for index, hospital in all_hospitals.iterrows():
            hospital_coord = (hospital['latitude'], hospital['longitude'])
            # minimal distance to any failed substation
            min_dist_km = min(
                [haversine(hospital_coord, sub_coord, unit=Unit.KILOMETERS) for sub_coord in failing_substation_coords]
            )
            # Scaled rule: hospitals fail if within (radius_km * 3.0) OR within 10km (whichever larger)
            cascade_threshold_km = max(10.0, radius_km * 3.0)
            if min_dist_km <= cascade_threshold_km:
                cascade_failures_list.append(hospital)

    cascade_failures = pd.DataFrame(cascade_failures_list)

    # PART C: Action plan
    action_plan_text = get_dummy_action_plan(direct_failures, cascade_failures, clicked_point=clicked_point, radius_km=radius_km)

    # PART D: Combine both lists for frontend
    all_failing_for_frontend = pd.concat([direct_failures, cascade_failures]).drop_duplicates().reset_index(drop=True)

    # Debug logging
    print(f"DEBUG: clicked_point={clicked_point}, radius_km={radius_km}")
    print(f"DEBUG: direct_failures count = {len(direct_failures)}")
    print(f"DEBUG: cascade_failures count = {len(cascade_failures)}")
    print(f"DEBUG FRONTEND RESPONSE SAMPLE (first 5):\n{all_failing_for_frontend.head().to_string()}")

    return jsonify({
        "failing_assets": all_failing_for_frontend.to_json(orient='records'),
        "action_plan": action_plan_text
    })

# Serve dashboard
@app.route("/")
def serve_root():
    return send_from_directory("frontend", "Dashboard.html")

@app.route("/dashboard")
def serve_dashboard():
    return send_from_directory("frontend", "Dashboard.html")

if __name__ == '__main__':
    print("🚀 Starting Flask server (Resume-Ready Plan - NO GEMINI)...")
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
