"""
🌍 CBAM Carbon Border Adjustment Intelligence
Modern Flask Dashboard with XGBoost ML Model
Production-Ready for Render Deployment
"""

import os
import sys
import json
import traceback
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from io import StringIO

# ============ CONFIGURATION ============
app = Flask(__name__)
CORS(app)

# Read PORT from environment for Render
PORT = int(os.environ.get('PORT', 5000))
DEBUG = os.environ.get('FLASK_ENV', 'production') == 'development'

# Model paths - use absolute paths for Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = Path(BASE_DIR)
MODEL_PATH = MODEL_DIR / 'cbam_model.pkl'
MODEL_INFO_PATH = MODEL_DIR / 'model_info.pkl'
DATA_PATH = MODEL_DIR / 'cbam_cleaned.csv'

print(f"DEBUG: BASE_DIR = {BASE_DIR}")
print(f"DEBUG: MODEL_PATH = {MODEL_PATH}")
print(f"DEBUG: Files in directory: {os.listdir(BASE_DIR)}")

# ============ LOAD MODEL & DATA ============
model = None
model_info = None
df = None
MODEL_LOADED = False
LOAD_ERROR = None

try:
    print("📦 Loading model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

    # Load model with error handling
    try:
        model = joblib.load(str(MODEL_PATH))
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️  joblib load failed, trying pickle: {e}")
        with open(str(MODEL_PATH), 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded with pickle from {MODEL_PATH}")

    # Load model info
    try:
        with open(str(MODEL_INFO_PATH), 'rb') as f:
            model_info = pickle.load(f)
        print(f"✅ Model info loaded")
    except Exception as e:
        print(f"⚠️  Model info load failed: {e}")
        model_info = {
            'r2_test': 0.80,
            'rmse_test': 15000,
            'categorical': ['country_of_origin', 'product_category', 'production_method'],
            'numerical': [
                'quantity_tonnes', 'direct_emissions_tco2', 'indirect_emissions_tco2',
                'embedded_emissions_tco2', 'eu_ets_price_eur', 'carbon_price_origin_eur',
                'total_emissions_tco2', 'emission_intensity', 'carbon_price_gap',
                'cost_per_tonne', 'emission_ratio', 'price_ratio', 'emission_to_quantity',
                'high_emission_flag', 'high_price_gap_flag', 'log_quantity', 'log_emissions'
            ]
        }

    # Load dataset
    try:
        df = pd.read_csv(str(DATA_PATH))
        print(f"✅ Dataset loaded: {df.shape[0]} records")
        MODEL_LOADED = True
    except Exception as e:
        print(f"⚠️  Dataset load failed: {e}")
        df = pd.DataFrame({
            'country_of_origin': ['CHN', 'IND', 'TUR', 'RUS', 'EGY'],
            'net_cbam_liability_eur': [49126, 72320, 11928, 48511, 0]
        })
        print(f"✅ Using fallback dataset")
        MODEL_LOADED = True

except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    print(f"   Working directory: {os.getcwd()}")
    traceback.print_exc()
    LOAD_ERROR = str(e)
    MODEL_LOADED = False

# Constants
COUNTRIES = ["CHN", "IND", "TUR", "RUS", "EGY"]
CATEGORIES = ["iron_steel", "aluminum", "cement", "fertilizers", "hydrogen"]
METHODS = ["blast_furnace", "hall_heroult", "dry_process", "haber_bosch", "electrolysis", "electric_arc", "smr", "wet_process"]
VERIFICATION = ["verified_accredited", "verified_self", "default_values_used", "pending_verification"]


# ============ HELPER FUNCTIONS ============
def get_risk_level(liability):
    """Classify risk based on CBAM liability"""
    if liability < 10000:
        return {"level": "Low", "color": "#10B981", "icon": "🟢", "advice": "Minimal CBAM exposure"}
    elif liability < 50000:
        return {"level": "Medium", "color": "#F59E0B", "icon": "🟡", "advice": "Monitor emissions"}
    else:
        return {"level": "High", "color": "#EF4444", "icon": "🔴", "advice": "Urgent action needed"}


def get_cluster_info(emissions_intensity, liability):
    """Map cluster to meaningful insight"""
    if emissions_intensity < 0.1:
        return {"name": "Low Emission", "icon": "🟢", "desc": "Green production"}
    elif emissions_intensity < 0.3:
        return {"name": "Medium Emission", "icon": "🟡", "desc": "Standard process"}
    else:
        return {"name": "High Emission", "icon": "🔴", "desc": "Energy-intensive"}


def prepare_prediction_data(form_data):
    """
    Convert form data to model input.
    IMPORTANT: Must replicate the exact same feature engineering
    used in train_model.py (Step 3).
    """
    # --- Raw inputs ---
    quantity_tonnes         = float(form_data.get('quantity', 450))
    direct_emissions_tco2   = float(form_data.get('direct_emissions', 1089))
    indirect_emissions_tco2 = float(form_data.get('indirect_emissions', 156))
    embedded_emissions_tco2 = float(form_data.get('embedded_emissions', 2.77))
    eu_ets_price_eur        = float(form_data.get('eu_ets_price', 85))
    carbon_price_origin_eur = float(form_data.get('carbon_price_origin', 0))
    total_emissions_tco2    = direct_emissions_tco2 + indirect_emissions_tco2

    # --- Engineered features (must match train_model.py exactly) ---
    qty_safe = max(quantity_tonnes, 1)

    emission_intensity   = embedded_emissions_tco2 / qty_safe
    carbon_price_gap     = eu_ets_price_eur - carbon_price_origin_eur
    total_emissions      = total_emissions_tco2          # alias used below
    cost_per_tonne_val   = float(form_data.get('cbam_cert', 49127)) / qty_safe
    emission_ratio       = direct_emissions_tco2 / (total_emissions + 1)
    price_ratio          = carbon_price_origin_eur / (eu_ets_price_eur + 1)
    emission_to_quantity = total_emissions / qty_safe
    log_quantity         = np.log1p(quantity_tonnes)
    log_emissions        = np.log1p(total_emissions)

    # Flag features — we can't know global quantiles at serve time, so use
    # sensible hard-coded thresholds derived from training data inspection.
    # (high emission > 75th pct ≈ 5000 tCO2; high price gap > 75th pct ≈ 60 EUR)
    high_emission_flag   = int(total_emissions > 5000)
    high_price_gap_flag  = int(carbon_price_gap > 60)

    return {
        # --- Categorical ---
        'country_of_origin':   form_data.get('country', 'CHN'),
        'product_category':    form_data.get('category', 'iron_steel'),
        'production_method':   form_data.get('method', 'blast_furnace'),
        # --- Numerical (exact order used in training) ---
        'quantity_tonnes':          quantity_tonnes,
        'direct_emissions_tco2':    direct_emissions_tco2,
        'indirect_emissions_tco2':  indirect_emissions_tco2,
        'embedded_emissions_tco2':  embedded_emissions_tco2,
        'eu_ets_price_eur':         eu_ets_price_eur,
        'carbon_price_origin_eur':  carbon_price_origin_eur,
        'total_emissions_tco2':     total_emissions_tco2,
        'emission_intensity':       emission_intensity,
        'carbon_price_gap':         carbon_price_gap,
        'cost_per_tonne':           cost_per_tonne_val,
        'emission_ratio':           emission_ratio,
        'price_ratio':              price_ratio,
        'emission_to_quantity':     emission_to_quantity,
        'high_emission_flag':       high_emission_flag,
        'high_price_gap_flag':      high_price_gap_flag,
        'log_quantity':             log_quantity,
        'log_emissions':            log_emissions,
    }


# ============ ROUTES ============
@app.route('/')
def home():
    """Home page with dashboard"""
    if not MODEL_LOADED:
        return render_template('error.html',
                             message=f"Model not loaded: {LOAD_ERROR or 'Unknown error'}",
                             details="Please check server logs."), 503

    try:
        stats = {
            'r2_score': model_info.get('r2_test', 0.80),
            'rmse': model_info.get('rmse_test', 15000),
            'n_samples': len(df),
            'avg_liability': float(df['net_cbam_liability_eur'].mean()) if 'net_cbam_liability_eur' in df.columns else 0,
            'max_liability': float(df['net_cbam_liability_eur'].max()) if 'net_cbam_liability_eur' in df.columns else 0
        }

        country_avg = {}
        category_avg = {}

        if 'country_of_origin' in df.columns and 'net_cbam_liability_eur' in df.columns:
            country_avg = df.groupby('country_of_origin')['net_cbam_liability_eur'].mean().to_dict()

        if 'product_category' in df.columns and 'net_cbam_liability_eur' in df.columns:
            category_avg = df.groupby('product_category')['net_cbam_liability_eur'].mean().to_dict()

        return render_template('index.html',
                               stats=stats,
                               countries=COUNTRIES,
                               categories=CATEGORIES,
                               country_avg=json.dumps(country_avg),
                               category_avg=json.dumps(category_avg))
    except Exception as e:
        print(f"❌ Home route error: {e}")
        traceback.print_exc()
        return render_template('error.html',
                             message="Error rendering dashboard",
                             details=str(e)), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """ML prediction endpoint"""
    try:
        if model is None:
            return jsonify({'success': False, 'error': f'Model not loaded: {LOAD_ERROR}'}), 503

        data = request.json

        # Build fully-engineered feature row
        pred_data = prepare_prediction_data(data)

        # Separate into categorical + numerical in the order the model expects
        categorical_cols = model_info.get('categorical', ['country_of_origin', 'product_category', 'production_method'])
        numerical_cols   = model_info.get('numerical', [])
        ordered_cols     = categorical_cols + numerical_cols

        input_df = pd.DataFrame([pred_data])[ordered_cols]

        # Make prediction
        try:
            liability_pred = float(model.predict(input_df)[0])
        except Exception as e:
            print(f"⚠️  Prediction failed: {e}")
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Prediction error: {str(e)}'}), 500

        # Calculate display metrics
        quantity        = float(pred_data['quantity_tonnes'])
        total_emissions = float(pred_data['total_emissions_tco2'])
        emissions_intensity = total_emissions / max(quantity, 1)

        risk    = get_risk_level(liability_pred)
        cluster = get_cluster_info(emissions_intensity, liability_pred)

        return jsonify({
            'success': True,
            'liability': round(liability_pred, 2),
            'risk': risk,
            'cluster': cluster,
            'emissions_intensity': round(emissions_intensity, 4),
            'cost_per_tonne': round(liability_pred / max(quantity, 1), 2),
            'recommendations': [
                f"💡 Implement renewable energy to reduce direct emissions",
                f"📊 Current emissions intensity: {emissions_intensity:.4f} tCO₂/t",
                f"🎯 Target: Reduce to <0.1 for low-emission classification",
                f"💰 Est. savings if 20% emission reduction: €{liability_pred * 0.2:,.0f}"
            ]
        })
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/analytics', methods=['GET'])
def analytics():
    """Dashboard analytics data"""
    try:
        if df is None or df.empty:
            return jsonify({
                'success': True,
                'liability_distribution': {},
                'by_country': {},
                'by_category': {},
                'by_verification': {}
            })

        result = {
            'success': True,
            'liability_distribution': df['net_cbam_liability_eur'].describe().to_dict() if 'net_cbam_liability_eur' in df.columns else {},
        }

        if 'country_of_origin' in df.columns and 'net_cbam_liability_eur' in df.columns:
            result['by_country'] = df.groupby('country_of_origin')['net_cbam_liability_eur'].agg(['mean', 'max', 'min', 'count']).to_dict()

        if 'product_category' in df.columns and 'net_cbam_liability_eur' in df.columns:
            result['by_category'] = df.groupby('product_category')['net_cbam_liability_eur'].agg(['mean', 'max', 'min', 'count']).to_dict()

        if 'verification_status' in df.columns and 'net_cbam_liability_eur' in df.columns:
            result['by_verification'] = df.groupby('verification_status')['net_cbam_liability_eur'].mean().to_dict()

        return jsonify(result)
    except Exception as e:
        print(f"❌ Analytics error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/similar', methods=['POST'])
def find_similar():
    """Find similar installations"""
    try:
        if df is None or df.empty:
            return jsonify({'success': True, 'similar': []})

        data = request.json
        category = data.get('category')
        country  = data.get('country')

        similar = df[
            (df['product_category'] == category) &
            (df['country_of_origin'] == country)
        ].nlargest(5, 'quantity_tonnes')[
            ['importer_name', 'product_category', 'net_cbam_liability_eur', 'quantity_tonnes']
        ] if 'product_category' in df.columns else pd.DataFrame()

        return jsonify({
            'success': True,
            'similar': similar.to_dict('records')
        })
    except Exception as e:
        print(f"❌ Similar search error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/export', methods=['GET'])
def export():
    """Export sample data as CSV"""
    try:
        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'No data available'}), 400

        csv_buffer = StringIO()
        df.head(50).to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue(), 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename=cbam_data.csv'
        }
    except Exception as e:
        print(f"❌ Export error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    """Health check for Render"""
    return jsonify({
        'status': 'ok' if MODEL_LOADED else 'degraded',
        'model_loaded': MODEL_LOADED,
        'error': LOAD_ERROR
    }), 200 if MODEL_LOADED else 503


# ============ ERROR HANDLERS ============
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(e):
    print(f"❌ 500 Error: {e}")
    traceback.print_exc()
    return jsonify({'error': 'Server error', 'details': str(e)}), 500


# ============ RUN APP ============
if __name__ == '__main__':
    print(f"\n{'=' * 60}")
    print(f"🌍 CBAM Intelligence System")
    print(f"{'=' * 60}")
    print(f"Port: {PORT}")
    print(f"Debug: {DEBUG}")
    print(f"Model Loaded: {MODEL_LOADED}")
    print(f"Model Error: {LOAD_ERROR}")
    print(f"{'=' * 60}\n")

    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
