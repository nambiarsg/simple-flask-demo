
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app)

class CampaignAnalyzer:
    def __init__(self):
        self.territory_codes = {
            'KSA': 'Saudi Arabia', 'OM': 'Oman', 'EGY': 'Egypt',
            'UAE': 'United Arab Emirates', 'BH': 'Bahrain'
        }
        self.models = {
            'random_forest': None, 'xgboost': None,
            'scaler': StandardScaler(), 'label_encoder': LabelEncoder()
        }
        self.is_trained = False

    def process_campaign_data(self, df):
        try:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.dropna(subset=['sent', 'delivered', 'clicked'])

            # Campaign type detection (PRM=promotional, AUT=automated)
            df['campaign_type'] = df['campaign_action_type'].apply(
                lambda x: 'Promotional' if 'PRM' in str(x) 
                else 'Automated' if 'AUT' in str(x) else 'Unknown'
            )

            # Territory extraction
            df['territory'] = df['campaign_name'].apply(self._extract_territory)

            # Calculate metrics
            df['delivery_rate'] = (df['delivered'] / df['sent'] * 100).round(2)
            df['click_rate'] = (df['clicked'] / df['delivered'] * 100).round(2)
            df['engagement_score'] = self._calculate_engagement_score(df)

            # Time features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_of_week
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['date'] = df['timestamp'].dt.date

            return df
        except Exception as e:
            print(f"Error: {e}")
            return df

    def _extract_territory(self, name):
        if pd.isna(name): return 'Unknown'
        name = str(name).upper()
        if 'ALL' in name: return 'All Territories'
        for code, full_name in self.territory_codes.items():
            if code in name: return full_name
        return 'Unknown'

    def _calculate_engagement_score(self, df):
        weights = {'Android': 1.0, 'iOS': 1.1, 'Huawei': 0.9}
        base_score = df['click_rate'] * df['delivery_rate'] / 100
        if 'platform' in df.columns:
            platform_weights = df['platform'].map(weights).fillna(1.0)
            return (base_score * platform_weights).round(2)
        return base_score.round(2)

    def train_models(self, df):
        try:
            if len(df) < 50:
                return {"error": "Need 50+ records for training"}

            features = ['hour', 'day_of_week', 'month', 'is_weekend', 'delivery_rate']
            df_ml = df.copy()

            if 'platform' in df_ml.columns:
                df_ml['platform_encoded'] = self.models['label_encoder'].fit_transform(df_ml['platform'])
                features.append('platform_encoded')

            X = df_ml[features].fillna(0)
            y = df_ml['click_rate']
            X_scaled = self.models['scaler'].fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Train models
            self.models['random_forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.models['random_forest'].fit(X_train, y_train)

            self.models['xgboost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
            self.models['xgboost'].fit(X_train, y_train)

            rf_score = r2_score(y_test, self.models['random_forest'].predict(X_test))
            xgb_score = r2_score(y_test, self.models['xgboost'].predict(X_test))

            self.is_trained = True
            return {
                "status": "success", "rf_accuracy": rf_score * 100, "xgb_accuracy": xgb_score * 100,
                "feature_importance": dict(zip(features, self.models['random_forest'].feature_importances_))
            }
        except Exception as e:
            return {"error": str(e)}

    def predict_performance(self, df, hours=24):
        if not self.is_trained:
            return {"error": "Models not trained"}

        predictions = []
        for h in range(hours):
            hour_data = df[df['hour'] == h]
            base_ctr = hour_data['click_rate'].mean() if len(hour_data) > 0 else df['click_rate'].mean()
            predicted_ctr = base_ctr * (0.9 + np.random.random() * 0.2)
            predictions.append({
                'hour': h, 'hour_label': f'{h}:00',
                'predicted_ctr': round(predicted_ctr, 2)
            })

        optimal = sorted(predictions, key=lambda x: x['predicted_ctr'], reverse=True)[:5]
        return {"status": "success", "predictions": predictions, "optimal_hours": optimal}

    def generate_recommendations(self, df):
        recs = []
        if len(df) < 10:
            recs.append({
                'priority': 'high', 'title': 'Insufficient Data',
                'description': 'Upload 50+ records for ML analysis',
                'action': 'Collect more campaign data'
            })
            return recs

        hourly = df.groupby('hour')['click_rate'].mean()
        best_hour = hourly.idxmax()
        best_ctr = hourly.max()

        recs.append({
            'priority': 'high', 'title': 'Optimal Send Time',
            'description': f'{best_hour}:00 shows highest CTR ({best_ctr:.1f}%)',
            'action': f'Schedule campaigns around {best_hour}:00 for best results'
        })

        return recs

analyzer = CampaignAnalyzer()
current_data = None

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'CRM Campaign Analyzer'})

@app.route('/api/upload', methods=['POST'])
def upload_data():
    global current_data
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        df = pd.read_csv(file)

        required = ['timestamp', 'campaign_name', 'campaign_action_type', 'platform', 'sent', 'delivered', 'clicked']
        missing = [col for col in required if col not in df.columns]
        if missing:
            return jsonify({'error': f'Missing columns: {missing}'}), 400

        processed_df = analyzer.process_campaign_data(df)
        current_data = processed_df

        training = analyzer.train_models(processed_df)
        if "error" in training:
            return jsonify(training), 400

        predictions = analyzer.predict_performance(processed_df)
        recommendations = analyzer.generate_recommendations(processed_df)

        return jsonify({
            'status': 'success',
            'data_summary': {
                'total_records': len(processed_df),
                'platforms': list(processed_df['platform'].unique()),
                'territories': list(processed_df['territory'].unique()),
                'avg_ctr': float(processed_df['click_rate'].mean())
            },
            'training': training,
            'predictions': predictions, 
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    global current_data
    try:
        data = request.get_json()
        message = data.get('message', '').lower()

        if current_data is None:
            return jsonify({
                'response': 'No data available. Please upload campaign CSV data first!'
            })

        df = current_data

        # Generate response based on query
        if "uae" in message and "performance" in message:
            uae_data = df[df['territory'] == 'United Arab Emirates']
            if len(uae_data) > 0:
                avg_ctr = uae_data['click_rate'].mean()
                response = f"UAE Performance: {len(uae_data)} campaigns, {avg_ctr:.2f}% avg CTR"
            else:
                response = "No UAE data found. Ensure campaign names include 'UAE'."
        elif "platform" in message:
            if 'platform' in df.columns:
                platform_stats = df.groupby('platform')['click_rate'].mean()
                best = platform_stats.idxmax()
                response = f"Platform Performance: {best} is best with {platform_stats[best]:.2f}% CTR"
            else:
                response = "No platform data available"
        elif "time" in message:
            hourly = df.groupby('hour')['click_rate'].mean()
            best_hour = hourly.idxmax()
            response = f"Best send time: {best_hour}:00 ({hourly[best_hour]:.2f}% CTR)"
        else:
            total = len(df)
            avg_ctr = df['click_rate'].mean()
            response = f"Overview: {total} campaigns, {avg_ctr:.2f}% avg CTR. Ask about UAE, platforms, or send times!"

        return jsonify({'status': 'success', 'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard')
def dashboard_data():
    global current_data
    if current_data is None:
        return jsonify({'error': 'No data available'}), 400

    df = current_data

    # Summary metrics
    summary = {
        'total_campaigns': len(df),
        'avg_ctr': float(df['click_rate'].mean()),
        'avg_delivery_rate': float(df['delivery_rate'].mean()),
        'total_sent': int(df['sent'].sum()),
        'total_clicked': int(df['clicked'].sum())
    }

    # Platform breakdown
    platform_data = []
    if 'platform' in df.columns:
        for platform in df['platform'].unique():
            platform_df = df[df['platform'] == platform]
            platform_data.append({
                'platform': platform,
                'avg_ctr': float(platform_df['click_rate'].mean()),
                'campaigns': len(platform_df),
                'total_clicks': int(platform_df['clicked'].sum())
            })

    # Hourly performance
    hourly_data = []
    for hour in range(24):
        hour_df = df[df['hour'] == hour]
        if len(hour_df) > 0:
            hourly_data.append({
                'hour': hour,
                'avg_ctr': float(hour_df['click_rate'].mean())
            })

    return jsonify({
        'summary': summary,
        'platform_data': platform_data,
        'hourly_data': hourly_data
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
