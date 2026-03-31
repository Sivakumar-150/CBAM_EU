"""
🌍 CBAM ML Model Training & Pickle Generation
Trains XGBoost model and saves as pickle files for deployment
Run: python train_model.py
"""

import os
import sys
import pickle
import traceback
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from xgboost import XGBRegressor

print("\n" + "=" * 70)
print("🌍 CBAM ML MODEL TRAINING & PICKLE GENERATION")
print("=" * 70 + "\n")

# ============ STEP 1: LOAD & PREPARE DATA ============
print("📂 STEP 1: Loading Data...")
try:
    df = pd.read_csv("cbam_cleaned.csv")
    print(f"✅ Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")
    print(f"\nFirst few rows:\n{df.head(3)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
except Exception as e:
    print(f"❌ ERROR loading CSV: {e}")
    sys.exit(1)

# ============ STEP 2: EXPLORATORY DATA ANALYSIS ============
print("\n" + "=" * 70)
print("📊 STEP 2: Exploratory Data Analysis")
print("=" * 70)

print(f"\nDataset Shape: {df.shape}")
print(f"\nTarget Variable Statistics (net_cbam_liability_eur):")
print(df['net_cbam_liability_eur'].describe())

print(f"\nData Types:")
print(df.dtypes)

# Visualizations
print("\n📈 Generating visualizations...")

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.histplot(df['net_cbam_liability_eur'], bins=50, kde=True, color='green')
plt.title("Distribution of CBAM Liability")
plt.xlabel("CBAM Liability (€)")
plt.ylabel("Frequency")

plt.subplot(2, 2, 2)
sns.boxplot(x="country_of_origin", y="net_cbam_liability_eur", data=df, palette="Set2")
plt.xticks(rotation=45)
plt.title("CBAM Liability by Country")

plt.subplot(2, 2, 3)
sns.scatterplot(x="total_emissions_tco2", y="net_cbam_liability_eur",
                hue="product_category", data=df, s=80, alpha=0.7)
plt.title("Emissions vs CBAM Liability")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.subplot(2, 2, 4)
df['product_category'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Product Categories Distribution")
plt.xlabel("Category")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("✅ Saved: eda_analysis.png")

# ============ STEP 3: FEATURE ENGINEERING ============
print("\n" + "=" * 70)
print("🔧 STEP 3: Feature Engineering")
print("=" * 70)

# Handle division by zero safely
df['emission_intensity'] = df['embedded_emissions_tco2'] / df['quantity_tonnes'].replace(0, 1)
df['emission_intensity'] = df['emission_intensity'].fillna(df['emission_intensity'].median())

# Additional robust features
df['carbon_price_gap'] = df['eu_ets_price_eur'] - df['carbon_price_origin_eur']
df['total_emissions'] = df['direct_emissions_tco2'] + df['indirect_emissions_tco2']
df['cost_per_tonne'] = df['net_cbam_liability_eur'] / df['quantity_tonnes'].replace(0, 1)
df['cost_per_tonne'] = df['cost_per_tonne'].fillna(df['cost_per_tonne'].median())

# Interaction features
df['emission_ratio'] = df['direct_emissions_tco2'] / (df['total_emissions'] + 1)
df['price_ratio'] = df['carbon_price_origin_eur'] / (df['eu_ets_price_eur'] + 1)
df['emission_to_quantity'] = df['total_emissions'] / df['quantity_tonnes'].replace(0, 1)
df['high_emission_flag'] = (df['total_emissions'] > df['total_emissions'].quantile(0.75)).astype(int)
df['high_price_gap_flag'] = (df['carbon_price_gap'] > df['carbon_price_gap'].quantile(0.75)).astype(int)

# Log transforms for skewed features
df['log_quantity'] = np.log1p(df['quantity_tonnes'])
df['log_emissions'] = np.log1p(df['total_emissions'])
df['log_liability'] = np.log1p(df['net_cbam_liability_eur'])

print(f"✅ Created {len(df.columns)} features total")
print(f"\nNew features:")
print(df[['emission_intensity', 'carbon_price_gap', 'cost_per_tonne',
          'emission_ratio', 'price_ratio', 'log_quantity']].head())

# ============ STEP 4: DEFINE FEATURES ============
print("\n" + "=" * 70)
print("🎯 STEP 4: Feature Selection")
print("=" * 70)

categorical = ['country_of_origin', 'product_category', 'production_method']
numerical = [
    'quantity_tonnes', 'direct_emissions_tco2', 'indirect_emissions_tco2',
    'embedded_emissions_tco2', 'eu_ets_price_eur', 'carbon_price_origin_eur',
    'total_emissions_tco2', 'emission_intensity', 'carbon_price_gap',
    'cost_per_tonne', 'emission_ratio', 'price_ratio', 'emission_to_quantity',
    'high_emission_flag', 'high_price_gap_flag', 'log_quantity', 'log_emissions'
]

# Remove features that don't exist
numerical = [col for col in numerical if col in df.columns]

print(f"✅ Categorical features ({len(categorical)}): {categorical}")
print(f"✅ Numerical features ({len(numerical)}): {numerical[:5]}... (showing first 5)")

# ============ STEP 5: DATA SPLIT ============
print("\n" + "=" * 70)
print("📋 STEP 5: Train-Test Split")
print("=" * 70)

X = df[categorical + numerical]
y = df['net_cbam_liability_eur']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Train set: {X_train.shape[0]} samples ({X_train.shape[1]} features)")
print(f"✅ Test set: {X_test.shape[0]} samples")
print(f"✅ Target variable range: €{y.min():,.0f} - €{y.max():,.0f}")

# ============ STEP 6: PREPROCESSING PIPELINE ============
print("\n" + "=" * 70)
print("⚙️ STEP 6: Preprocessing Pipeline Setup")
print("=" * 70)

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=10), categorical),
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numerical)
])

print("✅ Preprocessing pipeline created")

# ============ STEP 7: MODEL TRAINING ============
print("\n" + "=" * 70)
print("🤖 STEP 7: XGBoost Model Training")
print("=" * 70)

start_time = time.time()

# Full pipeline with model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        tree_method='hist',
        eval_metric='rmse'
    ))
])

# Hyperparameter search
print("\n🔍 Running RandomizedSearchCV (50 iterations)...")

param_dist = {
    'regressor__n_estimators': [800, 1000, 1200, 1500],
    'regressor__max_depth': [5, 6, 7, 8, 9],
    'regressor__learning_rate': [0.01, 0.03, 0.05, 0.07],
    'regressor__subsample': [0.7, 0.8, 0.9, 1.0],
    'regressor__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'regressor__min_child_weight': [1, 3, 5],
    'regressor__gamma': [0, 1, 2],
}

random_search = RandomizedSearchCV(
    model,
    param_dist,
    n_iter=50,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1,
    random_state=42,
    pre_dispatch='2*n_jobs'
)

random_search.fit(X_train, y_train)
training_time = time.time() - start_time

print(f"\n✅ Training completed in {training_time:.2f} seconds")
print(f"\n🏆 Best Parameters Found:")
for param, value in random_search.best_params_.items():
    print(f"   {param}: {value}")

# ============ STEP 8: MODEL EVALUATION ============
print("\n" + "=" * 70)
print("📊 STEP 8: Model Evaluation")
print("=" * 70)

y_pred_train = random_search.predict(X_train)
y_pred_test = random_search.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"\n📈 Training Metrics:")
print(f"   R² Score: {r2_train:.4f}")
print(f"   RMSE: €{rmse_train:,.2f}")
print(f"   MAE: €{mae_train:,.2f}")

print(f"\n✅ Testing Metrics:")
print(f"   R² Score: {r2_test:.4f}")
print(f"   RMSE: €{rmse_test:,.2f}")
print(f"   MAE: €{mae_test:,.2f}")

print(f"\n🎯 Best CV Score: {random_search.best_score_:.4f}")

# Check for overfitting
if r2_train - r2_test > 0.15:
    print(f"\n⚠️  WARNING: Model shows signs of overfitting")
    print(f"   Difference: {r2_train - r2_test:.4f}")
else:
    print(f"\n✅ Good generalization! (difference: {r2_train - r2_test:.4f})")

# ============ STEP 9: FEATURE IMPORTANCE ============
print("\n" + "=" * 70)
print("🔍 STEP 9: Feature Importance Analysis")
print("=" * 70)

try:
    fitted_preprocessor = random_search.best_estimator_.named_steps['preprocessor']
    cat_features = list(fitted_preprocessor.named_transformers_['cat'].get_feature_names_out(categorical))
    all_feature_names = np.array(cat_features + numerical)

    importances = random_search.best_estimator_.named_steps['regressor'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\n🏆 Top 15 Most Important Features:")
    print(feature_importance_df.head(15).to_string(index=False))

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_features = feature_importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'].values, color='teal')
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance Score')
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("\n✅ Saved: feature_importance.png")

except Exception as e:
    print(f"⚠️  Feature importance error: {e}")

# ============ STEP 10: CLUSTERING ANALYSIS ============
print("\n" + "=" * 70)
print("🎯 STEP 10: Clustering Analysis")
print("=" * 70)

try:
    X_preprocessed = fitted_preprocessor.transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_preprocessed)
    df['cluster'] = clusters

    print(f"\n✅ K-Means clustering complete (k=3)")
    print(f"Cluster distribution:")
    print(df['cluster'].value_counts().sort_index())

    # Cluster analysis
    print(f"\nCluster Statistics:")
    for cluster_id in range(3):
        cluster_data = df[df['cluster'] == cluster_id]
        print(f"\n  Cluster {cluster_id}:")
        print(f"    Count: {len(cluster_data)}")
        print(f"    Avg Liability: €{cluster_data['net_cbam_liability_eur'].mean():,.2f}")
        print(f"    Avg Emissions: {cluster_data['total_emissions_tco2'].mean():.2f} tCO₂")

except Exception as e:
    print(f"⚠️  Clustering error: {e}")

# ============ STEP 11: SAVE PICKLE FILES ============
print("\n" + "=" * 70)
print("💾 STEP 11: Saving Pickle Files")
print("=" * 70)

try:
    # Save main model
    model_path = 'cbam_model.pkl'
    joblib.dump(random_search.best_estimator_, model_path)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✅ Model saved: {model_path} ({model_size_mb:.2f} MB)")

    # Save model info
    model_info = {
        'categorical': categorical,
        'numerical': numerical,
        'r2_train': float(r2_train),
        'r2_test': float(r2_test),
        'rmse_train': float(rmse_train),
        'rmse_test': float(rmse_test),
        'mae_train': float(mae_train),
        'mae_test': float(mae_test),
        'best_cv_score': float(random_search.best_score_),
        'best_params': random_search.best_params_,
        'n_samples': len(df),
        'feature_names': list(all_feature_names),
        'training_time': training_time,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    model_info_path = 'model_info.pkl'
    with open(model_info_path, 'wb') as f:
        pickle.dump(model_info, f)
    print(f"✅ Model info saved: {model_info_path}")

    # Save training data stats
    data_stats = {
        'n_rows': len(df),
        'n_features': len(categorical + numerical),
        'target_min': float(y.min()),
        'target_max': float(y.max()),
        'target_mean': float(y.mean()),
        'target_std': float(y.std()),
        'countries': list(df['country_of_origin'].unique()),
        'categories': list(df['product_category'].unique()),
        'methods': list(df['production_method'].unique())
    }

    data_stats_path = 'data_stats.pkl'
    with open(data_stats_path, 'wb') as f:
        pickle.dump(data_stats, f)
    print(f"✅ Data stats saved: {data_stats_path}")

except Exception as e:
    print(f"❌ ERROR saving pickle files: {e}")
    traceback.print_exc()
    sys.exit(1)

# ============ STEP 12: VERIFY PICKLE FILES ============
print("\n" + "=" * 70)
print("🔍 STEP 12: Verifying Pickle Files")
print("=" * 70)

try:
    # Load and verify model
    loaded_model = joblib.load(model_path)
    test_pred = loaded_model.predict(X_test[:5])
    print(f"✅ Model loaded successfully")
    print(f"   Sample predictions: {test_pred[:3]}")

    # Load and verify info
    with open(model_info_path, 'rb') as f:
        loaded_info = pickle.load(f)
    print(f"✅ Model info loaded successfully")
    print(f"   R² Score: {loaded_info['r2_test']:.4f}")
    print(f"   RMSE: €{loaded_info['rmse_test']:,.2f}")

    # Load and verify stats
    with open(data_stats_path, 'rb') as f:
        loaded_stats = pickle.load(f)
    print(f"✅ Data stats loaded successfully")
    print(f"   Records: {loaded_stats['n_rows']}")
    print(f"   Countries: {loaded_stats['countries']}")

except Exception as e:
    print(f"❌ ERROR verifying pickle files: {e}")
    sys.exit(1)

# ============ FINAL SUMMARY ============
print("\n" + "=" * 70)
print("✅ COMPLETE! MODEL TRAINING FINISHED")
print("=" * 70)

summary = f"""
📊 TRAINING SUMMARY:
   Training Time: {training_time:.2f} seconds
   Train R² Score: {r2_train:.4f}
   Test R² Score: {r2_test:.4f}
   Test RMSE: €{rmse_test:,.2f}
   Test MAE: €{mae_test:,.2f}

📁 FILES CREATED:
   ✅ cbam_model.pkl ({model_size_mb:.2f} MB)
   ✅ model_info.pkl
   ✅ data_stats.pkl
   ✅ eda_analysis.png
   ✅ feature_importance.png

🚀 READY FOR DEPLOYMENT!
   Run: gunicorn app:app
   Visit: http://localhost:5000
"""

print(summary)

print("\n" + "=" * 70)
print("🎉 Happy Predicting! 🌍")
print("=" * 70 + "\n")