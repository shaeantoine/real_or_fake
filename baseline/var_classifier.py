import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

# Local imports
from var_feat import VarianceFeatureExtractor

class VarianceBasedClassifier:
    """
    Baseline classifier focusing on variance-based detection of fake texts
    """
    def __init__(self):
        self.feature_extractor = VarianceFeatureExtractor()
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importance = {}
    
    def prepare_data(self, df):
        """Prepare training data with variance features"""
        print("Preparing training data...")
        
        # Extract features
        feature_df = self.feature_extractor.extract_features_dataframe(df)
        
        # Create labels for binary classification
        # We'll create two approaches: anomaly detection and binary classification
        
        # For binary classification: create labels for each text in the pair
        labels = []
        features_for_classification = []
        
        for idx, row in df.iterrows():
            real_features = {k: v for k, v in feature_df.iloc[idx].items() if k.startswith('real_')}
            fake_features = {k: v for k, v in feature_df.iloc[idx].items() if k.startswith('fake_')}
            
            # Remove prefixes for consistency
            real_features = {k.replace('real_', ''): v for k, v in real_features.items()}
            fake_features = {k.replace('fake_', ''): v for k, v in fake_features.items()}
            
            # Add real text features with label 1 (real)
            features_for_classification.append(real_features)
            labels.append(1)
            
            # Add fake text features with label 0 (fake)
            features_for_classification.append(fake_features)
            labels.append(0)
        
        classification_df = pd.DataFrame(features_for_classification)
        
        return classification_df, np.array(labels), feature_df
    
    def train_isolation_forest(self, real_features):
        """Train isolation forest on real texts only (anomaly detection)"""
        print("🌲 Training Isolation Forest (Anomaly Detection)...")
        
        # Scale features
        real_features_scaled = self.scaler.fit_transform(real_features)
        
        # Train isolation forest on real texts only
        isolation_forest = IsolationForest(
            contamination=0.1,  # Assume 10% contamination
            random_state=42,
            n_estimators=100
        )
        isolation_forest.fit(real_features_scaled)
        
        self.models['isolation_forest'] = isolation_forest
        
        return isolation_forest
    
    def train_binary_classifiers(self, X, y):
        """Train binary classifiers"""
        print("🤖 Training Binary Classifiers...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_scaled, y)
        self.models['random_forest'] = rf
        
        # Store feature importance
        self.feature_importance['random_forest'] = dict(zip(X.columns, rf.feature_importances_))
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=1
        )
        xgb_model.fit(X_scaled, y)
        self.models['xgboost'] = xgb_model
        
        # Store feature importance
        self.feature_importance['xgboost'] = dict(zip(X.columns, xgb_model.feature_importances_))
        
        return rf, xgb_model
    
    def evaluate_models(self, X, y):
        """Evaluate all trained models using cross-validation"""
        print("📈 Evaluating Models...")
        
        X_scaled = self.scaler.transform(X)
        
        # Evaluate binary classifiers
        for name, model in self.models.items():
            if name == 'isolation_forest':
                continue  # Handle separately
                
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
            print(f"{name.title()}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")
        
        # Evaluate isolation forest (different approach)
        if 'isolation_forest' in self.models:
            # For isolation forest, we need to evaluate differently
            # since it's trained only on real texts
            print("Isolation Forest: Trained for anomaly detection (evaluate on test set)")
    
    def predict_fake_text(self, file_1, file_2):
        """Predict which text in a pair is fake"""
        # Extract features for both texts
        features_1 = self.feature_extractor.extract_single_text_features(file_1)
        features_2 = self.feature_extractor.extract_single_text_features(file_2)
        
        # Convert to dataframe for scaling
        features_df = pd.DataFrame([features_1, features_2])
        features_scaled = self.scaler.transform(features_df)
        
        predictions = {}
        
        # Binary classifier predictions
        for name, model in self.models.items():
            if name == 'isolation_forest':
                # Anomaly scores (lower = more anomalous = more likely fake)
                anomaly_scores = model.decision_function(features_scaled)
                predictions[name] = 1 if anomaly_scores[0] < anomaly_scores[1] else 2
            else:
                # Probability of being real
                proba = model.predict_proba(features_scaled)[:, 1]  # Probability of class 1 (real)
                predictions[name] = 1 if proba[0] > proba[1] else 2
        
        return predictions
    
    def get_feature_importance(self, top_n=10):
        """Get top important features from trained models"""
        print(f"\n🔍 Top {top_n} Most Important Features:")
        print("=" * 50)
        
        for model_name, importance_dict in self.feature_importance.items():
            print(f"\n{model_name.title()}:")
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:top_n]:
                print(f"  {feature}: {importance:.4f}")
