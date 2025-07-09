import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def debug_baseline_results(classifier, X, y, feature_df, df_original):
    """
    Debug why our baseline is performing poorly and identify improvements
    """
    print("1. LABEL & FEATURE ANALYSIS")
    print("-" * 30)
    
    # Check if features actually differ between real and fake
    real_features = X[y == 1]
    fake_features = X[y == 0]
    
    print(f"Real texts features shape: {real_features.shape}")
    print(f"Fake texts features shape: {fake_features.shape}")
    
    # Compare means of top features
    top_features = ['proper_noun_density', 'word_length_iqr', 'word_length_range', 
                   'very_short_words', 'unique_word_ratio', 'sent_word_var']
    
    print(f"\nTop Feature Comparison (Real vs Fake):")
    for feature in top_features:
        if feature in X.columns:
            real_mean = real_features[feature].mean()
            fake_mean = fake_features[feature].mean()
            real_std = real_features[feature].std()
            fake_std = fake_features[feature].std()
            
            print(f"{feature}:")
            print(f"  Real: {real_mean:.4f} ± {real_std:.4f}")
            print(f"  Fake: {fake_mean:.4f} ± {fake_std:.4f}")
            print(f"  Difference: {real_mean - fake_mean:.4f}")
            print(f"  Effect size: {(real_mean - fake_mean) / (real_std + fake_std + 1e-8):.4f}")
            print()
    
    # 2. Visualize feature distributions
    print("2. FEATURE VISUALIZATION")
    print("-" * 30)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features[:6]):
        if feature in X.columns:
            axes[i].hist(real_features[feature], alpha=0.7, label='Real', bins=20, color='green')
            axes[i].hist(fake_features[feature], alpha=0.7, label='Fake', bins=20, color='red')
            axes[i].set_title(f'{feature}')
            axes[i].legend()
    
    plt.tight_layout()
    plt.show()
    
    # 3. Check for data leakage or labeling issues
    print("3. DATA QUALITY CHECKS")
    print("-" * 30)
    
    # Check if any features are too similar between real and fake
    feature_correlations = []
    for feature in X.columns:
        real_vals = real_features[feature].values
        fake_vals = fake_features[feature].values
        correlation = np.corrcoef(real_vals, fake_vals)[0, 1]
        feature_correlations.append((feature, correlation))
    
    # Sort by correlation
    feature_correlations.sort(key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)
    
    print("Features with highest real-fake correlation (potential issues):")
    for feature, corr in feature_correlations[:10]:
        if not np.isnan(corr):
            print(f"  {feature}: {corr:.4f}")
    
    # 4. PCA visualization
    print("\n4. PCA VISUALIZATION")
    print("-" * 30)
    
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(classifier.scaler.transform(X))
    
    plt.figure(figsize=(10, 8))
    real_mask = y == 1
    fake_mask = y == 0
    
    plt.scatter(X_pca[real_mask, 0], X_pca[real_mask, 1], 
               c='green', label='Real', alpha=0.6, s=50)
    plt.scatter(X_pca[fake_mask, 0], X_pca[fake_mask, 1], 
               c='red', label='Fake', alpha=0.6, s=50)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA: Real vs Fake Texts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_[:2]}")
    
    return feature_correlations, X_pca

def create_improved_features(df_original):
    """
    Create improved features based on our debugging insights
    """
    print("🔧 CREATING IMPROVED FEATURES")
    print("=" * 50)
    
    improved_features = []
    
    for idx, row in df_original.iterrows():
        # Get real and fake texts
        real_text = row['real_text'] if 'real_text' in row else (
            row['file_1'] if row['real_file_label'] == 1 else row['file_2']
        )
        fake_text = row['fake_text'] if 'fake_text' in row else (
            row['file_2'] if row['real_file_label'] == 1 else row['file_1']
        )
        
        features = {}
        
        # === RELATIVE FEATURES (comparing real vs fake directly) ===
        
        # Length ratios
        real_len = len(str(real_text))
        fake_len = len(str(fake_text))
        features['length_ratio'] = real_len / (fake_len + 1e-8)
        features['length_diff_abs'] = abs(real_len - fake_len)
        
        # Word count ratios
        real_words = len(str(real_text).split())
        fake_words = len(str(fake_text).split())
        features['word_count_ratio'] = real_words / (fake_words + 1e-8)
        features['word_diff_abs'] = abs(real_words - fake_words)
        
        # Complexity ratios
        real_avg_word_len = np.mean([len(w) for w in str(real_text).split()]) if real_words > 0 else 0
        fake_avg_word_len = np.mean([len(w) for w in str(fake_text).split()]) if fake_words > 0 else 0
        features['avg_word_len_ratio'] = real_avg_word_len / (fake_avg_word_len + 1e-8)
        
        # === VARIANCE COMPARISON FEATURES ===
        
        # Compare variance between real and fake
        def get_word_length_variance(text):
            words = str(text).split()
            if len(words) > 1:
                return np.var([len(w) for w in words])
            return 0
        
        real_var = get_word_length_variance(real_text)
        fake_var = get_word_length_variance(fake_text)
        features['word_var_ratio'] = real_var / (fake_var + 1e-8)
        features['word_var_diff'] = abs(real_var - fake_var)
        features['higher_variance'] = 1 if fake_var > real_var else 0  # Binary: is fake more variable?
        
        # === CONTENT QUALITY FEATURES ===
        
        # Numbers and specificity
        real_numbers = len(re.findall(r'\d+\.?\d*', str(real_text)))
        fake_numbers = len(re.findall(r'\d+\.?\d*', str(fake_text)))
        features['numbers_ratio'] = real_numbers / (fake_numbers + 1e-8)
        features['numbers_diff'] = abs(real_numbers - fake_numbers)
        
        # Proper nouns
        def count_proper_nouns(text):
            words = str(text).split()
            count = 0
            for i, word in enumerate(words):
                if word and word[0].isupper() and i > 0 and words[i-1][-1] not in '.!?':
                    count += 1
            return count
        
        real_proper = count_proper_nouns(real_text)
        fake_proper = count_proper_nouns(fake_text)
        features['proper_noun_ratio'] = real_proper / (fake_proper + 1e-8)
        features['proper_noun_diff'] = abs(real_proper - fake_proper)
        
        # Technical terms
        tech_terms = ['telescope', 'survey', 'observation', 'stellar', 'galaxy', 'star', 
                     'astronomical', 'magnitude', 'photometric', 'spectroscopic', 
                     'wavelength', 'redshift', 'luminosity', 'parsec', 'virsa', 'vista']
        
        real_tech = sum(1 for term in tech_terms if term.lower() in str(real_text).lower())
        fake_tech = sum(1 for term in tech_terms if term.lower() in str(fake_text).lower())
        features['tech_terms_ratio'] = real_tech / (fake_tech + 1e-8)
        features['tech_terms_diff'] = abs(real_tech - fake_tech)
        
        # === CONSISTENCY FEATURES ===
        
        # Repetition patterns
        real_word_freq = Counter(str(real_text).lower().split())
        fake_word_freq = Counter(str(fake_text).lower().split())
        
        real_repetition = sum(1 for count in real_word_freq.values() if count > 1)
        fake_repetition = sum(1 for count in fake_word_freq.values() if count > 1)
        features['repetition_ratio'] = real_repetition / (fake_repetition + 1e-8)
        
        # Vocabulary diversity
        real_unique_ratio = len(real_word_freq) / (real_words + 1e-8)
        fake_unique_ratio = len(fake_word_freq) / (fake_words + 1e-8)
        features['vocab_diversity_ratio'] = real_unique_ratio / (fake_unique_ratio + 1e-8)
        
        improved_features.append(features)
    
    print(f"✅ Created {len(improved_features[0])} improved features")
    return pd.DataFrame(improved_features)

def train_improved_model(improved_features_df, df_original):
    """
    Train models with improved features
    """
    print("🚀 TRAINING IMPROVED MODELS")
    print("=" * 50)
    
    # Create labels: for each pair, predict which file is real (1 or 2)
    y = df_original['real_file_label'].values - 1  # Convert 1,2 to 0,1
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(improved_features_df)
    
    # Train models
    models = {}
    
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_scaled, y)
    models['random_forest'] = rf
    
    # XGBoost
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        xgb_model.fit(X_scaled, y)
        models['xgboost'] = xgb_model
    except:
        print("XGBoost not available, skipping...")
    
    # Evaluate
    from sklearn.model_selection import cross_val_score
    
    print("Model Performance:")
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
        print(f"{name}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Feature importance
    if 'random_forest' in models:
        feature_importance = dict(zip(improved_features_df.columns, 
                                    models['random_forest'].feature_importances_))
        print(f"\nTop 10 Important Features:")
        for feature, importance in sorted(feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {feature}: {importance:.4f}")
    
    return models, scaler, improved_features_df

# Usage functions
def run_complete_analysis(df_original, classifier, X, y, feature_df):
    """Run complete debugging and improvement pipeline"""
    
    # Debug current results
    correlations, X_pca = debug_baseline_results(classifier, X, y, feature_df, df_original)
    
    # Create improved features
    improved_features = create_improved_features(df_original)
    
    # Train improved models
    improved_models, improved_scaler, _ = train_improved_model(improved_features, df_original)
    
    return improved_features, improved_models, improved_scaler

import pandas as pd
from collections import Counter
from var_classifier import VarianceBasedClassifier

# Loading data into memory 
data_file = "data/train_df.csv"
df = pd.read_csv(data_file)

def run_baseline_experiment(df):
    # Initialize classifier
    classifier = VarianceBasedClassifier()
    
    # Prepare data
    X, y, feature_df = classifier.prepare_data(df)
    
    print(f"Data prepared: {len(X)} samples, {len(X.columns)} features")
    print(f"Class distribution: {Counter(y)}")
    
    # Train models
    real_features = X[y == 1]  # Features from real texts only
    classifier.train_isolation_forest(real_features)
    classifier.train_binary_classifiers(X, y)
    
    # Evaluate models
    classifier.evaluate_models(X, y)
    
    # Show feature importance
    classifier.get_feature_importance()
    
    return classifier, X, y, feature_df

classifier, X, y, features = run_baseline_experiment(df)
run_complete_analysis(df, classifier, X, y, features)