import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Extracting basic features 
def extract_baseline_features(df):
    # Create real and fake text columns
    df['real_text'] = df.apply(
        lambda row: row['file_1'] if row['real_file_label'] == 1 else row['file_2'], 
        axis=1
    )
    df['fake_text'] = df.apply(
        lambda row: row['file_2'] if row['real_file_label'] == 1 else row['file_1'], 
        axis=1
    )
    
    features = []
    
    for idx, row in df.iterrows():
        real_text = row['real_text']
        fake_text = row['fake_text']
        
        # Basic length features
        real_chars = len(real_text)
        fake_chars = len(fake_text)
        real_words = len(real_text.split())
        fake_words = len(fake_text.split())
        
        # Basic punctuation features
        real_punct = len(re.findall(r'[.!?,:;]', real_text))
        fake_punct = len(re.findall(r'[.!?,:;]', fake_text))
        
        # Basic sentence features
        real_sents = len([s for s in re.split(r'[.!?]+', real_text) if s.strip()])
        fake_sents = len([s for s in re.split(r'[.!?]+', fake_text) if s.strip()])
        
        # Word variance
        real_word_lengths = [len(w) for w in real_text.split()] if real_words > 0 else [0]
        fake_word_lengths = [len(w) for w in fake_text.split()] if fake_words > 0 else [0]
        real_word_var = np.var(real_word_lengths) if len(real_word_lengths) > 1 else 0
        fake_word_var = np.var(fake_word_lengths) if len(fake_word_lengths) > 1 else 0
        
        # Proper nouns
        real_proper = len([w for i, w in enumerate(real_text.split()) 
                          if w and w[0].isupper() and i > 0])
        fake_proper = len([w for i, w in enumerate(fake_text.split()) 
                          if w and w[0].isupper() and i > 0])
        
        # Numbers
        real_numbers = len(re.findall(r'\d+', real_text))
        fake_numbers = len(re.findall(r'\d+', fake_text))
        
        # Select most promising features for baseline
        feature_row = {
            # Length ratios
            'char_ratio': real_chars / (fake_chars + 1),
            'word_ratio': real_words / (fake_words + 1),
            
            # Difference features
            'char_diff': real_chars - fake_chars,
            'word_diff': real_words - fake_words,
            'punct_diff': real_punct - fake_punct,
            'sent_diff': real_sents - fake_sents,
            
            # Density features
            'punct_density_diff': (real_punct / (real_words + 1)) - (fake_punct / (fake_words + 1)),
            'proper_density_diff': (real_proper / (real_words + 1)) - (fake_proper / (fake_words + 1)),
            
            # Variance features
            'word_var_diff': real_word_var - fake_word_var,
            'word_var_ratio': real_word_var / (fake_word_var + 1e-6),
            
            # Content features
            'numbers_diff': real_numbers - fake_numbers,
            'proper_diff': real_proper - fake_proper,
        }
        
        features.append(feature_row)
    
    return pd.DataFrame(features)

def train_baseline_models(X, y, test_size=0.2, random_state=42):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    results = {}
    
    # Logistic Regression
    lr = LogisticRegression(random_state=random_state, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    models['Logistic Regression'] = lr
    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, lr_pred),
        'cv_score': cross_val_score(lr, X_train_scaled, y_train, cv=5).mean()
    }
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    models['Random Forest'] = rf
    results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'cv_score': cross_val_score(rf, X_train_scaled, y_train, cv=5).mean()
    }
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    gb.fit(X_train_scaled, y_train)
    gb_pred = gb.predict(X_test_scaled)
    models['Gradient Boosting'] = gb
    results['Gradient Boosting'] = {
        'accuracy': accuracy_score(y_test, gb_pred),
        'cv_score': cross_val_score(gb, X_train_scaled, y_train, cv=5).mean()
    }
    
    return models, results, scaler, (X_train, X_test, y_train, y_test)

def evaluate_baseline(models, results, X, feature_names):    
    print("BASELINE MODEL PERFORMANCE")
    print("=" * 40)
    
    best_model = None
    best_accuracy = 0
    
    for model_name, metrics in results.items():
        test_acc = metrics['accuracy']
        cv_acc = metrics['cv_score']
        
        print(f"{model_name}:")
        print(f"  Test Accuracy: {test_acc:.3f}")
        print(f"  CV Accuracy:   {cv_acc:.3f}")
        print()
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = model_name
    
    print(f"Best Model: {best_model} ({best_accuracy:.3f} accuracy)")
    
    # Feature importance for best tree-based model
    if best_model in ['Random Forest', 'Gradient Boosting']:
        model = models[best_model]
        if hasattr(model, 'feature_importances_'):
            print(f"\nTop 5 {best_model} Features:")
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:5]:
                print(f"  {feature}: {importance:.3f}")
    
    return best_model, best_accuracy


# Main execution
def run_baseline_experiment(df):    
    print("Extracting baseline features...")
    features_df = extract_baseline_features(df)
    
    print(f"Features extracted: {list(features_df.columns)}")
    print(f"Dataset size: {len(features_df)} samples")
    
    # Create labels (0 = file_1 is real, 1 = file_2 is real)  
    labels = (df['real_file_label'] - 1).values
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Train models
    print("\nTraining baseline models...")
    models, results, scaler, data_splits = train_baseline_models(features_df, labels)
    
    # Evaluate
    best_model, best_accuracy = evaluate_baseline(models, results, features_df, features_df.columns)
    
    # Detailed results for best model
    X_train, X_test, y_train, y_test = data_splits
    X_test_scaled = scaler.transform(X_test)
    best_pred = models[best_model].predict(X_test_scaled)
    
    print(f"\nDetailed {best_model} Results:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, best_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, best_pred))
    
    return models, scaler, features_df.columns

# Data ingestion
train_df = pd.read_csv("data/train_df.csv")
clean_df = train_df.copy()
clean_df.dropna(subset=["file_1", "file_2"], inplace=True)

models, scaler, feature_names = run_baseline_experiment(clean_df)


def extract_test_features(df):    
    # Fill NaN values
    df['file_1'] = df['file_1'].fillna('').astype(str)
    df['file_2'] = df['file_2'].fillna('').astype(str)
    
    features = []
    
    for idx, row in df.iterrows():
        text1 = row['file_1']
        text2 = row['file_2']
        
        # Basic length features
        chars1 = len(text1)
        chars2 = len(text2)
        words1 = len(text1.split())
        words2 = len(text2.split())
        
        # Basic punctuation features
        punct1 = len(re.findall(r'[.!?,:;]', text1))
        punct2 = len(re.findall(r'[.!?,:;]', text2))
        
        # Basic sentence features
        sents1 = len([s for s in re.split(r'[.!?]+', text1) if s.strip()])
        sents2 = len([s for s in re.split(r'[.!?]+', text2) if s.strip()])
        
        # Word variance
        word_lengths1 = [len(w) for w in text1.split()] if words1 > 0 else [0]
        word_lengths2 = [len(w) for w in text2.split()] if words2 > 0 else [0]
        word_var1 = np.var(word_lengths1) if len(word_lengths1) > 1 else 0
        word_var2 = np.var(word_lengths2) if len(word_lengths2) > 1 else 0
        
        # Proper nouns
        proper1 = len([w for i, w in enumerate(text1.split()) 
                      if w and w[0].isupper() and i > 0])
        proper2 = len([w for i, w in enumerate(text2.split()) 
                      if w and w[0].isupper() and i > 0])
        
        # Numbers
        numbers1 = len(re.findall(r'\d+', text1))
        numbers2 = len(re.findall(r'\d+', text2))
        
        # Create features assuming file_1 is "real" and file_2 is "fake"
        # The model will predict if this assumption is correct
        feature_row = {
            # Length ratios (file_1 / file_2)
            'char_ratio': chars1 / (chars2 + 1),
            'word_ratio': words1 / (words2 + 1),
            
            # Difference features (file_1 - file_2)
            'char_diff': chars1 - chars2,
            'word_diff': words1 - words2,
            'punct_diff': punct1 - punct2,
            'sent_diff': sents1 - sents2,
            
            # Density features
            'punct_density_diff': (punct1 / (words1 + 1)) - (punct2 / (words2 + 1)),
            'proper_density_diff': (proper1 / (words1 + 1)) - (proper2 / (words2 + 1)),
            
            # Variance features
            'word_var_diff': word_var1 - word_var2,
            'word_var_ratio': word_var1 / (word_var2 + 1e-6),
            
            # Content features
            'numbers_diff': numbers1 - numbers2,
            'proper_diff': proper1 - proper2,
        }
        
        features.append(feature_row)
    
    return pd.DataFrame(features)


def predict_on_test_data(test_df, best_model, scaler, feature_names):    
    print("Generating predictions on test data...")
    
    # Extract features from test data using same process
    test_features_df = extract_test_features(test_df)
    
    # Ensure feature order matches training
    test_features_df = test_features_df[feature_names]
    
    # Scale features using fitted scaler
    test_features_scaled = scaler.transform(test_features_df)
    
    # Generate predictions
    predictions = best_model.predict(test_features_scaled)
    prediction_probabilities = best_model.predict_proba(test_features_scaled)
    
    # Convert predictions back to original format (1 or 2)
    final_predictions = predictions + 1  # Convert 0,1 back to 1,2
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'prediction': final_predictions,
        'confidence': np.max(prediction_probabilities, axis=1),
        'prob_file1_real': prediction_probabilities[:, 0],
        'prob_file2_real': prediction_probabilities[:, 1]
    })
    
    print(f"Generated {len(results_df)} predictions")
    print(f"Prediction distribution: {np.bincount(final_predictions)}")
    print(f"Average confidence: {results_df['confidence'].mean():.3f}")
    
    return results_df, test_features_df



# Producing test predictions
test_df = pd.read_csv("data/test_df.csv")
res_df = predict_on_test_data(test_df, models["Random Forest"], scaler, feature_names)[0]["prediction"]
res_df.to_csv("predictions_rf.csv")