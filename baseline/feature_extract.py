import pandas as pd
from collections import Counter
from var_classifier import VarianceBasedClassifier

# Loading data into memory 
data_file = "../data/train_df.csv"
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

classifier, X, y, features = VarianceBasedClassifier.run_baseline_experiment(df)