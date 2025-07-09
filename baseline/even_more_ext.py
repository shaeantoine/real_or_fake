import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Try to import xgboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class BackToBasicsClassifier:
    """
    Back to basics: Take what worked (0.60 AUC) and improve it carefully
    Focus on the features that actually matter
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        
    def extract_focused_features(self, df):
        """Extract only the features that showed promise in previous experiments"""
        print("🎯 Extracting Focused Features (Based on What Actually Worked)...")
        
        features_list = []
        
        for idx, row in df.iterrows():
            file_1 = str(row['file_1']) if not pd.isna(row['file_1']) else ''
            file_2 = str(row['file_2']) if not pd.isna(row['file_2']) else ''
            
            features = {}
            
            # === TOP PERFORMING FEATURES FROM PREVIOUS EXPERIMENTS ===
            # Based on your results: punct_density_diff, sent_diff_abs, word_var_diff were top performers
            
            # 1. PUNCTUATION FEATURES (your #1 performer)
            features.update(self._extract_simple_punct_features(file_1, file_2))
            
            # 2. SENTENCE STRUCTURE FEATURES (your #2 performer) 
            features.update(self._extract_simple_structure_features(file_1, file_2))
            
            # 3. WORD VARIANCE FEATURES (your #3 performer)
            features.update(self._extract_simple_variance_features(file_1, file_2))
            
            # 4. CONTENT DENSITY FEATURES (showed promise)
            features.update(self._extract_simple_content_features(file_1, file_2))
            
            # 5. BASIC LENGTH FEATURES (foundation)
            features.update(self._extract_simple_length_features(file_1, file_2))
            
            features_list.append(features)
        
        feature_df = pd.DataFrame(features_list)
        self.feature_names = list(feature_df.columns)
        
        print(f"✅ Extracted {len(self.feature_names)} focused features")
        return feature_df
    
    def _extract_simple_punct_features(self, text1, text2):
        """Simple but effective punctuation features"""
        def safe_punct_analysis(text):
            try:
                if pd.isna(text) or str(text).strip() == '':
                    return {'punct_count': 0, 'punct_density': 0, 'period_count': 0, 'comma_count': 0}
                
                text = str(text)
                words = text.split()
                word_count = len(words)
                
                # Count punctuation
                periods = len(re.findall(r'\.', text))
                commas = len(re.findall(r',', text))
                all_punct = len(re.findall(r'[.!?,:;]', text))
                
                return {
                    'punct_count': all_punct,
                    'punct_density': all_punct / (word_count + 1e-8),
                    'period_count': periods,
                    'comma_count': commas,
                }
            except Exception as e:
                print(f"Warning: Error in punctuation analysis: {e}")
                return {'punct_count': 0, 'punct_density': 0, 'period_count': 0, 'comma_count': 0}
        
        punct1 = safe_punct_analysis(text1)
        punct2 = safe_punct_analysis(text2)
        
        return {
            # The winning feature from your experiments
            'punct_density_diff': abs(punct1['punct_density'] - punct2['punct_density']),
            'punct_count_ratio': punct1['punct_count'] / (punct2['punct_count'] + 1e-8),
            'period_ratio': punct1['period_count'] / (punct2['period_count'] + 1e-8),
            'comma_ratio': punct1['comma_count'] / (punct2['comma_count'] + 1e-8),
        }
    
    def _extract_simple_structure_features(self, text1, text2):
        """Simple but effective sentence structure features"""
        def safe_structure_analysis(text):
            try:
                if pd.isna(text) or str(text).strip() == '':
                    return {'sent_count': 0, 'avg_sent_len': 0}
                
                text = str(text)
                sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
                
                if not sentences:
                    return {'sent_count': 0, 'avg_sent_len': 0}
                
                sent_lengths = [len(s.split()) for s in sentences]
                
                return {
                    'sent_count': len(sentences),
                    'avg_sent_len': np.mean(sent_lengths) if sent_lengths else 0,
                }
            except Exception as e:
                print(f"Warning: Error in structure analysis: {e}")
                return {'sent_count': 0, 'avg_sent_len': 0}
        
        struct1 = safe_structure_analysis(text1)
        struct2 = safe_structure_analysis(text2)
        
        return {
            # The winning feature from your experiments
            'sent_diff_abs': abs(struct1['sent_count'] - struct2['sent_count']),
            'sent_count_ratio': struct1['sent_count'] / (struct2['sent_count'] + 1e-8),
            'avg_sent_len_ratio': struct1['avg_sent_len'] / (struct2['avg_sent_len'] + 1e-8),
        }
    
    def _extract_simple_variance_features(self, text1, text2):
        """Simple but effective word variance features"""
        def safe_variance_analysis(text):
            try:
                if pd.isna(text) or str(text).strip() == '':
                    return {'word_len_var': 0, 'word_len_std': 0}
                
                text = str(text)
                words = text.split()
                
                if len(words) <= 1:
                    return {'word_len_var': 0, 'word_len_std': 0}
                
                word_lengths = [len(w) for w in words]
                
                return {
                    'word_len_var': np.var(word_lengths),
                    'word_len_std': np.std(word_lengths),
                }
            except Exception as e:
                print(f"Warning: Error in variance analysis: {e}")
                return {'word_len_var': 0, 'word_len_std': 0}
        
        var1 = safe_variance_analysis(text1)
        var2 = safe_variance_analysis(text2)
        
        return {
            # The winning feature from your experiments
            'word_var_diff': abs(var1['word_len_var'] - var2['word_len_var']),
            'word_std_ratio': var1['word_len_std'] / (var2['word_len_std'] + 1e-8),
            'higher_variance_text': 1 if var1['word_len_var'] > var2['word_len_var'] else 2,
        }
    
    def _extract_simple_content_features(self, text1, text2):
        """Simple content quality features that showed promise"""
        def safe_content_analysis(text):
            try:
                if pd.isna(text) or str(text).strip() == '':
                    return {
                        'proper_nouns': 0, 'numbers': 0, 'tech_terms': 0, 'proper_density': 0
                    }
                
                text = str(text)
                words = text.split()
                
                if not words:
                    return {
                        'proper_nouns': 0, 'numbers': 0, 'tech_terms': 0, 'proper_density': 0
                    }
                
                # Proper nouns (simple heuristic)
                proper_nouns = 0
                for i, word in enumerate(words):
                    if word and len(word) > 0 and word[0].isupper() and i > 0 and len(words[i-1]) > 0 and words[i-1][-1] not in '.!?':
                        proper_nouns += 1
                
                # Numbers
                numbers = len(re.findall(r'\d+\.?\d*', text))
                
                # Technical terms (limited list of most common)
                tech_terms = ['telescope', 'survey', 'observation', 'stellar', 'galaxy', 'star']
                tech_count = sum(1 for term in tech_terms if term.lower() in text.lower())
                
                return {
                    'proper_nouns': proper_nouns,
                    'numbers': numbers,
                    'tech_terms': tech_count,
                    'proper_density': proper_nouns / len(words),
                }
            except Exception as e:
                print(f"Warning: Error in content analysis: {e}")
                return {
                    'proper_nouns': 0, 'numbers': 0, 'tech_terms': 0, 'proper_density': 0
                }
        
        content1 = safe_content_analysis(text1)
        content2 = safe_content_analysis(text2)
        
        return {
            'proper_nouns_diff': abs(content1['proper_nouns'] - content2['proper_nouns']),
            'proper_density_1': content1['proper_density'],
            'proper_density_2': content2['proper_density'],
            'numbers_diff': abs(content1['numbers'] - content2['numbers']),
            'tech_terms_diff': abs(content1['tech_terms'] - content2['tech_terms']),
        }
    
    def _extract_simple_length_features(self, text1, text2):
        """Basic length features as foundation"""
        def safe_length_analysis(text):
            try:
                if pd.isna(text) or str(text).strip() == '':
                    return {'char_count': 0, 'word_count': 0}
                
                text = str(text)
                return {
                    'char_count': len(text),
                    'word_count': len(text.split()),
                }
            except Exception as e:
                print(f"Warning: Error in length analysis: {e}")
                return {'char_count': 0, 'word_count': 0}
        
        len1 = safe_length_analysis(text1)
        len2 = safe_length_analysis(text2)
        
        return {
            'char_ratio': len1['char_count'] / (len2['char_count'] + 1e-8),
            'word_ratio': len1['word_count'] / (len2['word_count'] + 1e-8),
            'char_diff_abs': abs(len1['char_count'] - len2['char_count']),
            'word_diff_abs': abs(len1['word_count'] - len2['word_count']),
        }
    
    def train_focused_models(self, X, y):
        """Train models with focus on what works"""
        print("🎯 Training Focused Models (Less Complexity, Better Performance)...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # 1. Random Forest (worked well before)
        rf = RandomForestClassifier(
            n_estimators=150,  # Slightly more than before
            max_depth=8,       # Not too deep to avoid overfitting
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_scaled, y)
        self.models['random_forest'] = rf
        
        # 2. Gradient Boosting (your best performer at 0.60)
        gb = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,       # Keep shallow to avoid overfitting
            learning_rate=0.1,
            random_state=42
        )
        gb.fit(X_scaled, y)
        self.models['gradient_boosting'] = gb
        
        # 3. XGBoost if available
        if XGBOOST_AVAILABLE:
            xgb_model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            xgb_model.fit(X_scaled, y)
            self.models['xgboost'] = xgb_model
        
        # 4. Logistic Regression for interpretability
        lr = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000,
            C=1.0  # Not too much regularization
        )
        lr.fit(X_scaled, y)
        self.models['logistic_regression'] = lr
        
        print(f"✅ Trained {len(self.models)} focused models")
        
    def evaluate_models(self, X, y):
        """Evaluate models"""
        print("📊 Evaluating Focused Models...")
        
        X_scaled = self.scaler.transform(X)
        results = {}
        
        for name, model in self.models.items():
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
            results[name] = {
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std()
            }
            print(f"{name}: AUC = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return results
    
    def get_feature_importance(self, top_n=10):
        """Get feature importance from models"""
        print(f"\n🔍 Top {top_n} Most Important Features:")
        print("=" * 50)
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(self.feature_names, model.feature_importances_))
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                print(f"\n{model_name.title()}:")
                for feature, importance in sorted_features[:top_n]:
                    print(f"  {feature}: {importance:.4f}")

def run_focused_experiment(df):
    """Run the focused experiment - back to what works"""
    print("🎯 BACK TO BASICS - FOCUSED APPROACH")
    print("=" * 50)
    print("Goal: Build on the 0.60 AUC success, not over-engineer")
    print("=" * 50)
    
    # Initialize classifier
    classifier = BackToBasicsClassifier()
    
    # Extract focused features
    features_df = classifier.extract_focused_features(df)
    
    # Prepare labels
    y = (df['real_file_label'] - 1).values
    
    print(f"📊 Dataset: {len(features_df)} samples, {len(features_df.columns)} features")
    print(f"📊 Label distribution: {Counter(y)}")
    
    # Train focused models
    classifier.train_focused_models(features_df, y)
    
    # Evaluate models
    results = classifier.evaluate_models(features_df, y)
    
    # Show feature importance
    classifier.get_feature_importance()
    
    # Target analysis
    best_score = max(result['cv_auc_mean'] for result in results.values())
    print(f"\n🎯 PERFORMANCE ANALYSIS:")
    print(f"Best model AUC: {best_score:.4f}")
    print(f"Previous best (Enhanced): 0.6000")
    print(f"Change: {best_score - 0.6000:+.4f}")
    
    if best_score >= 0.65:
        print("🎉 IMPROVEMENT ACHIEVED!")
    elif best_score >= 0.58:
        print("✅ STABLE PERFORMANCE - Good foundation")
    else:
        print("⚠️  Performance dropped - need to investigate")
    
    print("\n✅ FOCUSED EXPERIMENT COMPLETE!")
    print("=" * 50)
    
    return classifier, features_df, y, results

# Usage:
data_file = "data/train_df.csv"
df = pd.read_csv(data_file)
classifier, features, labels, results = run_focused_experiment(df)