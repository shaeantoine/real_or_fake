import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

# Try to import xgboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class EnhancedPairClassifier:
    """
    Enhanced classifier that focuses on pair-wise differences
    Based on insight that ratios/differences work better than absolute features
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        
    def extract_comprehensive_pair_features(self, df):
        """Extract comprehensive pair-wise features"""        
        features_list = []
        
        for idx, row in df.iterrows():
            # Get texts
            file_1 = str(row['file_1']) if not pd.isna(row['file_1']) else ''
            file_2 = str(row['file_2']) if not pd.isna(row['file_2']) else ''
            
            features = {}
            
            # === BASIC LENGTH & STRUCTURE FEATURES ===
            features.update(self._extract_length_features(file_1, file_2))
            
            # === VARIANCE & CONSISTENCY FEATURES ===
            features.update(self._extract_variance_features(file_1, file_2))
            
            # === CONTENT QUALITY FEATURES ===
            features.update(self._extract_content_features(file_1, file_2))
            
            # === LINGUISTIC FEATURES ===
            features.update(self._extract_linguistic_features(file_1, file_2))
            
            # === DOMAIN-SPECIFIC FEATURES ===
            features.update(self._extract_domain_features(file_1, file_2))
            
            features_list.append(features)
        
        feature_df = pd.DataFrame(features_list)
        self.feature_names = list(feature_df.columns)
        
        print(f"✅ Extracted {len(self.feature_names)} enhanced features")
        return feature_df
    
    def _extract_length_features(self, text1, text2):
        """Extract length-based comparison features"""
        # Character counts
        len1, len2 = len(text1), len(text2)
        
        # Word counts
        words1 = text1.split()
        words2 = text2.split()
        wc1, wc2 = len(words1), len(words2)
        
        # Sentence counts
        sents1 = len([s for s in re.split(r'[.!?]+', text1) if s.strip()])
        sents2 = len([s for s in re.split(r'[.!?]+', text2) if s.strip()])
        
        return {
            'char_ratio': len1 / (len2 + 1e-8),
            'char_diff_abs': abs(len1 - len2),
            'char_diff_rel': abs(len1 - len2) / (max(len1, len2) + 1e-8),
            
            'word_ratio': wc1 / (wc2 + 1e-8),
            'word_diff_abs': abs(wc1 - wc2),
            'word_diff_rel': abs(wc1 - wc2) / (max(wc1, wc2) + 1e-8),
            
            'sent_ratio': sents1 / (sents2 + 1e-8),
            'sent_diff_abs': abs(sents1 - sents2),
            
            'avg_word_len_1': np.mean([len(w) for w in words1]) if words1 else 0,
            'avg_word_len_2': np.mean([len(w) for w in words2]) if words2 else 0,
            'avg_word_len_ratio': (np.mean([len(w) for w in words1]) if words1 else 0) / 
                                 (np.mean([len(w) for w in words2]) if words2 else 1e-8),
        }
    
    def _extract_variance_features(self, text1, text2):
        """Extract variance and consistency comparison features"""
        def get_text_variance_stats(text):
            words = text.split()
            if len(words) <= 1:
                return {'word_len_var': 0, 'word_len_std': 0, 'word_len_cv': 0}
            
            word_lens = [len(w) for w in words]
            mean_len = np.mean(word_lens)
            
            return {
                'word_len_var': np.var(word_lens),
                'word_len_std': np.std(word_lens),
                'word_len_cv': np.std(word_lens) / (mean_len + 1e-8)
            }
        
        stats1 = get_text_variance_stats(text1)
        stats2 = get_text_variance_stats(text2)
        
        return {
            'word_var_ratio': stats1['word_len_var'] / (stats2['word_len_var'] + 1e-8),
            'word_var_diff': abs(stats1['word_len_var'] - stats2['word_len_var']),
            'word_std_ratio': stats1['word_len_std'] / (stats2['word_len_std'] + 1e-8),
            'word_cv_ratio': stats1['word_len_cv'] / (stats2['word_len_cv'] + 1e-8),
            
            # Which text is more variable?
            'text1_more_variable': 1 if stats1['word_len_var'] > stats2['word_len_var'] else 0,
            
            # Vocabulary diversity
            'vocab_diversity_1': len(set(text1.lower().split())) / (len(text1.split()) + 1e-8),
            'vocab_diversity_2': len(set(text2.lower().split())) / (len(text2.split()) + 1e-8),
            'vocab_diversity_ratio': (len(set(text1.lower().split())) / (len(text1.split()) + 1e-8)) /
                                   (len(set(text2.lower().split())) / (len(text2.split()) + 1e-8) + 1e-8),
        }
    
    def _extract_content_features(self, text1, text2):
        """Extract content quality comparison features"""
        # Numbers
        nums1 = len(re.findall(r'\d+\.?\d*', text1))
        nums2 = len(re.findall(r'\d+\.?\d*', text2))
        
        # Proper nouns (simple heuristic)
        def count_proper_nouns(text):
            words = text.split()
            count = 0
            for i, word in enumerate(words):
                if word and word[0].isupper() and i > 0 and words[i-1][-1] not in '.!?':
                    count += 1
            return count
        
        proper1 = count_proper_nouns(text1)
        proper2 = count_proper_nouns(text2)
        
        # Precise numbers (with decimals)
        precise1 = len([n for n in re.findall(r'\d+\.?\d*', text1) if '.' in n])
        precise2 = len([n for n in re.findall(r'\d+\.?\d*', text2) if '.' in n])
        
        return {
            'numbers_ratio': nums1 / (nums2 + 1e-8),
            'numbers_diff': abs(nums1 - nums2),
            'numbers_density_1': nums1 / (len(text1.split()) + 1e-8),
            'numbers_density_2': nums2 / (len(text2.split()) + 1e-8),
            
            'proper_nouns_ratio': proper1 / (proper2 + 1e-8),
            'proper_nouns_diff': abs(proper1 - proper2),
            'proper_density_1': proper1 / (len(text1.split()) + 1e-8),
            'proper_density_2': proper2 / (len(text2.split()) + 1e-8),
            
            'precise_nums_ratio': precise1 / (precise2 + 1e-8),
            'precise_nums_diff': abs(precise1 - precise2),
        }
    
    def _extract_linguistic_features(self, text1, text2):
        """Extract linguistic pattern features"""
        # Punctuation patterns
        punct1 = len(re.findall(r'[.!?,:;]', text1))
        punct2 = len(re.findall(r'[.!?,:;]', text2))
        
        # Capitalization patterns
        caps1 = len(re.findall(r'[A-Z]', text1))
        caps2 = len(re.findall(r'[A-Z]', text2))
        
        # Parentheses/quotes (might indicate citations or technical notation)
        parens1 = len(re.findall(r'[()"\']', text1))
        parens2 = len(re.findall(r'[()"\']', text2))
        
        return {
            'punct_ratio': punct1 / (punct2 + 1e-8),
            'punct_density_diff': abs(punct1 / (len(text1) + 1e-8) - punct2 / (len(text2) + 1e-8)),
            
            'caps_ratio': caps1 / (caps2 + 1e-8),
            'caps_density_diff': abs(caps1 / (len(text1) + 1e-8) - caps2 / (len(text2) + 1e-8)),
            
            'parens_ratio': parens1 / (parens2 + 1e-8),
            'parens_diff': abs(parens1 - parens2),
        }
    
    def _extract_domain_features(self, text1, text2):
        """Extract astronomy/science domain-specific features"""
        # Technical terms
        tech_terms = ['telescope', 'survey', 'observation', 'stellar', 'galaxy', 'star', 
                     'astronomical', 'magnitude', 'photometric', 'spectroscopic', 
                     'wavelength', 'redshift', 'luminosity', 'parsec', 'virsa', 'vista',
                     'eso', 'nasa', 'infrared', 'visible', 'catalog', 'dataset', 'archive',
                     'calibrated', 'epochs', 'angular', 'resolution']
        
        tech1 = sum(1 for term in tech_terms if term.lower() in text1.lower())
        tech2 = sum(1 for term in tech_terms if term.lower() in text2.lower())
        
        # Abbreviations
        abbrevs1 = len(re.findall(r'\b[A-Z]{2,}\b', text1))
        abbrevs2 = len(re.findall(r'\b[A-Z]{2,}\b', text2))
        
        # Scientific units
        units = ['km', 'pc', 'kpc', 'Mpc', 'ly', 'AU', 'deg', 'arcmin', 'arcsec', 
                'mag', 'nm', 'μm', 'mm', 'cm', 'TB', 'GB', 'MB', 'petabyte']
        
        units1 = sum(1 for unit in units if unit in text1)
        units2 = sum(1 for unit in units if unit in text2)
        
        return {
            'tech_terms_ratio': tech1 / (tech2 + 1e-8),
            'tech_terms_diff': abs(tech1 - tech2),
            'tech_density_1': tech1 / (len(text1.split()) + 1e-8),
            'tech_density_2': tech2 / (len(text2.split()) + 1e-8),
            
            'abbrevs_ratio': abbrevs1 / (abbrevs2 + 1e-8),
            'abbrevs_diff': abs(abbrevs1 - abbrevs2),
            
            'units_ratio': units1 / (units2 + 1e-8),
            'units_diff': abs(units1 - units2),
            
            # Combined domain score
            'domain_score_1': (tech1 + abbrevs1 + units1) / (len(text1.split()) + 1e-8),
            'domain_score_2': (tech2 + abbrevs2 + units2) / (len(text2.split()) + 1e-8),
            'domain_score_ratio': ((tech1 + abbrevs1 + units1) / (len(text1.split()) + 1e-8)) / 
                                 ((tech2 + abbrevs2 + units2) / (len(text2.split()) + 1e-8) + 1e-8),
        }
    
    def train_models(self, X, y):
        """Train multiple models with enhanced features"""
        print("🤖 Training Enhanced Models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Logistic Regression (interpretable baseline)
        lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        lr.fit(X_scaled, y)
        self.models['logistic_regression'] = lr
        
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42, 
            class_weight='balanced'
        )
        rf.fit(X_scaled, y)
        self.models['random_forest'] = rf
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        gb.fit(X_scaled, y)
        self.models['gradient_boosting'] = gb
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=1
        )
        xgb_model.fit(X_scaled, y)
        self.models['xgboost'] = xgb_model
        
        print(f"Trained {len(self.models)} models")
        
    def evaluate_models(self, X, y):
        """Evaluate all models with cross-validation"""
        print("📈 Evaluating Enhanced Models...")
        
        X_scaled = self.scaler.transform(X)
        results = {}
        
        for name, model in self.models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
            results[name] = {
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"{name.title()}: AUC = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
        return results
    
    def get_feature_importance(self, top_n=15):
        """Get feature importance from tree-based models"""
        print(f"\n🔍 Top {top_n} Most Important Features:")
        print("=" * 60)
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(self.feature_names, model.feature_importances_))
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                print(f"\n{model_name.title()}:")
                for feature, importance in sorted_features[:top_n]:
                    print(f"  {feature}: {importance:.4f}")
            elif hasattr(model, 'coef_'):
                # For logistic regression, use absolute coefficients
                importance_dict = dict(zip(self.feature_names, abs(model.coef_[0])))
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                print(f"\n{model_name.title()} (absolute coefficients):")
                for feature, importance in sorted_features[:top_n]:
                    print(f"  {feature}: {importance:.4f}")
    
    def predict_which_is_real(self, file_1_text, file_2_text):
        """Predict which file is real (1 or 2) for a new pair"""
        # Create a temporary dataframe
        temp_df = pd.DataFrame({
            'file_1': [file_1_text],
            'file_2': [file_2_text],
            'real_file_label': [1]  # Dummy label
        })
        
        # Extract features
        features = self.extract_comprehensive_pair_features(temp_df)
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred_proba = model.predict_proba(features_scaled)[0]
            predictions[name] = 1 if pred_proba[0] > pred_proba[1] else 2
            probabilities[name] = {
                'file_1_real_prob': pred_proba[0],
                'file_2_real_prob': pred_proba[1]
            }
        
        return predictions, probabilities

def run_enhanced_experiment(df):
    print("ENHANCED PAIR-WISE CLASSIFICATION")
    print("=" * 60)
    # Initialize classifier
    classifier = EnhancedPairClassifier()
    
    # Extract enhanced features
    features_df = classifier.extract_comprehensive_pair_features(df)
    
    # Prepare labels (0 = file_1 is real, 1 = file_2 is real)
    y = (df['real_file_label'] - 1).values
    
    print(f"📊 Dataset: {len(features_df)} samples, {len(features_df.columns)} features")
    print(f"📊 Label distribution: {Counter(y)}")
    
    # Train models
    classifier.train_models(features_df, y)
    
    # Evaluate models
    results = classifier.evaluate_models(features_df, y)
    
    # Show feature importance
    classifier.get_feature_importance()
    
    return classifier, features_df, y, results

# Usage:
data_file = "data/train_df.csv"
df = pd.read_csv(data_file)
classifier, features, labels, results = run_enhanced_experiment(df)

test_data_file = "data/test_df.csv"
test_df = pd.read_csv(test_data_file)

predictions = []
for idx, row in test_df.iterrows():
    prediction = classifier.predict_which_is_real(row["file_1"], row["file_2"])[0]
    predictions.append((idx, prediction["gradient_boosting"]))

output_csv = "predictions_GB.csv"
df_pred = pd.DataFrame(predictions)
df_pred.to_csv(output_csv, index=False)