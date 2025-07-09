import re
import numpy as np
import pandas as pd
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

class VarianceFeatureExtractor:
    """
    Extract variance and consistency features that distinguish real from fake texts
    Based on EDA insight: fake texts have higher variance/inconsistency
    """
    def __init__(self):
        self.feature_names = []
    
    def extract_single_text_features(self, text):
        """Extract variance features from a single text"""
        if pd.isna(text) or text == '' or len(str(text).strip()) < 10:
            return self._get_zero_features()
        
        text = str(text)
        features = {}
        
        # Word level variance
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) > 1:
            word_lengths = [len(w) for w in words]
            features.update({
                'word_length_var': np.var(word_lengths),
                'word_length_std': np.std(word_lengths),
                'word_length_cv': np.std(word_lengths) / (np.mean(word_lengths) + 1e-8),  # Coefficient of variation
                'word_length_range': max(word_lengths) - min(word_lengths),
                'word_length_iqr': np.percentile(word_lengths, 75) - np.percentile(word_lengths, 25),
            })
        else:
            features.update({
                'word_length_var': 0, 'word_length_std': 0, 'word_length_cv': 0,
                'word_length_range': 0, 'word_length_iqr': 0
            })
        
        # === SENTENCE-LEVEL VARIANCE ===
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) > 1:
            sent_lengths = [len(s.split()) for s in sentences]
            sent_char_lengths = [len(s) for s in sentences]
            
            features.update({
                'sent_word_var': np.var(sent_lengths),
                'sent_word_std': np.std(sent_lengths),
                'sent_word_cv': np.std(sent_lengths) / (np.mean(sent_lengths) + 1e-8),
                'sent_char_var': np.var(sent_char_lengths),
                'sent_char_std': np.std(sent_char_lengths),
                'sent_char_cv': np.std(sent_char_lengths) / (np.mean(sent_char_lengths) + 1e-8),
            })
        else:
            features.update({
                'sent_word_var': 0, 'sent_word_std': 0, 'sent_word_cv': 0,
                'sent_char_var': 0, 'sent_char_std': 0, 'sent_char_cv': 0
            })
        
        # === VOCABULARY VARIANCE ===
        word_freq = Counter(words)
        if len(word_freq) > 1:
            freq_values = list(word_freq.values())
            features.update({
                'vocab_freq_var': np.var(freq_values),
                'vocab_freq_std': np.std(freq_values),
                'vocab_freq_cv': np.std(freq_values) / (np.mean(freq_values) + 1e-8),
                'unique_word_ratio': len(word_freq) / len(words),  # Vocabulary diversity
            })
        else:
            features.update({
                'vocab_freq_var': 0, 'vocab_freq_std': 0, 'vocab_freq_cv': 0,
                'unique_word_ratio': 0
            })
        
        # === PUNCTUATION VARIANCE ===
        punct_spaces = self._extract_punctuation_spacing(text)
        if len(punct_spaces) > 1:
            features.update({
                'punct_spacing_var': np.var(punct_spaces),
                'punct_spacing_std': np.std(punct_spaces),
                'punct_spacing_cv': np.std(punct_spaces) / (np.mean(punct_spaces) + 1e-8),
            })
        else:
            features.update({
                'punct_spacing_var': 0, 'punct_spacing_std': 0, 'punct_spacing_cv': 0
            })
        
        # === OUTLIER FEATURES ===
        features.update(self._extract_outlier_features(text, words, sentences))
        
        # === SPECIFICITY FEATURES ===
        features.update(self._extract_specificity_features(text))
        
        return features
    
    def _extract_punctuation_spacing(self, text):
        """Extract spacing patterns around punctuation"""
        punct_pattern = r'(\w)([.!?,:;])(\s*)'
        matches = re.findall(punct_pattern, text)
        return [len(match[2]) for match in matches] if matches else [0]
    
    def _extract_outlier_features(self, text, words, sentences):
        """Extract outlier-based features"""
        features = {}
        
        # Word length outliers
        if len(words) > 3:
            word_lengths = [len(w) for w in words]
            mean_wl, std_wl = np.mean(word_lengths), np.std(word_lengths)
            outliers = [abs(wl - mean_wl) > 2 * std_wl for wl in word_lengths]
            features['word_length_outlier_ratio'] = sum(outliers) / len(outliers)
        else:
            features['word_length_outlier_ratio'] = 0
        
        # Sentence length outliers
        if len(sentences) > 2:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_sl, std_sl = np.mean(sent_lengths), np.std(sent_lengths)
            outliers = [abs(sl - mean_sl) > 2 * std_sl for sl in sent_lengths]
            features['sent_length_outlier_ratio'] = sum(outliers) / len(outliers)
        else:
            features['sent_length_outlier_ratio'] = 0
        
        # Extremely long/short elements
        if words:
            word_lengths = [len(w) for w in words]
            features['very_long_words'] = sum(1 for wl in word_lengths if wl > 15) / len(words)
            features['very_short_words'] = sum(1 for wl in word_lengths if wl <= 2) / len(words)
        else:
            features['very_long_words'] = 0
            features['very_short_words'] = 0
        
        return features
    
    def _extract_specificity_features(self, text):
        """Extract specificity features (proper nouns, precise numbers, etc.)"""
        features = {}
        
        # Proper noun patterns (simple heuristic)
        words = text.split()
        if words:
            # Count capitalized words not at sentence start
            proper_nouns = 0
            for i, word in enumerate(words):
                if word and word[0].isupper() and i > 0 and words[i-1][-1] not in '.!?':
                    proper_nouns += 1
            features['proper_noun_density'] = proper_nouns / len(words)
        else:
            features['proper_noun_density'] = 0
        
        # Number specificity
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            # Precise vs round numbers
            precise_numbers = sum(1 for num in numbers if '.' in num or not num.endswith('0'))
            features['precise_number_ratio'] = precise_numbers / len(numbers)
            features['number_density'] = len(numbers) / len(words) if words else 0
        else:
            features['precise_number_ratio'] = 0
            features['number_density'] = 0
        
        # Technical term density
        tech_terms = ['telescope', 'survey', 'observation', 'stellar', 'galaxy', 'star', 
                     'astronomical', 'magnitude', 'photometric', 'spectroscopic', 
                     'wavelength', 'redshift', 'luminosity', 'parsec', 'virsa', 'vista']
        tech_count = sum(1 for term in tech_terms if term.lower() in text.lower())
        features['tech_term_density'] = tech_count / len(words) if words else 0
        
        # Abbreviation density
        abbrevs = re.findall(r'\b[A-Z]{2,}\b', text)
        features['abbreviation_density'] = len(abbrevs) / len(words) if words else 0
        
        return features
    
    def _get_zero_features(self):
        """Return zero features for empty/invalid texts"""
        return {
            'word_length_var': 0, 'word_length_std': 0, 'word_length_cv': 0,
            'word_length_range': 0, 'word_length_iqr': 0,
            'sent_word_var': 0, 'sent_word_std': 0, 'sent_word_cv': 0,
            'sent_char_var': 0, 'sent_char_std': 0, 'sent_char_cv': 0,
            'vocab_freq_var': 0, 'vocab_freq_std': 0, 'vocab_freq_cv': 0,
            'unique_word_ratio': 0, 'punct_spacing_var': 0, 'punct_spacing_std': 0,
            'punct_spacing_cv': 0, 'word_length_outlier_ratio': 0,
            'sent_length_outlier_ratio': 0, 'very_long_words': 0, 'very_short_words': 0,
            'proper_noun_density': 0, 'precise_number_ratio': 0, 'number_density': 0,
            'tech_term_density': 0, 'abbreviation_density': 0
        }
    
    def extract_pair_features(self, text1, text2):
        """Extract features comparing two texts in a pair"""
        features1 = self.extract_single_text_features(text1)
        features2 = self.extract_single_text_features(text2)
        
        pair_features = {}
        
        # Variance differences between texts
        for key in features1.keys():
            if key.endswith('_var') or key.endswith('_std') or key.endswith('_cv'):
                pair_features[f'{key}_diff'] = abs(features1[key] - features2[key])
                pair_features[f'{key}_ratio'] = (features1[key] + 1e-8) / (features2[key] + 1e-8)
        
        # Which text is more variable?
        variance_features = [k for k in features1.keys() if k.endswith('_var')]
        text1_var_sum = sum(features1[k] for k in variance_features)
        text2_var_sum = sum(features2[k] for k in variance_features)
        
        pair_features.update({
            'text1_total_variance': text1_var_sum,
            'text2_total_variance': text2_var_sum,
            'variance_ratio': (text1_var_sum + 1e-8) / (text2_var_sum + 1e-8),
            'more_variable_text': 1 if text1_var_sum > text2_var_sum else 2,
        })
        
        return pair_features
    
    def extract_features_dataframe(self, df):
        """Extract features for entire dataframe"""
        print("Extracting variance features")
        
        # Generate real_text and fake_text columns
        if 'real_text' not in df.columns:
            df['real_text'] = df.apply(
                lambda row: row['file_1'] if row['real_file_label'] == 1 else row['file_2'], 
                axis=1
            )
            df['fake_text'] = df.apply(
                lambda row: row['file_2'] if row['real_file_label'] == 1 else row['file_1'], 
                axis=1
            )
        
        features_list = []
        
        for idx, row in df.iterrows():
            # Extract features for both texts
            real_features = self.extract_single_text_features(row['real_text'])
            fake_features = self.extract_single_text_features(row['fake_text'])
            
            # Add prefixes to distinguish real vs fake
            real_prefixed = {f'real_{k}': v for k, v in real_features.items()}
            fake_prefixed = {f'fake_{k}': v for k, v in fake_features.items()}
            
            # Extract pair features
            pair_features = self.extract_pair_features(row['real_text'], row['fake_text'])
            
            # Combine all features
            combined_features = {**real_prefixed, **fake_prefixed, **pair_features}
            features_list.append(combined_features)
        
        feature_df = pd.DataFrame(features_list)
        
        # Store feature names for later use
        self.feature_names = list(feature_df.columns)
        
        print(f"✅ Extracted {len(self.feature_names)} variance features")
        return feature_df

