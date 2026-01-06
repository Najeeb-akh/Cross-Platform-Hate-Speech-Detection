#!/usr/bin/env python3
"""
Feature Group Importance Comparison Across Datasets

This script performs ablation studies to compare the importance of different feature 
groups across three datasets (Reddit, 4chan, Twitter) for hate speech detection.

Feature Groups:
    - Lexical: Surface-level text statistics (word count, length, capitalization, etc.)
    - Hate Lexicon: Hate word detection features
    - TF-IDF: Term frequency-inverse document frequency features (unigrams and bigrams)
    - Sentiment: Sentiment polarity and subjectivity features
    - Embedding: Semantic features including sentence embeddings, POS ratios, and 
                 experimental embedding-based features

Experiments:
    The script runs 11 experiments per dataset:
    - A_FULL: All features combined
    - B-F: Single feature group only (one group at a time)
    - G-K: Leave-one-group-out ablations (all features except one group)

Model:
    Uses Logistic Regression classifier to train models and evaluate performance metrics
    (F1, accuracy, precision, recall, ROC-AUC).

Outputs:
    - Per-dataset results CSV files
    - Combined results CSV
    - Visualization plots (F1 comparison, Delta F1, heatmaps, rankings)
    - Markdown report summarizing findings

Usage:
    python scripts/feature_group_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.features import TextFeatureExtractor, load_dataset
from src.utils import set_seed, create_run_dir, save_json

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def identify_feature_groups(feature_columns):
    """
    Identify which features belong to which group based on column names.
    
    Returns:
        dict: Mapping of group names to feature column names
    """
    groups = {
        'lexical': [],
        'hate_lexicon': [],
        'tfidf': [],
        'sentiment': [],
        'embedding': [],
    }
    
    for col in feature_columns:
        if col == 'label':
            continue
        elif col.startswith('tfidf_'):
            groups['tfidf'].append(col)
        elif col.startswith('sentiment_') or col in [
            'sentiment_polarity',
            'sentiment_subjectivity',
            'sentiment_magnitude',
            'sentiment_negative',
            'sentiment_neutral',
            'sentiment_positive',
            'sentiment_very_negative',
            'sentiment_very_positive',
            'highly_subjective',
            'highly_objective',
        ]:
            groups['sentiment'].append(col)
        elif col in ['hate_word_count', 'hate_word_ratio', 'contains_hate_word']:
            groups['hate_lexicon'].append(col)
        elif col in [
            'num_words',
            'avg_word_len',
            'uppercase_ratio',
            'exclamation_ratio',
            'question_ratio',
            'period_ratio',
            'punctuation_count',
            'multiple_count',
            'char_count',
            'repeated_chars',
            'unique_words',
            'word_repetition',
            'longest_word_length',
            'shortest_word_length',
            'language_complexity_index',
        ]:
            groups['lexical'].append(col)
        elif col in ['noun_ratio', 'verb_ratio', 'adj_ratio', 'they_ratio', 'us_ratio', 'you_ratio']:
            groups['embedding'].append(col)
        elif col.startswith('embed_'):
            groups['embedding'].append(col)
        elif col in ['hate_lexicon_similarity', 'embedding_distance_to_neutral']:
            groups['embedding'].append(col)
        else:
            groups['embedding'].append(col)
    
    return groups


def train_model_with_features(X_train, y_train, X_test, y_test, feature_cols, random_state=42):
    """
    Train a Logistic Regression model with specified features.
    
    Returns:
        tuple: (metrics dict, trained model)
    """
    X_train_subset = X_train[feature_cols].copy()
    X_test_subset = X_test[feature_cols].copy()
    
    for col in X_train_subset.columns:
        if hasattr(X_train_subset[col], 'sparse'):
            X_train_subset[col] = X_train_subset[col].sparse.to_dense()
    for col in X_test_subset.columns:
        if hasattr(X_test_subset[col], 'sparse'):
            X_test_subset[col] = X_test_subset[col].sparse.to_dense()
    
    X_train_array = X_train_subset.values
    X_test_array = X_test_subset.values
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight='balanced',
        solver='saga',
        n_jobs=1
    )
    model.fit(X_train_array, y_train)
    
    y_pred = model.predict(X_test_array)
    y_proba = model.predict_proba(X_test_array)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0,
        'n_features': len(feature_cols),
    }
    
    return metrics, model


def run_ablation_experiments(dataset_name, df, random_state=42):
    """
    Run feature ablation experiments on a single dataset.
    
    Experiments:
        A. FULL - All features
        B. TF-IDF ONLY - Only TF-IDF features
        C. NO TF-IDF - Lexical + Semantic only
        D. NO SEMANTIC - TF-IDF + Lexical only
        E. NO LEXICAL - TF-IDF + Semantic only
        F. LEXICAL ONLY - Only lexical features
        G. SEMANTIC ONLY - Only semantic features
    
    Returns:
        DataFrame with results for all experiments
    """
    print(f"\n{'='*80}")
    print(f"Running Feature Ablation on {dataset_name.upper()}")
    print(f"{'='*80}\n")
    
    # Split data
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=random_state,
        stratify=df['label']
    )
    
    print(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")
    print(f"Train positive ratio: {train_df['label'].mean():.3f}")
    print(f"Test positive ratio: {test_df['label'].mean():.3f}\n")
    
    print("Extracting features...")
    extractor = TextFeatureExtractor()
    train_features = extractor.fit_transform(train_df.copy(), source_name=dataset_name)
    test_features = extractor.transform(test_df.copy(), source_name=dataset_name)
    
    print(f"Feature matrix shape: {train_features.shape}")
    
    feature_cols = [col for col in train_features.columns if col != 'label']
    groups = identify_feature_groups(train_features.columns)
    
    print(f"\nFeature groups:")
    for group_name, group_cols in groups.items():
        print(f"  {group_name}: {len(group_cols)} features")
    
    X_train = train_features.drop('label', axis=1).fillna(0)
    y_train = train_features['label'].values
    X_test = test_features.drop('label', axis=1).fillna(0)
    y_test = test_features['label'].values
    
    experiments = {
        'A_FULL': feature_cols,
        'B_TFIDF_ONLY': groups['tfidf'],
        'C_LEXICAL_ONLY': groups['lexical'],
        'D_HATE_LEXICON_ONLY': groups['hate_lexicon'],
        'E_SENTIMENT_ONLY': groups['sentiment'],
        'F_EMBEDDING_ONLY': groups['embedding'],
        'G_NO_TFIDF': groups['lexical'] + groups['hate_lexicon'] + groups['sentiment'] + groups['embedding'],
        'H_NO_LEXICAL': groups['hate_lexicon'] + groups['sentiment'] + groups['embedding'] + groups['tfidf'],
        'I_NO_HATE_LEXICON': groups['lexical'] + groups['sentiment'] + groups['embedding'] + groups['tfidf'],
        'J_NO_SENTIMENT': groups['lexical'] + groups['hate_lexicon'] + groups['embedding'] + groups['tfidf'],
        'K_NO_EMBEDDING': groups['lexical'] + groups['hate_lexicon'] + groups['sentiment'] + groups['tfidf'],
    }
    
    results = {}
    models = {}
    
    print(f"\n{'='*80}")
    print("Running Experiments")
    print(f"{'='*80}\n")
    
    for exp_name, exp_features in experiments.items():
        if len(exp_features) == 0:
            print(f"⚠️  Skipping {exp_name}: No features in this group")
            continue
        
        print(f"Experiment {exp_name}: {len(exp_features)} features")
        metrics, model = train_model_with_features(
            X_train, y_train, X_test, y_test,
            exp_features, random_state=random_state
        )
        
        results[exp_name] = metrics
        models[exp_name] = model
        
        print(f"  F1: {metrics['f1']:.4f} | Accuracy: {metrics['accuracy']:.4f} | "
              f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
    
    results_df = pd.DataFrame(results).T
    results_df['dataset'] = dataset_name
    results_df['experiment'] = results_df.index
    
    return results_df, models, groups


def compute_delta_f1(results_df):
    """
    Compute ΔF1 for each feature group.
    
    ΔF1 = F1(FULL) - F1(without that group)
    """
    full_f1 = results_df.loc['A_FULL', 'f1']
    
    delta_f1 = {
        'TF-IDF': full_f1 - results_df.loc['G_NO_TFIDF', 'f1'],
        'Lexical': full_f1 - results_df.loc['H_NO_LEXICAL', 'f1'],
        'Hate Lexicon': full_f1 - results_df.loc['I_NO_HATE_LEXICON', 'f1'],
        'Sentiment': full_f1 - results_df.loc['J_NO_SENTIMENT', 'f1'],
        'Embedding': full_f1 - results_df.loc['K_NO_EMBEDDING', 'f1'],
    }
    
    return delta_f1


def plot_comparison_across_datasets(all_results, save_dir):
    """
    Create comprehensive comparison plots across all datasets.
    """
    print(f"\n{'='*80}")
    print("Generating Comparison Plots")
    print(f"{'='*80}\n")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_data = all_results.copy()
    plot_data['experiment_clean'] = plot_data['experiment'].str.replace('_', ' ')
    
    datasets = plot_data['dataset'].unique()
    experiments = sorted(plot_data['experiment'].unique())
    
    x = np.arange(len(experiments))
    width = 0.25
    
    for i, dataset in enumerate(datasets):
        dataset_data = plot_data[plot_data['dataset'] == dataset].set_index('experiment')
        f1_scores = [dataset_data.loc[exp, 'f1'] if exp in dataset_data.index else 0 
                     for exp in experiments]
        ax.bar(x + i * width, f1_scores, width, label=dataset.capitalize(), alpha=0.8)
    
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Feature Group Ablation: F1 Scores Across Datasets', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([exp.replace('_', '\n') for exp in experiments], rotation=0)
    ax.legend(title='Dataset', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'f1_comparison_all_experiments.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: f1_comparison_all_experiments.png")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    delta_data = []
    for dataset in datasets:
        dataset_results = all_results[all_results['dataset'] == dataset].set_index('experiment')
        delta_f1 = compute_delta_f1(dataset_results)
        for group, value in delta_f1.items():
            delta_data.append({
                'dataset': dataset,
                'feature_group': group,
                'delta_f1': value
            })
    
    delta_df = pd.DataFrame(delta_data)
    feature_groups = delta_df['feature_group'].unique()
    x = np.arange(len(feature_groups))
    width = 0.25
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    for i, dataset in enumerate(datasets):
        dataset_data = delta_df[delta_df['dataset'] == dataset]
        values = [dataset_data[dataset_data['feature_group'] == fg]['delta_f1'].values[0]
                  for fg in feature_groups]
        ax.bar(x + i * width, values, width, label=dataset.capitalize(), 
               alpha=0.8, color=colors[i])
    
    ax.set_xlabel('Feature Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('ΔF1 (Contribution to Performance)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Group Contribution Comparison Across Datasets', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(feature_groups, fontsize=11)
    ax.legend(title='Dataset', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'delta_f1_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: delta_f1_comparison.png")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot_data = all_results.pivot(index='experiment', columns='dataset', values='f1')
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, 
                cbar_kws={'label': 'F1 Score'}, vmin=0, vmax=1.0)
    ax.set_title('F1 Score Heatmap: Experiments × Datasets', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Experiment', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'f1_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: f1_heatmap.png")
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, dataset in enumerate(datasets):
        dataset_delta = delta_df[delta_df['dataset'] == dataset].sort_values('delta_f1', ascending=False)
        
        axes[idx].barh(dataset_delta['feature_group'], dataset_delta['delta_f1'], 
                       color=colors[idx], alpha=0.7)
        axes[idx].set_xlabel('ΔF1', fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{dataset.capitalize()} Dataset', fontsize=12, fontweight='bold')
        axes[idx].grid(axis='x', alpha=0.3)
        axes[idx].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.suptitle('Feature Group Rankings by Dataset', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_group_rankings.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: feature_group_rankings.png")
    plt.close()


def main():
    """Main execution pipeline."""
    
    print("\n" + "="*80)
    print("FEATURE GROUP IMPORTANCE COMPARISON")
    print("="*80 + "\n")
    
    RANDOM_STATE = 42
    set_seed(RANDOM_STATE)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path("runs") / f"feature_group_comparison_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {save_dir}\n")
    
    print("Loading datasets...")
    datasets = {
        'reddit': load_dataset("data/reddit_dataset.csv", 
                              text_column="Comment", label_column="Hateful"),
        '4chan': load_dataset("data/4chan_dataset.csv", 
                             text_column="Comment", label_column="Hateful"),
        'twitter': load_dataset("data/twitter_dataset.csv", 
                               text_column="Comment", label_column="Hateful"),
    }
    
    for name, df in datasets.items():
        print(f"  {name.capitalize()}: {len(df)} samples, {df['label'].sum()} hateful")
    
    all_results = []
    all_models = {}
    
    for dataset_name, df in datasets.items():
        results_df, models, groups = run_ablation_experiments(
            dataset_name, df, RANDOM_STATE
        )
        all_results.append(results_df)
        all_models[dataset_name] = models
        
        results_df.to_csv(save_dir / f'{dataset_name}_results.csv', index=False)
        print(f"\n✅ Saved results for {dataset_name}")
    
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(save_dir / 'all_datasets_results.csv', index=False)
    print(f"\n✅ Saved combined results")
    
    plot_comparison_across_datasets(combined_results, save_dir)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {save_dir}")
    print(f"\nKey outputs:")
    print(f"  - <dataset>_results.csv: Per-dataset ablation tables")
    print(f"  - all_datasets_results.csv: Combined metrics table")
    print(f"  - f1_comparison_all_experiments.png / delta_f1_comparison.png")
    print(f"  - f1_heatmap.png / feature_group_rankings.png")
    print()


if __name__ == "__main__":
    main()

