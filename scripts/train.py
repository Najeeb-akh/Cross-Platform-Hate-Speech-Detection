#!/usr/bin/env python3
"""
Main training script for hate speech detection using comprehensive text features.

This script:
1. Loads Reddit, 4chan, and Twitter datasets
2. Extracts comprehensive features (lexical, semantic, TF-IDF, embeddings, experimental)
3. Trains logistic regression classifier
4. Evaluates on test and cross-dataset test sets
5. Saves metrics, plots, and model artifacts

Usage:
    python scripts/train.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.features import TextFeatureExtractor, extract_features, load_dataset
from src.utils import set_seed, create_run_dir, save_json


def plot_confusion_matrix(y_true, y_pred, save_path, title):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Hateful', 'Hateful'],
                yticklabels=['Non-Hateful', 'Hateful'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def compute_and_print_metrics(y_true, y_pred, y_proba, dataset_name):
    """Compute and print evaluation metrics."""
    print(f"\n{'='*80}")
    print(f"Results: {dataset_name}")
    print('='*80)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Non-Hateful', 'Hateful']))
    
    # Additional metrics
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba)
        print(f"ROC-AUC Score: {auc:.4f}")
    except:
        auc = None
        print("ROC-AUC Score: N/A")
    
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix stats
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    
    metrics = {
        'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
        'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'f1_score': float(f1),
        'roc_auc': float(auc) if auc is not None else None,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
    }
    
    return metrics


def train_separate_classifiers(
    dataset_splits,
    run_dir,
    random_state=42,
):
    """
    Train a separate classifier on each dataset (using 100% of training data) 
    and evaluate it on every other dataset's test splits (using all 5 folds).
    
    For each training dataset:
      - Train on 100% of that dataset (1000 samples, using full feature matrix)
      - Test on all 5 folds of each other dataset
      - Aggregate metrics across folds (mean ± std)
    
    Args:
        dataset_splits: Mapping of dataset name -> dict containing 5-fold CV splits
            with 'folds' list and 'n_folds' count.
        run_dir: Path to the current run directory for saving artifacts.
        random_state: Random seed for reproducibility.
    
    Returns:
        Nested dictionary of evaluation metrics indexed by
        [training_dataset][testing_dataset] with aggregated metrics (mean ± std).
    """
    print("\nPreparing per-dataset classifiers and cross-dataset evaluations...\n")
    
    if not dataset_splits:
        print("No dataset splits supplied; skipping per-dataset training.")
        return {}
    
    per_dataset_dir = run_dir / "per_dataset_models"
    per_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    models = {}
    
    # Train models on 100% of each dataset
    for train_name, split_info in dataset_splits.items():
        print(f"{'-'*40}\nTraining model on 100% of {train_name.upper()} dataset\n{'-'*40}")
        
        # Use full dataset features directly
        full_features = split_info.get('full_features')
        if full_features is None:
            # Fallback: use first fold's train_features (which now contains full dataset)
            folds = split_info.get('folds', [])
            if not folds:
                print(f"  Skipping {train_name}: no folds available.")
                continue
            full_features = folds[0]['train_features'].copy()
        
        feature_cols = [col for col in full_features.columns if col != 'label']
        X_train = full_features[feature_cols].values
        y_train = full_features['label'].values
        
        print(f"  Training on {len(X_train)} samples ({len(feature_cols)} features)...")
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight='balanced',
            solver='saga',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        print("  Model training complete.")
        
        models[train_name] = {
            'model': model,
            'feature_cols': feature_cols,
            'full_columns': list(full_features.columns),
        }
    
    if not models:
        print("No eligible datasets found for per-dataset training.")
        return {}
    
    results = {}
    
    # Evaluate each model on all folds of other datasets
    for train_name, model_info in models.items():
        model = model_info['model']
        feature_cols = model_info['feature_cols']
        results[train_name] = {}
        
        for test_name, split_info in dataset_splits.items():
            # Skip testing on the same dataset - we already have CV results for that
            if test_name == train_name:
                continue
            
            print(f"\n{'='*60}")
            print(f"Evaluating {train_name.upper()} model on {test_name.upper()} (all 5 folds)")
            print(f"{'='*60}")
            
            folds = split_info.get('folds', [])
            if not folds:
                print(f"  Skipping {test_name}: no folds available.")
                continue
            
            fold_metrics = []
            # Collect predictions from all folds for combined confusion matrix
            all_y_test = []
            all_y_pred = []
            
            # Test on each fold of the test dataset
            for fold_data in folds:
                fold_idx = fold_data['fold']
                print(f"\n  Fold {fold_idx}/{split_info['n_folds']}...")
                
                test_features = fold_data['test_features'].copy()
                # Align columns with training data
                test_features = test_features.reindex(
                    columns=model_info['full_columns'],
                    fill_value=0
                )
                
                X_test = test_features[feature_cols].fillna(0).values
                y_test = test_features['label'].values
                
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                # Collect for combined confusion matrix
                all_y_test.append(y_test)
                all_y_pred.append(y_pred)
                
                # Compute metrics for this fold
                cm = confusion_matrix(y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                metrics = {
                    'fold': fold_idx,
                    'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
                    'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                    'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                    'f1_score': float(f1_score(y_test, y_pred)),
                    'roc_auc': float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else 0.0,
                    'true_positives': int(tp),
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                }
                
                fold_metrics.append(metrics)
                print(f"    F1: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            
            # Aggregate metrics across folds
            aggregated = {
                'training_dataset': train_name,
                'testing_dataset': test_name,
                'n_folds': len(fold_metrics),
                'f1_mean': float(np.mean([m['f1_score'] for m in fold_metrics])),
                'f1_std': float(np.std([m['f1_score'] for m in fold_metrics])),
                'accuracy_mean': float(np.mean([m['accuracy'] for m in fold_metrics])),
                'accuracy_std': float(np.std([m['accuracy'] for m in fold_metrics])),
                'precision_mean': float(np.mean([m['precision'] for m in fold_metrics])),
                'precision_std': float(np.std([m['precision'] for m in fold_metrics])),
                'recall_mean': float(np.mean([m['recall'] for m in fold_metrics])),
                'recall_std': float(np.std([m['recall'] for m in fold_metrics])),
                'roc_auc_mean': float(np.mean([m['roc_auc'] for m in fold_metrics])),
                'roc_auc_std': float(np.std([m['roc_auc'] for m in fold_metrics])),
                'fold_metrics': fold_metrics,
            }
            
            results[train_name][test_name] = aggregated
            
            # Print aggregated results
            print(f"\n  {train_name.upper()} → {test_name.upper()} Results ({len(fold_metrics)}-Fold):")
            print(f"    F1 Score:    {aggregated['f1_mean']:.4f} ± {aggregated['f1_std']:.4f}")
            print(f"    Accuracy:    {aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f}")
            print(f"    Precision:   {aggregated['precision_mean']:.4f} ± {aggregated['precision_std']:.4f}")
            print(f"    Recall:      {aggregated['recall_mean']:.4f} ± {aggregated['recall_std']:.4f}")
            print(f"    ROC-AUC:     {aggregated['roc_auc_mean']:.4f} ± {aggregated['roc_auc_std']:.4f}")
            
            # Save aggregated metrics
            metrics_path = per_dataset_dir / f"{train_name}_model_eval_on_{test_name}.json"
            save_json(aggregated, metrics_path)
            
            # Save confusion matrix on all test folds combined
            if all_y_test and all_y_pred:
                # Combine all predictions from all folds
                y_test_all = np.concatenate(all_y_test)
                y_pred_all = np.concatenate(all_y_pred)
                
                cm_path = per_dataset_dir / f"cm_{train_name}_model_on_{test_name}_test.png"
                plot_confusion_matrix(
                    y_test_all,
                    y_pred_all,
                    cm_path,
                    f"Confusion Matrix - {train_name.upper()} Model on {test_name.upper()} Test Set (All Folds)",
                )
    
    summary_path = per_dataset_dir / "cross_dataset_results.json"
    save_json(results, summary_path)
    print(f"\nSaved cross-dataset evaluation summary to {summary_path}")
    
    return results


def train_leave_one_out_classifiers(
    dataset_splits,
    run_dir,
    random_state=42,
):
    """
    Train classifiers on 100% of all datasets except one and evaluate
    on the held-out dataset using all 5 folds.
    
    For each held-out dataset D:
      - Train once on 100% of all other datasets (2000 samples = 2 datasets × 1000, using full feature matrices)
      - Test on all 5 folds of dataset D
      - Aggregate metrics across folds (mean ± std)
      - Create confusion matrix on all test folds combined
    """
    if len(dataset_splits) < 2:
        print("At least two datasets are required for leave-one-out evaluation.")
        return {}
    
    loo_dir = run_dir / "leave_one_out_models"
    loo_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for held_out, split_info in dataset_splits.items():
        print(f"\n{'='*60}\nLeave-One-Out: Held-out dataset {held_out.upper()}\n{'='*60}")
        
        held_out_folds = split_info.get("folds", [])
        if not held_out_folds:
            print(f"  No folds found for {held_out}; skipping.")
            continue
        
        # 1. TRAIN ONCE: Combine full datasets from all other datasets
        print(f"\n  Training on 100% of all other datasets...")
        train_frames = []
        for name, info in dataset_splits.items():
            if name == held_out:
                continue
            # Use full dataset features directly
            full_features = info.get('full_features')
            if full_features is None:
                # Fallback: use first fold's train_features
                folds = info.get("folds", [])
                if not folds:
                    continue
                full_features = folds[0]["train_features"].copy()
            train_frames.append(full_features.copy())
        
        if not train_frames:
            print(f"  No training data available when leaving out {held_out}; skipping.")
            continue
        
        combined_train = pd.concat(train_frames, axis=0, ignore_index=True).fillna(0)
        feature_cols = [col for col in combined_train.columns if col != "label"]
        
        X_train = combined_train[feature_cols].values
        y_train = combined_train["label"].values
        
        print(f"  Training on {len(X_train)} samples ({len(feature_cols)} features)...")
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        print("  Model training complete.")
        
        # 2. TEST ON ALL FOLDS: Evaluate on all 5 folds of held-out dataset
        print(f"\n  Testing on all 5 folds of {held_out.upper()}...")
        all_y_test = []
        all_y_pred = []
        fold_metrics = []
        
        for fold_index, fold_data in enumerate(held_out_folds):
            fold_id = fold_index + 1
            print(f"\n    Fold {fold_id}/{len(held_out_folds)}...")
            
            # Test on this fold
            test_features = fold_data["test_features"].copy()
            test_features = test_features.reindex(columns=combined_train.columns, fill_value=0)
            
            X_test = test_features[feature_cols].fillna(0).values
            y_test = test_features["label"].values
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Collect for combined confusion matrix
            all_y_test.append(y_test)
            all_y_pred.append(y_pred)
            
            # Compute metrics for this fold
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            metrics = {
                'fold': fold_id,
                'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
                'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'f1_score': float(f1_score(y_test, y_pred)),
                'roc_auc': float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else 0.0,
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
            }
            
            fold_metrics.append(metrics)
            print(f"      F1: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        if not fold_metrics:
            print(f"  No fold metrics computed for {held_out}; skipping aggregation.")
            continue
        
        # 3. AGGREGATE METRICS across folds
        aggregated = {
            "held_out_dataset": held_out,
            "n_folds": len(fold_metrics),
            "f1_mean": float(np.mean([m["f1_score"] for m in fold_metrics])),
            "f1_std": float(np.std([m["f1_score"] for m in fold_metrics])),
            "accuracy_mean": float(np.mean([m["accuracy"] for m in fold_metrics])),
            "accuracy_std": float(np.std([m["accuracy"] for m in fold_metrics])),
            "precision_mean": float(np.mean([m["precision"] for m in fold_metrics])),
            "precision_std": float(np.std([m["precision"] for m in fold_metrics])),
            "recall_mean": float(np.mean([m["recall"] for m in fold_metrics])),
            "recall_std": float(np.std([m["recall"] for m in fold_metrics])),
            "roc_auc_mean": float(np.mean([m["roc_auc"] for m in fold_metrics])),
            "roc_auc_std": float(np.std([m["roc_auc"] for m in fold_metrics])),
            "fold_metrics": fold_metrics,
        }
        
        results[held_out] = aggregated
        
        # Print aggregated results
        print(f"\n  All-Except-{held_out.upper()} → {held_out.upper()} Results ({len(fold_metrics)}-Fold):")
        print(f"    F1 Score:    {aggregated['f1_mean']:.4f} ± {aggregated['f1_std']:.4f}")
        print(f"    Accuracy:    {aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f}")
        print(f"    Precision:   {aggregated['precision_mean']:.4f} ± {aggregated['precision_std']:.4f}")
        print(f"    Recall:      {aggregated['recall_mean']:.4f} ± {aggregated['recall_std']:.4f}")
        print(f"    ROC-AUC:     {aggregated['roc_auc_mean']:.4f} ± {aggregated['roc_auc_std']:.4f}")
        
        # Save aggregated metrics
        metrics_path = loo_dir / f"metrics_all_except_{held_out}_on_{held_out}.json"
        save_json(aggregated, metrics_path)
        
        # 4. CONFUSION MATRIX ON ALL FOLDS COMBINED
        if all_y_test and all_y_pred:
            y_test_all = np.concatenate(all_y_test)
            y_pred_all = np.concatenate(all_y_pred)
            
            cm_path = loo_dir / f"cm_all_except_{held_out}_on_{held_out}_test.png"
            plot_confusion_matrix(
                y_test_all,
                y_pred_all,
                cm_path,
                f"Confusion Matrix - All Except {held_out.upper()} on {held_out.upper()} Test Set (All Folds)",
            )
    
    results_path = loo_dir / "leave_one_out_summary.json"
    save_json(results, results_path)
    print(f"\nSaved leave-one-out summary to {results_path}")
    
    return results


def train_global_classifier(
    dataset_splits,
    run_dir,
    random_state=42,
):
    """
    Train a global classifier on the union of all training sets using 5-fold CV
    and evaluate on each dataset's test split separately.

    For each fold k:
      - Combine training splits of all datasets for fold k (2400 samples = 3 datasets × 800 per fold).
      - Train a single global model on this combined training data.
      - Evaluate on each dataset's test split for fold k (200 samples per dataset).
    Metrics are then aggregated across folds per dataset.
    """
    if not dataset_splits:
        print("No dataset splits supplied; skipping global classifier training.")
        return {}
    
    global_dir = run_dir / "global_model"
    global_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    last_model = None  # keep the last trained model as a representative global model
    
    # Determine number of folds from any dataset entry
    any_entry = next(iter(dataset_splits.values()))
    n_folds = any_entry.get("n_folds", 0)
    
    for fold_index in range(n_folds):
        fold_id = fold_index + 1
        print(f"\n{'='*40}\nGlobal Model Training - Fold {fold_id}/{n_folds}\n{'='*40}")
        
        train_frames = []
        for name, split_info in dataset_splits.items():
            folds = split_info.get("folds", [])
            if len(folds) <= fold_index:
                print(
                    f"  Skipping dataset {name} for fold {fold_id}: "
                    f"missing fold data."
                )
                continue
            train_frames.append(folds[fold_index]["train_features"].copy())
        
        if not train_frames:
            print(f"  No training data available for global model on fold {fold_id}.")
            continue
        
        combined_train = pd.concat(train_frames, axis=0, ignore_index=True).fillna(0)
        feature_cols = [col for col in combined_train.columns if col != "label"]
        
        X_train = combined_train[feature_cols].values
        y_train = combined_train["label"].values
        
        print(f"  Total training samples (combined): {len(X_train)}")
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        last_model = model
        print("  Global model training complete for this fold.")
        
        # Evaluate on each dataset's test split for this fold
        for test_name, split_info in dataset_splits.items():
            folds = split_info.get("folds", [])
            if len(folds) <= fold_index:
                print(
                    f"  Skipping evaluation on {test_name} for fold {fold_id}: "
                    f"missing fold data."
                )
                continue
            
            test_features = folds[fold_index]["test_features"].copy()
            test_features = test_features.reindex(columns=combined_train.columns, fill_value=0)
            
            X_test = test_features[feature_cols].fillna(0).values
            y_test = test_features["label"].values
            
            dataset_label = (
                f"Global Model (Fold {fold_id}) on {test_name.upper()} Test Set"
            )
            print(f"\n  Evaluating {dataset_label}...")
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = compute_and_print_metrics(y_test, y_pred, y_proba, dataset_label)
            metrics["fold"] = fold_id
            
            results.setdefault(test_name, {"fold_metrics": []})
            results[test_name]["fold_metrics"].append(metrics)
            
            metrics_path = (
                global_dir
                / f"metrics_global_fold{fold_id}_on_{test_name}.json"
            )
            save_json(metrics, metrics_path)
            
            cm_path = (
                global_dir
                / f"cm_global_fold{fold_id}_on_{test_name}.png"
            )
            plot_confusion_matrix(
                y_test,
                y_pred,
                cm_path,
                f"Confusion Matrix - Global Model (Fold {fold_id}) "
                f"on {test_name.upper()} Test Set",
            )
    
    # Aggregate metrics across folds for each dataset
    for test_name, info in results.items():
        fold_metrics = info.get("fold_metrics", [])
        if not fold_metrics:
            continue
        
        f1_scores = [m["f1_score"] for m in fold_metrics]
        accuracies = [m["accuracy"] for m in fold_metrics]
        precisions = [m["precision"] for m in fold_metrics]
        recalls = [m["recall"] for m in fold_metrics]
        roc_aucs = [
            m["roc_auc"] for m in fold_metrics
            if m.get("roc_auc") is not None
        ]
        
        aggregated = {
            "dataset": test_name,
            "n_folds": len(fold_metrics),
            "f1_mean": float(np.mean(f1_scores)),
            "f1_std": float(np.std(f1_scores)),
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "precision_mean": float(np.mean(precisions)),
            "precision_std": float(np.std(precisions)),
            "recall_mean": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)),
            "roc_auc_mean": float(np.mean(roc_aucs)) if roc_aucs else None,
            "roc_auc_std": float(np.std(roc_aucs)) if roc_aucs else None,
            "fold_metrics": fold_metrics,
        }
        
        results[test_name] = aggregated
        
        metrics_path = global_dir / f"metrics_global_on_{test_name}.json"
        save_json(aggregated, metrics_path)
    
    results_path = global_dir / "global_model_summary.json"
    save_json(results, results_path)
    print(f"\nSaved global model CV evaluation summary to {results_path}")
    
    return results, last_model


def train_with_cross_validation(dataset_splits, run_dir, random_state=42):
    """
    Train models using 5-fold CV and report mean ± std metrics.
    
    Args:
        dataset_splits: Dict with fold information for each dataset
        run_dir: Path to save results
        random_state: Random seed
        
    Returns:
        Dict with aggregated results per dataset
    """
    print("\nTraining with 5-fold cross-validation...")
    
    cv_dir = run_dir / "cross_validation"
    cv_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for name, split_info in dataset_splits.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {name.upper()} - 5-Fold Cross-Validation")
        print(f"{'='*60}")
        
        fold_metrics = []
        
        for fold_data in split_info['folds']:
            fold_idx = fold_data['fold']
            print(f"\n  Training Fold {fold_idx}/{split_info['n_folds']}...")
            
            # Get data from fold (800 train, 200 test)
            train_features = fold_data['train_features']
            test_features = fold_data['test_features']
            feature_cols = fold_data['feature_cols']
            
            X_train = train_features[feature_cols].values
            y_train = train_features['label'].values
            X_test = test_features[feature_cols].values
            y_test = test_features['label'].values
            
            # Train model
            model = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight='balanced',
                solver='saga',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Compute metrics
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            metrics = {
                'fold': fold_idx,
                'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
                'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'f1_score': float(f1_score(y_test, y_pred)),
                'roc_auc': float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else 0.0,
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
            }
            
            fold_metrics.append(metrics)
            print(f"    F1: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        # Aggregate across folds
        aggregated = {
            'dataset': name,
            'n_folds': split_info['n_folds'],
            'f1_mean': np.mean([m['f1_score'] for m in fold_metrics]),
            'f1_std': np.std([m['f1_score'] for m in fold_metrics]),
            'accuracy_mean': np.mean([m['accuracy'] for m in fold_metrics]),
            'accuracy_std': np.std([m['accuracy'] for m in fold_metrics]),
            'precision_mean': np.mean([m['precision'] for m in fold_metrics]),
            'precision_std': np.std([m['precision'] for m in fold_metrics]),
            'recall_mean': np.mean([m['recall'] for m in fold_metrics]),
            'recall_std': np.std([m['recall'] for m in fold_metrics]),
            'roc_auc_mean': np.mean([m['roc_auc'] for m in fold_metrics]),
            'roc_auc_std': np.std([m['roc_auc'] for m in fold_metrics]),
            'fold_metrics': fold_metrics,
        }
        
        all_results[name] = aggregated
        
        # Print aggregated results
        print(f"\n  {name.upper()} Results ({split_info['n_folds']}-Fold CV):")
        print(f"    F1 Score:    {aggregated['f1_mean']:.4f} ± {aggregated['f1_std']:.4f}")
        print(f"    Accuracy:    {aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f}")
        print(f"    Precision:   {aggregated['precision_mean']:.4f} ± {aggregated['precision_std']:.4f}")
        print(f"    Recall:      {aggregated['recall_mean']:.4f} ± {aggregated['recall_std']:.4f}")
        print(f"    ROC-AUC:     {aggregated['roc_auc_mean']:.4f} ± {aggregated['roc_auc_std']:.4f}")
        
        # Save results
        save_json(aggregated, cv_dir / f"{name}_cv_results.json")
    
    # Save summary
    summary_df = pd.DataFrame([
        {
            'Dataset': name.capitalize(),
            'F1 (mean ± std)': f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}",
            'Accuracy (mean ± std)': f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}",
            'Precision (mean ± std)': f"{r['precision_mean']:.4f} ± {r['precision_std']:.4f}",
            'Recall (mean ± std)': f"{r['recall_mean']:.4f} ± {r['recall_std']:.4f}",
        }
        for name, r in all_results.items()
    ])
    
    summary_path = cv_dir / "cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved CV summary to {summary_path}")
    
    return all_results


def main():
    """Main training and evaluation pipeline."""
    
    print("\n" + "="*80)
    print("HATE SPEECH DETECTION - COMPREHENSIVE FEATURE TRAINING")
    print("="*80 + "\n")
    
    # Configuration
    RANDOM_STATE = 42
    N_FOLDS = 5
    
    set_seed(RANDOM_STATE)
    
    # Create run directory
    run_dir = create_run_dir("runs")
    print(f"Run directory: {run_dir}\n")
    
    # -------------------------------------------------------------------------
    # 1. Load all datasets
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("STEP 1: Loading Datasets")
    print("-"*80 + "\n")
    
    print("Loading Reddit dataset...")
    reddit_df = load_dataset("data/reddit_dataset.csv", text_column="Comment", label_column="Hateful")
    print(f"  Reddit: {len(reddit_df)} samples, {reddit_df['label'].sum()} hateful")
    
    print("Loading 4chan dataset...")
    fourchan_df = load_dataset("data/4chan_dataset.csv", text_column="Comment", label_column="Hateful")
    print(f"  4chan: {len(fourchan_df)} samples, {fourchan_df['label'].sum()} hateful")
    
    print("Loading Twitter dataset...")
    twitter_df = load_dataset("data/twitter_dataset.csv", text_column="Comment", label_column="Hateful")
    print(f"  Twitter: {len(twitter_df)} samples, {twitter_df['label'].sum()} hateful")

    # -------------------------------------------------------------------------
    # 2. Prepare dataset-specific 5-fold cross-validation splits
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print(f"STEP 2: Preparing {N_FOLDS}-Fold Cross-Validation Splits")
    print("-"*80 + "\n")

    dataset_splits = {}
    dataset_sources = {
        'reddit': reddit_df,
        '4chan': fourchan_df,
        'twitter': twitter_df,
    }
    
    for name, df in dataset_sources.items():
        print(f"{'='*40}\nDataset: {name.upper()}\n{'='*40}")
        
        unique_labels = df['label'].nunique()
        if unique_labels < 2:
            print(f"  Skipping {name}: requires at least two label classes (found {unique_labels}).")
            continue
        
        # Extract features from FULL dataset once
        print(f"  Extracting features from full {name} dataset ({len(df)} samples)...")
        full_extractor = TextFeatureExtractor()
        full_features = full_extractor.fit_transform(df, source_name=name).fillna(0)
        feature_cols = [col for col in full_features.columns if col != 'label']
        print(f"  Full dataset features shape: {full_features.shape}")
        
        # 5-fold stratified cross-validation (for test splits only)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, df['label'])):
            print(f"\n  --- Fold {fold_idx + 1}/{N_FOLDS} ---")
            
            # Split full features by indices for train and test
            ds_train_features = full_features.iloc[train_idx].copy().reset_index(drop=True)
            ds_test_features = full_features.iloc[test_idx].copy().reset_index(drop=True)
            
            print(f"    Train: {len(ds_train_features)} samples | Test: {len(ds_test_features)} samples")
            print(f"    Train features shape: {ds_train_features.shape}")
            print(f"    Test features shape: {ds_test_features.shape}")
            
            fold_results.append({
                'fold': fold_idx + 1,
                'train_features': ds_train_features,  # Actual training split (800 samples)
                'test_features': ds_test_features,  # Test split (200 samples)
                'feature_cols': feature_cols,
                'extractor': full_extractor,
            })
        
        dataset_splits[name] = {
            'folds': fold_results,
            'n_folds': N_FOLDS,
            'full_features': full_features,  # Store full features separately
        }

    if not dataset_splits:
        print("No dataset splits were created; exiting early.")
        return
    
    # -------------------------------------------------------------------------
    # 3. 5-Fold Cross-Validation Training
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("STEP 3: 5-Fold Cross-Validation Training")
    print("-"*80 + "\n")

    cv_results = train_with_cross_validation(
        dataset_splits=dataset_splits,
        run_dir=run_dir,
        random_state=RANDOM_STATE,
    )

    # -------------------------------------------------------------------------
    # 4. Cross-dataset and global experiments
    # -------------------------------------------------------------------------
    
    # 4.1 Train separate classifiers per dataset (100% data) and evaluate cross-dataset (all 5 folds)
    per_dataset_results = train_separate_classifiers(
        dataset_splits=dataset_splits,
        run_dir=run_dir,
        random_state=RANDOM_STATE,
    )
    
    # 4.2 Leave-one-out training across datasets using full 5-fold splits
    loo_results = train_leave_one_out_classifiers(
        dataset_splits=dataset_splits,
        run_dir=run_dir,
        random_state=RANDOM_STATE,
    )
    
    # 4.3 Global classifier trained on all datasets combined using 5-fold CV
    global_results, _ = train_global_classifier(
        dataset_splits=dataset_splits,
        run_dir=run_dir,
        random_state=RANDOM_STATE,
    )

    # -------------------------------------------------------------------------
    # 5. Print high-level summary and save overall summary JSON
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {run_dir}")

    print("\nSummary of 5-Fold Cross-Validation Results:")
    print("-" * 80)
    for name, results in cv_results.items():
        print(
            f"{name.upper():10s}: F1 = {results['f1_mean']:.4f} "
            f"± {results['f1_std']:.4f}"
        )

    print("\nCross-dataset and global model summaries written to:")
    print(f"  - {run_dir / 'per_dataset_models'}")
    print(f"  - {run_dir / 'leave_one_out_models'}")
    print(f"  - {run_dir / 'global_model'}")

    overall_summary = {
        "cross_validation": cv_results,
        "per_dataset": per_dataset_results,
        "leave_one_out": loo_results,
        "global": global_results,
    }
    save_json(overall_summary, run_dir / "overall_summary.json")
    print(f"\nSaved overall summary to {run_dir / 'overall_summary.json'}")
    

if __name__ == "__main__":
    main()
