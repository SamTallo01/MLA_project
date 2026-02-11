import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureLoader:
    """Loads features from .h5 files flexibly"""
    
    def __init__(self, features_dir: str, feature_key: str = 'features'):
        """
        Args:
            features_dir: Directory containing .h5 files
            feature_key: Key inside h5 file (e.g.: 'features', 'feats', etc.)
        """
        self.features_dir = Path(features_dir)
        self.feature_key = feature_key
        self.feature_dim = None
        
    def load_single_slide(self, slide_id: str) -> np.ndarray:
        """Load features for a single slide"""
        # Try different common extensions
        possible_paths = [
            self.features_dir / f"{slide_id}.h5",
            self.features_dir / f"{slide_id}.hdf5",
            self.features_dir / f"{slide_id}_features.h5",
        ]
        
        for h5_path in possible_paths:
            if h5_path.exists():
                with h5py.File(h5_path, 'r') as f:
                    # Try different common keys
                    for key in [self.feature_key, 'features', 'feats', 'data']:
                        if key in f:
                            features = f[key][:]
                            
                            # Auto-detect feature dimension
                            if self.feature_dim is None:
                                self.feature_dim = features.shape[-1]
                                print(f"Detected feature dimension: {self.feature_dim}")
                            
                            return features
                    
                    # If key not found, print available ones
                    print(f"Available keys in {h5_path.name}: {list(f.keys())}")
                    raise KeyError(f"Key '{self.feature_key}' not found")
        
        raise FileNotFoundError(f"No .h5 file found for slide {slide_id}")
    
    def load_all_slides(self, slide_ids: List[str]) -> Dict[str, np.ndarray]:
        """Load features for all slides"""
        features_dict = {}
        
        print(f"Loading features from: {self.features_dir}")
        print(f"Looking for key: '{self.feature_key}'")
        print("-" * 60)
        
        for i, slide_id in enumerate(slide_ids):
            try:
                feats = self.load_single_slide(slide_id)
                features_dict[slide_id] = feats
                
                if i == 0:
                    print(f"First slide '{slide_id}':")
                    print(f"  Shape: {feats.shape}")
                    print(f"  N patches: {feats.shape[0]}")
                    print(f"  Feature dim: {feats.shape[1]}")
                    print(f"  Value range: [{feats.min():.3f}, {feats.max():.3f}]")
                    print("-" * 60)
                
            except Exception as e:
                print(f"ERROR loading {slide_id}: {e}")
                raise
        
        print(f"Successfully loaded {len(features_dict)} slides")
        return features_dict


class BinaryConverter:
    """Converts labels from 3 classes to 2 classes"""
    
    STRATEGIES = {
        'class_0_vs_rest': 'Class 0 vs Rest (1+2)',
        'class_1_vs_rest': 'Class 1 vs Rest (0+2)',
        'class_2_vs_rest': 'Class 2 vs Rest (0+1)',
        'class_0_vs_1': 'Class 0 vs Class 1 (exclude 2)',
        'class_0_vs_2': 'Class 0 vs Class 2 (exclude 1)',
        'class_1_vs_2': 'Class 1 vs Class 2 (exclude 0)',
        'epithelioid_vs_non': 'Epithelioid (0) vs Non-Epithelioid (1+2)',
        'sarcomatoid_vs_rest': 'Sarcomatoid (2) vs Others (0+1)',
    }
    
    @staticmethod
    def convert(df: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, str]:
        """
        Converts dataframe to binary problem
        
        Args:
            df: DataFrame with 'label' column (0, 1, 2)
            strategy: Conversion strategy
            
        Returns:
            (df_binary, description): Modified DataFrame and description
        """
        df_copy = df.copy()
        
        if strategy == 'class_0_vs_rest':
            df_copy['label_binary'] = (df_copy['label'] != 0).astype(int)
            desc = "0 vs (1+2)"
            
        elif strategy == 'class_1_vs_rest':
            df_copy['label_binary'] = (df_copy['label'] != 1).astype(int)
            desc = "1 vs (0+2)"
            
        elif strategy == 'class_2_vs_rest':
            df_copy['label_binary'] = (df_copy['label'] == 2).astype(int)
            desc = "2 vs (0+1)"
            
        elif strategy == 'class_0_vs_1':
            df_copy = df_copy[df_copy['label'] != 2].copy()
            df_copy['label_binary'] = df_copy['label']
            desc = "0 vs 1 (no class 2)"
            
        elif strategy == 'class_0_vs_2':
            df_copy = df_copy[df_copy['label'] != 1].copy()
            df_copy['label_binary'] = (df_copy['label'] == 2).astype(int)
            desc = "0 vs 2 (no class 1)"
            
        elif strategy == 'class_1_vs_2':
            df_copy = df_copy[df_copy['label'] != 0].copy()
            df_copy['label_binary'] = (df_copy['label'] == 2).astype(int)
            desc = "1 vs 2 (no class 0)"
            
        elif strategy == 'epithelioid_vs_non':
            df_copy['label_binary'] = (df_copy['label'] != 0).astype(int)
            desc = "Epithelioid (0) vs Non-Epithelioid (1+2)"
            
        elif strategy == 'sarcomatoid_vs_rest':
            df_copy['label_binary'] = (df_copy['label'] == 2).astype(int)
            desc = "Sarcomatoid (2) vs Others (0+1)"
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Update label column
        df_copy['label'] = df_copy['label_binary']
        df_copy = df_copy.drop('label_binary', axis=1)
        
        return df_copy, desc


class FeatureAggregator:
    """Aggregates patch features into slide-level representations"""
    
    METHODS = ['mean', 'max', 'std', 'mean+max', 'mean+std', 'mean+max+std', 
               'percentile', 'top_k', 'attention_pool']
    
    @staticmethod
    def aggregate(features: np.ndarray, method: str = 'mean', k: int = 10) -> np.ndarray:
        """
        Aggregates patch features into a single vector
        
        Args:
            features: Array [N_patches, feature_dim]
            method: Aggregation method
            k: Number of top patches for 'top_k'
        
        Returns:
            Array [aggregated_feature_dim]
        """
        if method == 'mean':
            return np.mean(features, axis=0)
        
        elif method == 'max':
            return np.max(features, axis=0)
        
        elif method == 'std':
            return np.std(features, axis=0)
        
        elif method == 'mean+max':
            return np.concatenate([
                np.mean(features, axis=0),
                np.max(features, axis=0)
            ])
        
        elif method == 'mean+std':
            return np.concatenate([
                np.mean(features, axis=0),
                np.std(features, axis=0)
            ])
        
        elif method == 'mean+max+std':
            return np.concatenate([
                np.mean(features, axis=0),
                np.max(features, axis=0),
                np.std(features, axis=0)
            ])
        
        elif method == 'percentile':
            p25 = np.percentile(features, 25, axis=0)
            p50 = np.percentile(features, 50, axis=0)
            p75 = np.percentile(features, 75, axis=0)
            return np.concatenate([p25, p50, p75])
        
        elif method == 'top_k':
            # Average of top-k patches (based on L2 norm)
            norms = np.linalg.norm(features, axis=1)
            top_indices = np.argsort(norms)[-k:]
            return np.mean(features[top_indices], axis=0)
        
        elif method == 'attention_pool':
            # Simple attention: weighted mean based on norm
            norms = np.linalg.norm(features, axis=1)
            weights = np.exp(norms) / np.sum(np.exp(norms))
            return np.sum(features * weights[:, np.newaxis], axis=0)
        
        else:
            raise ValueError(f"Unknown method: {method}. Available: {FeatureAggregator.METHODS}")


class BaselineClassifierBenchmark:
    """Benchmark of simple classifiers for debugging"""
    
    def __init__(self, normalize: bool = True, use_pca: Optional[float] = None, 
                 n_classes: int = 3, output_dir: str = 'baseline_results',
                 features_dir_name: str = None):
        """
        Args:
            normalize: If True, applies StandardScaler
            use_pca: If specified, reduces dimensionality maintaining this % of variance
            n_classes: Number of classes (2 or 3)
            output_dir: Base directory where to save results
            features_dir_name: Name of features directory (will create subfolder)
        """
        self.normalize = normalize
        self.use_pca = use_pca
        self.n_classes = n_classes
        self.scaler = None
        self.pca = None
        
        # Create output directory with features subfolder
        base_output = Path(output_dir)
        if features_dir_name:
            self.output_dir = base_output / features_dir_name
        else:
            self.output_dir = base_output
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir.absolute()}")
        
    def get_classifiers(self):
        """Returns dictionary of classifiers to test"""
        return {
            'Logistic Regression': LogisticRegression(
                max_iter=2000,
                class_weight='balanced',
                random_state=1,
                solver='lbfgs'
            ),
            'Linear SVM': SVC(
                kernel='linear',
                class_weight='balanced',
                random_state=1,
                probability=True
            ),
            'RBF SVM': SVC(
                kernel='rbf',
                C=1.0,
                class_weight='balanced',
                random_state=1,
                probability=True
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=1,
                max_depth=5
            ),
        }
    
    def preprocess(self, X_train: np.ndarray, X_test: np.ndarray = None) -> Tuple:
        """Applies normalization and PCA if requested"""
        # Normalize
        if self.normalize:
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_train = self.scaler.fit_transform(X_train)
            else:
                X_train = self.scaler.transform(X_train)
            
            if X_test is not None:
                X_test = self.scaler.transform(X_test)
        
        # PCA
        if self.use_pca:
            if self.pca is None:
                self.pca = PCA(n_components=self.use_pca, random_state=1)
                X_train = self.pca.fit_transform(X_train)
                print(f"  PCA: {X_train.shape[1]} components ({self.use_pca*100:.0f}% variance)")
            else:
                X_train = self.pca.transform(X_train)
            
            if X_test is not None:
                X_test = self.pca.transform(X_test)
        
        return (X_train, X_test) if X_test is not None else X_train
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, n_splits: int = 3):
        """
        Executes cross-validation with all classifiers
        
        Returns:
            Dictionary with detailed results
        """
        print(f"\n{'='*70}")
        print(f"BASELINE CLASSIFIER BENCHMARK ({self.n_classes}-CLASS)")
        print(f"{'='*70}")
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Classes: {np.bincount(y)}")
        print(f"CV Splits: {n_splits}")
        print(f"Normalization: {self.normalize}")
        print(f"PCA: {self.use_pca if self.use_pca else 'None'}")
        print(f"{'='*70}\n")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
        classifiers = self.get_classifiers()
        results = {}
        
        for clf_name, clf in classifiers.items():
            print(f"\n{'─'*70}")
            print(f"[Classifier] {clf_name}")
            print(f"{'─'*70}")
            
            fold_accs = []
            fold_bal_accs = []
            fold_aucs = []
            all_y_true = []
            all_y_pred = []
            all_y_proba = []
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                # Reset preprocessing
                self.scaler = None
                self.pca = None
                
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Preprocess
                X_train, X_test = self.preprocess(X_train, X_test)
                
                # Train
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                # Metrics
                from sklearn.metrics import accuracy_score, balanced_accuracy_score
                acc = accuracy_score(y_test, y_pred)
                bal_acc = balanced_accuracy_score(y_test, y_pred)
                
                fold_accs.append(acc)
                fold_bal_accs.append(bal_acc)
                
                # AUC
                if hasattr(clf, 'predict_proba'):
                    y_proba = clf.predict_proba(X_test)
                    try:
                        if self.n_classes == 2:
                            # Binary: use roc_auc directly
                            auc = roc_auc_score(y_test, y_proba[:, 1])
                        else:
                            # Multi-class: use ovr
                            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                        fold_aucs.append(auc)
                    except:
                        fold_aucs.append(np.nan)
                else:
                    y_proba = None
                    fold_aucs.append(np.nan)
                
                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)
                if y_proba is not None:
                    all_y_proba.extend(y_proba)
                
                # Per-fold report
                print(f"\nFold {fold}:")
                print(f"  Train: {np.bincount(y_train, minlength=self.n_classes)}")
                print(f"  Test:  {np.bincount(y_test, minlength=self.n_classes)}")
                print(f"  Pred:  {np.bincount(y_pred, minlength=self.n_classes)}")
                print(f"  Accuracy: {acc:.3f} | Balanced Acc: {bal_acc:.3f} | AUC: {fold_aucs[-1]:.3f}")
            
            # Summary
            print(f"\n{'─'*40}")
            print(f"SUMMARY ({clf_name}):")
            print(f"  Accuracy:         {np.mean(fold_accs):.3f} +/- {np.std(fold_accs):.3f}")
            print(f"  Balanced Acc:     {np.mean(fold_bal_accs):.3f} +/- {np.std(fold_bal_accs):.3f}")
            print(f"  AUC ({'binary' if self.n_classes == 2 else 'macro'}):      {np.nanmean(fold_aucs):.3f} +/- {np.nanstd(fold_aucs):.3f}")
            
            # Aggregated confusion matrix
            cm = confusion_matrix(all_y_true, all_y_pred)
            print(f"\nAggregated Confusion Matrix:")
            print(cm)
            print(f"\nPer-class accuracy:")
            for i in range(len(cm)):
                class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
                print(f"  Class {i}: {class_acc:.3f} ({cm[i, i]}/{cm[i].sum()})")
            
            results[clf_name] = {
                'accuracy': fold_accs,
                'balanced_accuracy': fold_bal_accs,
                'auc': fold_aucs,
                'confusion_matrix': cm,
                'predictions': (all_y_true, all_y_pred)
            }
        
        return results
    
    def plot_results(self, results: Dict, save_name: str = 'baseline_results.png'):
        """Visualizes benchmark results"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Balanced Accuracy comparison
        clf_names = list(results.keys())
        bal_accs = [results[name]['balanced_accuracy'] for name in clf_names]
        
        axes[0].boxplot(bal_accs, labels=clf_names)
        axes[0].set_ylabel('Balanced Accuracy', fontsize=12)
        axes[0].set_title('Classifier Comparison', fontsize=14, fontweight='bold')
        
        if self.n_classes == 2:
            axes[0].axhline(y=0.50, color='r', linestyle='--', alpha=0.5, label='Random (binary)')
        else:
            axes[0].axhline(y=0.33, color='r', linestyle='--', alpha=0.5, label='Random (3-class)')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Confusion matrix of best classifier
        best_clf = max(results.items(), 
                       key=lambda x: np.mean(x[1]['balanced_accuracy']))[0]
        cm = results[best_clf]['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], 
                    cbar_kws={'label': 'Count'})
        axes[1].set_xlabel('Predicted', fontsize=12)
        axes[1].set_ylabel('True', fontsize=12)
        axes[1].set_title(f'Best: {best_clf}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save in output directory
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n[SAVED] Results saved to: {save_path}")
        plt.close()



def main():
    # ──────────────────────────────────────────────────────────────────────
    # CONFIGURATION - MODIFY THESE PARAMETERS
    # ──────────────────────────────────────────────────────────────────────
    
    FEATURES_DIR = "MLA_features_resnet50_merged"
    CSV_PATH = "dataset.csv"
    FEATURE_KEY = "features"  # Key inside .h5 file
    OUTPUT_DIR = "baseline_results"  # Base output directory for all results
    
    # ============ BINARY vs MULTICLASS ============
    # Options:
    # - None: 3-class problem (default)
    # - 'class_2_vs_rest': Sarcomatoid (2) vs Others (0+1)
    # - 'class_0_vs_rest': Epithelioid (0) vs Others (1+2)
    # - 'class_0_vs_1': Epithelioid vs Biphasic (exclude Sarcomatoid)
    # - 'sarcomatoid_vs_rest': Same as class_2_vs_rest
    # - 'epithelioid_vs_non': Same as class_0_vs_rest
    
    BINARY_STRATEGY = None  # Change this for binary problem
    # BINARY_STRATEGY = 'class_2_vs_rest'  # Sarcomatoid vs Rest
    # ==============================================
    
    # Aggregation methods
    AGGREGATION_METHODS = ['mean', 'max', 'mean+max']
    
    # Preprocessing
    NORMALIZE = True
    USE_PCA = 0.95
    
    # Cross-validation
    N_SPLITS = 3
    
    # ──────────────────────────────────────────────────────────────────────
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    print(f"Dataset shape: {df.shape}")
    print(f"Original class distribution:\n{df['label'].value_counts().sort_index()}\n")
    
    # Binary conversion if requested
    if BINARY_STRATEGY is not None:
        print(f"\n{'='*70}")
        print(f"CONVERTING TO BINARY PROBLEM")
        print(f"{'='*70}")
        print(f"Strategy: {BinaryConverter.STRATEGIES.get(BINARY_STRATEGY, BINARY_STRATEGY)}")
        
        df, binary_desc = BinaryConverter.convert(df, BINARY_STRATEGY)
        n_classes = 2
        
        print(f"Description: {binary_desc}")
        print(f"New class distribution:\n{df['label'].value_counts().sort_index()}")
        print(f"{'='*70}\n")
    else:
        n_classes = 3
        print("Using 3-class problem (original labels)\n")
    
    # Load features
    loader = FeatureLoader(FEATURES_DIR, feature_key=FEATURE_KEY)
    features_dict = loader.load_all_slides(df['slide_id'].tolist())
    
    # Extract features directory name for output subfolder
    features_dir_name = Path(FEATURES_DIR).name
    
    # Test different aggregation methods
    all_results = {}
    
    for agg_method in AGGREGATION_METHODS:
        print(f"\n{'#'*70}")
        print(f"# AGGREGATION METHOD: {agg_method.upper()}")
        print(f"{'#'*70}")
        
        # Aggregate features
        X = []
        y = []
        for _, row in df.iterrows():
            feats = features_dict[row['slide_id']]
            X.append(FeatureAggregator.aggregate(feats, method=agg_method))
            y.append(row['label'])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nAggregated feature matrix: {X.shape}")
        print(f"Feature stats: min={X.min():.3f}, max={X.max():.3f}, "
              f"mean={X.mean():.3f}, std={X.std():.3f}")
        
        # Benchmark
        benchmark = BaselineClassifierBenchmark(
            normalize=NORMALIZE,
            use_pca=USE_PCA,
            n_classes=n_classes,
            output_dir=OUTPUT_DIR,
            features_dir_name=features_dir_name
        )
        
        results = benchmark.evaluate(X, y, n_splits=N_SPLITS)
        all_results[agg_method] = results
        
        # Plot for this method
        suffix = f"_{BINARY_STRATEGY}" if BINARY_STRATEGY else "_3class"
        benchmark.plot_results(
            results, 
            save_name=f'baseline_{agg_method}{suffix}.png'
        )
    
    # ──────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"FINAL SUMMARY - BEST CONFIGURATIONS ({n_classes}-CLASS)")
    print(f"{'='*70}\n")
    
    for agg_method, results in all_results.items():
        print(f"\n{agg_method.upper()}:")
        for clf_name, metrics in results.items():
            bal_acc = np.mean(metrics['balanced_accuracy'])
            auc = np.nanmean(metrics['auc'])
            print(f"  {clf_name:25s}: Bal_Acc={bal_acc:.3f}, AUC={auc:.3f}")
    
    # Best overall
    best_combo = None
    best_score = 0
    
    for agg_method, results in all_results.items():
        for clf_name, metrics in results.items():
            score = np.mean(metrics['balanced_accuracy'])
            if score > best_score:
                best_score = score
                best_combo = (agg_method, clf_name)
    
    print(f"\n{'─'*70}")
    print(f"[BEST OVERALL]")
    print(f"   Aggregation: {best_combo[0]}")
    print(f"   Classifier:  {best_combo[1]}")
    print(f"   Balanced Accuracy: {best_score:.3f}")
    print(f"{'─'*70}\n")
    
    # Interpretation
    print("\n[INTERPRETATION GUIDE]")
    print("─" * 70)
    
    threshold_good = 0.75 if n_classes == 2 else 0.65
    threshold_med = 0.65 if n_classes == 2 else 0.50
    
    if best_score > threshold_good:
        print("[EXCELLENT] Features are highly discriminative!")
        print("   -> Problem likely in CLAM architecture/training")
        print("   -> Try: simpler aggregation, reduce overfitting, tune hyperparams")
    elif best_score > threshold_med:
        print("[GOOD] Features capture signal but can be improved")
        print("   -> Try: different feature extractors, more augmentation")
        print("   -> Consider: ensemble methods, better preprocessing")
    else:
        print("[POOR] Features are not very discriminative")
        print("   -> Try: different backbone (ResNet -> ViT/CTransPath/UNI)")
        print("   -> Check: tissue extraction, stain normalization")
    
    # Minority class check (only for 3-class)
    if n_classes == 3:
        for agg_method, results in all_results.items():
            for clf_name, metrics in results.items():
                cm = metrics['confusion_matrix']
                if cm.shape[0] == 3:
                    class2_recall = cm[2, 2] / cm[2].sum() if cm[2].sum() > 0 else 0
                    if class2_recall < 0.25:
                        print(f"\n[WARNING] Class 2 poorly predicted ({class2_recall:.2f} recall)")
                        print("   -> STRONGLY RECOMMEND: Convert to binary problem")
                        print("   -> Set BINARY_STRATEGY = 'class_2_vs_rest'")
                        break


if __name__ == "__main__":
    main()