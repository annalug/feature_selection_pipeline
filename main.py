import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, 
    chi2, 
    mutual_info_classif,
    VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

class MalwareFeatureSelector:
    """
    Feature selection for binary malware permission datasets
    """
    
    def __init__(self, df, target_col='class', variance_threshold=0.01):
        """
        Args:
            df: DataFrame with binary features and target column
            target_col: Name of the target column
            variance_threshold: Minimum variance to keep feature (default 0.01)
        """
        self.df = df.copy()
        self.target_col = target_col
        self.variance_threshold = variance_threshold
        
        # Separate features and target
        self.X = df.drop(columns=[target_col])
        self.y = df[target_col]
        self.original_features = self.X.columns.tolist()
        
        self.selected_features = []
        self.feature_scores = {}
        
    def remove_low_variance(self):
        """Remove features with very low variance"""
        selector = VarianceThreshold(threshold=self.variance_threshold)
        selector.fit(self.X)
        
        mask = selector.get_support()
        low_var_features = [f for f, m in zip(self.X.columns, mask) if m]
        
        print(f"Low variance removal: {len(self.X.columns)} -> {len(low_var_features)}")
        return low_var_features
    
    def correlation_filter(self, threshold=0.95):
        """Remove highly correlated features"""
        # Calculate correlation matrix
        corr_matrix = self.X.corr().abs()
        
        # Select upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation > threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        remaining_features = [f for f in self.X.columns if f not in to_drop]
        
        print(f"Correlation filter ({threshold}): {len(self.X.columns)} -> {len(remaining_features)}")
        return remaining_features
    
    def chi_squared_selection(self, k=50):
        """Chi-squared test for categorical features"""
        selector = SelectKBest(chi2, k=min(k, len(self.X.columns)))
        selector.fit(self.X, self.y)
        
        scores = pd.DataFrame({
            'feature': self.X.columns,
            'chi2_score': selector.scores_
        }).sort_values('chi2_score', ascending=False)
        
        self.feature_scores['chi2'] = scores
        
        top_features = scores.head(k)['feature'].tolist()
        print(f"Chi-squared top {k} features selected")
        
        return top_features
    
    def mutual_information_selection(self, k=50):
        """Mutual information for feature importance"""
        mi_scores = mutual_info_classif(self.X, self.y, random_state=42)
        
        scores = pd.DataFrame({
            'feature': self.X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        self.feature_scores['mutual_info'] = scores
        
        top_features = scores.head(k)['feature'].tolist()
        print(f"Mutual Information top {k} features selected")
        
        return top_features
    
    def random_forest_importance(self, k=50, n_estimators=100):
        """Random Forest feature importance"""
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X, self.y)
        
        scores = pd.DataFrame({
            'feature': self.X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_scores['rf_importance'] = scores
        
        top_features = scores.head(k)['feature'].tolist()
        print(f"Random Forest top {k} features selected")
        
        return top_features
    
    def ensemble_selection(self, methods=['chi2', 'mutual_info', 'rf'], 
                          min_votes=2, top_k=50):
        """
        Ensemble method: select features that appear in multiple methods
        
        Args:
            methods: List of methods to use
            min_votes: Minimum number of methods that must select a feature
            top_k: Number of top features to consider from each method
        """
        feature_votes = {}
        
        if 'chi2' in methods:
            for feat in self.chi_squared_selection(top_k):
                feature_votes[feat] = feature_votes.get(feat, 0) + 1
                
        if 'mutual_info' in methods:
            for feat in self.mutual_information_selection(top_k):
                feature_votes[feat] = feature_votes.get(feat, 0) + 1
                
        if 'rf' in methods:
            for feat in self.random_forest_importance(top_k):
                feature_votes[feat] = feature_votes.get(feat, 0) + 1
        
        # Select features with minimum votes
        selected = [f for f, v in feature_votes.items() if v >= min_votes]
        
        votes_df = pd.DataFrame(
            list(feature_votes.items()),
            columns=['feature', 'votes']
        ).sort_values('votes', ascending=False)
        
        self.feature_scores['ensemble_votes'] = votes_df
        
        print(f"\nEnsemble selection (min {min_votes} votes): {len(selected)} features")
        print(f"Vote distribution:\n{votes_df['votes'].value_counts().sort_index(ascending=False)}")
        
        return selected
    
    def auto_select(self, target_features=None, correlation_threshold=0.95):
        """
        Automatic feature selection pipeline
        
        Args:
            target_features: Target number of features (None for automatic)
            correlation_threshold: Threshold for correlation filtering
        """
        print(f"Starting with {len(self.X.columns)} features\n")
        
        # Step 1: Remove low variance
        features = self.remove_low_variance()
        self.X = self.X[features]
        
        # Step 2: Remove highly correlated
        features = self.correlation_filter(correlation_threshold)
        self.X = self.X[features]
        
        # Step 3: Ensemble selection
        if target_features is None:
            # Use 10% of original features or max 100
            target_features = min(100, max(10, len(self.original_features) // 10))
        
        self.selected_features = self.ensemble_selection(
            methods=['chi2', 'mutual_info', 'rf'],
            min_votes=2,
            top_k=target_features
        )
        
        print(f"\n{'='*60}")
        print(f"FINAL: {len(self.original_features)} -> {len(self.selected_features)} features")
        print(f"Reduction: {(1 - len(self.selected_features)/len(self.original_features))*100:.1f}%")
        print(f"{'='*60}")
        
        return self.selected_features
    
    def plot_feature_scores(self, top_n=20, save_path=None):
        """Plot feature importance scores"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Chi-squared
        if 'chi2' in self.feature_scores:
            df = self.feature_scores['chi2'].head(top_n)
            axes[0, 0].barh(df['feature'], df['chi2_score'])
            axes[0, 0].set_title('Chi-Squared Scores')
            axes[0, 0].invert_yaxis()
        
        # Mutual Information
        if 'mutual_info' in self.feature_scores:
            df = self.feature_scores['mutual_info'].head(top_n)
            axes[0, 1].barh(df['feature'], df['mi_score'])
            axes[0, 1].set_title('Mutual Information Scores')
            axes[0, 1].invert_yaxis()
        
        # Random Forest
        if 'rf_importance' in self.feature_scores:
            df = self.feature_scores['rf_importance'].head(top_n)
            axes[1, 0].barh(df['feature'], df['importance'])
            axes[1, 0].set_title('Random Forest Importance')
            axes[1, 0].invert_yaxis()
        
        # Ensemble Votes
        if 'ensemble_votes' in self.feature_scores:
            df = self.feature_scores['ensemble_votes'].head(top_n)
            axes[1, 1].barh(df['feature'], df['votes'])
            axes[1, 1].set_title('Ensemble Votes')
            axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.close()
    
    def get_reduced_dataset(self):
        """Return dataset with selected features only"""
        return self.df[self.selected_features + [self.target_col]]
    
    def save_results(self, output_dir, dataset_name):
        """Save all results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save selected features
        pd.DataFrame({'feature': self.selected_features}).to_csv(
            output_dir / f'{dataset_name}_selected_features.csv', index=False
        )
        
        # Save all scores
        for method, scores in self.feature_scores.items():
            scores.to_csv(output_dir / f'{dataset_name}_{method}_scores.csv', index=False)
        
        # Save reduced dataset
        reduced_df = self.get_reduced_dataset()
        reduced_df.to_csv(output_dir / f'{dataset_name}_reduced.csv', index=False)
        
        # Return summary stats
        return {
            'dataset': dataset_name,
            'original_features': len(self.original_features),
            'selected_features': len(self.selected_features),
            'reduction_percentage': round((1 - len(self.selected_features)/len(self.original_features))*100, 2),
            'original_samples': len(self.df),
            'malware_samples': int(self.y.sum()),
            'benign_samples': int(len(self.y) - self.y.sum())
        }


class BatchFeatureSelector:
    """
    Process multiple datasets from a directory
    """
    
    def __init__(self, data_dir='data', output_dir='results', target_col='class'):
        """
        Args:
            data_dir: Directory containing CSV files
            output_dir: Directory to save results
            target_col: Name of target column
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.target_col = target_col
        self.summaries = []
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
    def find_csv_files(self):
        """Find all CSV files in data directory"""
        csv_files = list(self.data_dir.glob('*.csv'))
        print(f"Found {len(csv_files)} CSV files in {self.data_dir}/")
        return csv_files
    
    def process_dataset(self, csv_path, target_features=None, correlation_threshold=0.95):
        """
        Process a single dataset
        
        Args:
            csv_path: Path to CSV file
            target_features: Target number of features
            correlation_threshold: Correlation threshold
        """
        # Extract dataset name (everything before .csv)
        dataset_name = csv_path.stem
        
        print(f"\n{'='*80}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*80}\n")
        
        try:
            # Load dataset
            df = pd.read_csv(csv_path)
            print(f"Dataset shape: {df.shape}")
            
            # Check if target column exists
            if self.target_col not in df.columns:
                print(f"ERROR: Target column '{self.target_col}' not found in {dataset_name}")
                print(f"Available columns: {df.columns.tolist()}")
                return None
            
            print(f"Target distribution:\n{df[self.target_col].value_counts()}\n")
            
            # Create selector
            selector = MalwareFeatureSelector(df, target_col=self.target_col)
            
            # Auto select features
            selected_features = selector.auto_select(
                target_features=target_features,
                correlation_threshold=correlation_threshold
            )
            
            # Create dataset-specific output directory
            dataset_output = self.output_dir / dataset_name
            dataset_output.mkdir(exist_ok=True)
            
            # Plot results
            selector.plot_feature_scores(
                top_n=30,
                save_path=dataset_output / f'{dataset_name}_feature_importance.png'
            )
            
            # Save results and get summary
            summary = selector.save_results(dataset_output, dataset_name)
            
            print(f"\n✓ Results saved to {dataset_output}/")
            
            return summary
            
        except Exception as e:
            print(f"\n✗ ERROR processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_all(self, target_features=None, correlation_threshold=0.95):
        """
        Process all CSV files in data directory
        
        Args:
            target_features: Target number of features for all datasets
            correlation_threshold: Correlation threshold for all datasets
        """
        start_time = datetime.now()
        
        # Find all CSV files
        csv_files = self.find_csv_files()
        
        if not csv_files:
            print(f"No CSV files found in {self.data_dir}/")
            return
        
        # Process each file
        for i, csv_path in enumerate(csv_files, 1):
            print(f"\n[{i}/{len(csv_files)}]")
            summary = self.process_dataset(
                csv_path,
                target_features=target_features,
                correlation_threshold=correlation_threshold
            )
            
            if summary:
                self.summaries.append(summary)
        
        # Save global summary
        self.save_global_summary()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Processed: {len(self.summaries)}/{len(csv_files)} datasets")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Results saved to: {self.output_dir}/")
        print(f"Global summary: {self.output_dir}/summary.txt")
        print(f"{'='*80}\n")
    
    def save_global_summary(self):
        """Save summary of all processed datasets"""
        if not self.summaries:
            print("No summaries to save")
            return
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(self.summaries)
        
        # Save as CSV
        summary_df.to_csv(self.output_dir / 'summary.csv', index=False)
        
        # Save as formatted text
        summary_path = self.output_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FEATURE SELECTION SUMMARY - ALL DATASETS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total datasets processed: {len(self.summaries)}\n\n")
            
            # Overall statistics
            f.write("-"*80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Average reduction: {summary_df['reduction_percentage'].mean():.2f}%\n")
            f.write(f"Total original features: {summary_df['original_features'].sum()}\n")
            f.write(f"Total selected features: {summary_df['selected_features'].sum()}\n")
            f.write(f"Total samples: {summary_df['original_samples'].sum()}\n")
            f.write(f"Total malware samples: {summary_df['malware_samples'].sum()}\n")
            f.write(f"Total benign samples: {summary_df['benign_samples'].sum()}\n\n")
            
            # Individual dataset details
            f.write("-"*80 + "\n")
            f.write("INDIVIDUAL DATASET DETAILS\n")
            f.write("-"*80 + "\n\n")
            
            for summary in self.summaries:
                f.write(f"Dataset: {summary['dataset']}\n")
                f.write(f"  Original features: {summary['original_features']}\n")
                f.write(f"  Selected features: {summary['selected_features']}\n")
                f.write(f"  Reduction: {summary['reduction_percentage']}%\n")
                f.write(f"  Samples: {summary['original_samples']} "
                       f"(Malware: {summary['malware_samples']}, "
                       f"Benign: {summary['benign_samples']})\n")
                f.write(f"  Output: {self.output_dir}/{summary['dataset']}/\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"\n✓ Global summary saved to {summary_path}")


# Main execution
if __name__ == "__main__":
    # Configuration
    DATA_DIR = 'data'
    OUTPUT_DIR = 'results'
    TARGET_COL = 'class'
    
    # Optional: set target features (None for automatic)
    TARGET_FEATURES = None  # or specific number like 50
    CORRELATION_THRESHOLD = 0.95
    
    # Create batch processor
    processor = BatchFeatureSelector(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        target_col=TARGET_COL
    )
    
    # Process all datasets
    processor.process_all(
        target_features=TARGET_FEATURES,
        correlation_threshold=CORRELATION_THRESHOLD
    )
