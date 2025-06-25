import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import os
import pandas as pd
from datetime import datetime

# Import ADBench components
from adbench.myutils import Utils
from adbench.run import RunPipeline
from adbench.baseline.PyOD import PYOD

# Create a timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create results directory if it doesn't exist
results_dir = f"results_{timestamp}"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Settings
RANDOM_SEED = 42
ANOMALY_TYPES = ['local', 'cluster', 'dependency', 'global']
PYOD_MODELS = ['IForest', 'HBOS', 'LODA', 'COPOD', 'ECOD', 'XGBOD']

# Results storage
performance_results = {
    'anomaly_type': [],
    'model': [],
    'auc': [],
    'precision': [],
    'recall': [],
    'f1': []
}

# Download and load cardio dataset
utils = Utils()
utils.download_datasets(repo='github')
data = np.load('adbench/datasets/Classical/6_cardio.npz', allow_pickle=True)
X_original, y_original = data['X'], data['y']

# Function to run experiment for one anomaly type
def run_experiment(anomaly_type):
    print(f"\n{'='*50}\nRunning experiment for anomaly type: {anomaly_type}\n{'='*50}")
    
    # Create a copy of the original data to avoid modifying it between runs
    X = X_original.copy()
    y = y_original.copy()
    
    # Create dataset dictionary
    dataset = {'X': X, 'y': y}
    
    # Set up pipeline for synthetic anomaly generation
    pipeline = RunPipeline(
        suffix=f'cardio_{anomaly_type}',
        parallel='unsupervise',
        realistic_synthetic_mode=anomaly_type,
        noise_type=None
    )
    
    # # Run pipeline to generate synthetic anomalies
    results = pipeline.run(dataset=dataset)
    
    # Get updated dataset with synthetic anomalies
    X_with_synthetic = dataset['X']
    y_with_synthetic = dataset['y']
    
    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_with_synthetic, y_with_synthetic, 
        test_size=0.4, random_state=RANDOM_SEED, 
        stratify=y_with_synthetic
    )
    
    # Create t-SNE visualization
    print("Generating t-SNE visualization...")
    
    # Create and fit t-SNE (may take a few minutes for larger datasets)
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30)
    X_train_tsne = tsne.fit_transform(X_train)
    X_test_tsne = tsne.fit_transform(X_test)
    
    # Create visualization of train and test data
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot training data
    axs[0].scatter(X_train_tsne[y_train == 0, 0], X_train_tsne[y_train == 0, 1], 
                   c='blue', label='Normal', alpha=0.5, s=10)
    axs[0].scatter(X_train_tsne[y_train == 1, 0], X_train_tsne[y_train == 1, 1], 
                   c='red', label='Anomaly', alpha=0.8, s=20)
    axs[0].set_title(f'Training Data t-SNE (Anomaly Type: {anomaly_type})')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot test data
    axs[1].scatter(X_test_tsne[y_test == 0, 0], X_test_tsne[y_test == 0, 1], 
                   c='blue', label='Normal', alpha=0.5, s=10)
    axs[1].scatter(X_test_tsne[y_test == 1, 0], X_test_tsne[y_test == 1, 1], 
                   c='red', label='Anomaly', alpha=0.8, s=20)
    axs[1].set_title(f'Test Data t-SNE (Anomaly Type: {anomaly_type})')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    # Add super title
    plt.suptitle(f'ADBench Synthetic Anomaly Visualization - {anomaly_type.capitalize()} Anomalies', fontsize=16)
    plt.tight_layout()
    
    # Save visualization
    tsne_filename = os.path.join(results_dir, f'tsne_visualization_{anomaly_type}.png')
    plt.savefig(tsne_filename, dpi=300)
    plt.close()
    print(f"t-SNE visualization saved to {tsne_filename}")
    
    # Evaluate PYOD models
    print("\nEvaluating PYOD models...")
    
    for model_name in PYOD_MODELS:
        print(f"  - Training {model_name}...")
        
        # Initialize model
        model = PYOD(seed=RANDOM_SEED, model_name=model_name)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Get anomaly scores
        y_scores = model.predict_score(X_test)
        
        # Convert scores to binary predictions (using default threshold)
        y_pred = (y_scores >= 0.5).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_scores)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store results
        performance_results['anomaly_type'].append(anomaly_type)
        performance_results['model'].append(model_name)
        performance_results['auc'].append(auc)
        performance_results['precision'].append(precision)
        performance_results['recall'].append(recall)
        performance_results['f1'].append(f1)
        
        print(f"    AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return X_with_synthetic, y_with_synthetic

# Run experiments for each anomaly type
for anomaly_type in ANOMALY_TYPES:
    X_with_synthetic, y_with_synthetic = run_experiment(anomaly_type)

# Create performance comparison dataframe
performance_df = pd.DataFrame(performance_results)

# Save performance results to CSV
performance_filename = os.path.join(results_dir, 'model_performance.csv')
performance_df.to_csv(performance_filename, index=False)
print(f"\nPerformance results saved to {performance_filename}")

# Create bar chart of AUC scores for each model across anomaly types
plt.figure(figsize=(12, 8))

# Set width of bars
bar_width = 0.15
index = np.arange(len(PYOD_MODELS))

# Plot bars for each anomaly type
for i, anomaly_type in enumerate(ANOMALY_TYPES):
    # Filter data for this anomaly type
    data_subset = performance_df[performance_df['anomaly_type'] == anomaly_type]
    
    # Extract AUC values in the same order as PYOD_MODELS
    auc_values = [data_subset[data_subset['model'] == model]['auc'].values[0] for model in PYOD_MODELS]
    
    # Plot bars
    plt.bar(index + i * bar_width, auc_values, bar_width, 
            label=f'{anomaly_type.capitalize()} Anomalies')

# Add labels and legend
plt.xlabel('PyOD Model')
plt.ylabel('AUC Score')
plt.title('PYOD Model Performance Across Different Anomaly Types')
plt.xticks(index + bar_width * (len(ANOMALY_TYPES) - 1) / 2, PYOD_MODELS, rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# Save comparison chart
comparison_filename = os.path.join(results_dir, 'model_comparison.png')
plt.savefig(comparison_filename, dpi=300)
plt.close()
print(f"Model comparison chart saved to {comparison_filename}")

# Final visualization of all anomaly types together for comparison
print("\nCreating final comparison visualization...")

# Create a figure with one row for each anomaly type
fig, axs = plt.subplots(len(ANOMALY_TYPES), 1, figsize=(12, 16))

for i, anomaly_type in enumerate(ANOMALY_TYPES):
    # Re-run experiment to get fresh data (but don't evaluate models again)
    dataset = {'X': X_original.copy(), 'y': y_original.copy()}
    
    pipeline = RunPipeline(
        suffix=f'cardio_{anomaly_type}_final',
        parallel='unsupervise',
        realistic_synthetic_mode=anomaly_type,
        noise_type=None
    )
    
    results = pipeline.run(dataset=dataset)
    
    X_with_synthetic = dataset['X']
    y_with_synthetic = dataset['y']
    
    # Generate t-SNE visualization
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30)
    X_tsne = tsne.fit_transform(X_with_synthetic)
    
    # Plot in the corresponding axis
    axs[i].scatter(X_tsne[y_with_synthetic == 0, 0], X_tsne[y_with_synthetic == 0, 1],
                  c='blue', label='Normal', alpha=0.5, s=10)
    axs[i].scatter(X_tsne[y_with_synthetic == 1, 0], X_tsne[y_with_synthetic == 1, 1],
                  c='red', label='Anomaly', alpha=0.8, s=20)
    axs[i].set_title(f'{anomaly_type.capitalize()} Anomalies')
    axs[i].legend()
    axs[i].grid(True, linestyle='--', alpha=0.7)

plt.suptitle('Comparison of Different Synthetic Anomaly Types (t-SNE Visualization)', fontsize=16)
plt.tight_layout()

# Save final comparison
final_comparison_filename = os.path.join(results_dir, 'anomaly_types_comparison.png')
plt.savefig(final_comparison_filename, dpi=300)
plt.close()
print(f"Final comparison visualization saved to {final_comparison_filename}")

print(f"\nAll results stored in directory: {results_dir}")