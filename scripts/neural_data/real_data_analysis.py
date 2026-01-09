"""
Real Neural Data Analysis with RRMD
====================================
Test RRMD on open-source datasets and compare to SOTA methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import eig
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our validation functions
from scripts.math.math_experiments import (
    compute_pca_axis, 
    compute_rotational_symmetry,
    validate_neural_conditions,
    angle_between_vectors,
    print_progress
)

# Try to import NWB reader
try:
    from pynwb import NWBHDF5IO
    HAS_PYNWB = True
except ImportError:
    HAS_PYNWB = False
    print("Warning: pynwb not installed. Install with: pip install pynwb")


# ============================================================================
# SOTA COMPARISON METHODS
# ============================================================================

def jpca_analysis(neural_data, n_components=6):
    """
    jPCA: Find rotational dynamics using skew-symmetric decomposition.
    
    Reference: Churchland et al. (2012) Nature
    """
    print("  Running jPCA...")
    
    # Center data
    X = neural_data.T  # (time, neurons)
    X_centered = X - np.mean(X, axis=0)
    
    # Compute derivative (velocity)
    dX = np.gradient(X_centered, axis=0)
    
    # Find linear mapping M such that dX ≈ M @ X
    # Using least squares: M = (dX.T @ X) @ (X.T @ X)^-1
    XTX = X_centered.T @ X_centered
    dXTX = dX.T @ X_centered
    
    # Regularize to avoid singularity
    M = dXTX @ np.linalg.pinv(XTX + 1e-6 * np.eye(XTX.shape[0]))
    
    # Decompose M into symmetric and skew-symmetric parts
    M_skew = (M - M.T) / 2  # Rotational component
    
    # Find eigenvalues/eigenvectors of skew-symmetric part
    eigenvalues, eigenvectors = eig(M_skew)
    
    # Sort by imaginary part (rotation frequency)
    rotation_strength = np.abs(np.imag(eigenvalues))
    sorted_idx = np.argsort(rotation_strength)[::-1]
    
    jpca_plane = np.real(eigenvectors[:, sorted_idx[:2]])
    rotation_freq = np.imag(eigenvalues[sorted_idx[0]])
    
    # Project data onto jPCA plane
    projected = X_centered @ jpca_plane
    
    return {
        'M_skew': M_skew,
        'jpca_plane': jpca_plane,
        'rotation_freq': rotation_freq,
        'projected': projected,
        'rotation_strength': rotation_strength[sorted_idx[0]]
    }


def compute_tangling(neural_traj):
    """
    Compute trajectory tangling (Russo et al. 2018).
    Lower tangling = smoother dynamics.
    """
    # Compute pairwise distances at each time
    n_time = neural_traj.shape[0]
    
    tangles = []
    for t in range(1, n_time - 1):
        # Distance between states at time t
        d_state = np.linalg.norm(neural_traj[t] - neural_traj[t-1])
        
        # Distance between velocities
        v1 = neural_traj[t] - neural_traj[t-1]
        v2 = neural_traj[t+1] - neural_traj[t]
        d_vel = np.linalg.norm(v2 - v1)
        
        if d_state > 1e-8:
            tangles.append(d_vel / d_state)
    
    return np.mean(tangles)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_nlb_mc_maze():
    """
    Load Neural Latents Benchmark - MC_Maze dataset.
    
    Dataset: Motor cortex during maze navigation (Shenoy/Sussillo labs)
    Source: https://neurallatents.github.io/
    """
    print("Attempting to load NLB MC_Maze dataset...")
    
    # Try multiple possible paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    
    possible_paths = [
        os.path.join(project_root, "raw_data/000128/sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"),
        "raw_data/000128/sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb",
        "000128/sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"
    ]
    
    nwb_path = None
    for path in possible_paths:
        if os.path.exists(path):
            nwb_path = path
            break
    
    if nwb_path is None:
        print(f"  ✗ Data not found in raw_data/000128/")
        return None, None
    
    if not HAS_PYNWB:
        print("  ✗ pynwb not installed. Install with: pip install pynwb")
        return None, None
    
    print(f"  Loading from {nwb_path}")
    
    try:
        with NWBHDF5IO(nwb_path, 'r') as io:
            nwb = io.read()
            
            units = nwb.units
            n_neurons = len(units)
            print(f"  Found {n_neurons} neurons")
            
            trial_info = nwb.trials
            max_time = trial_info['stop_time'][:].max()
            
            bin_size = 0.05  # 50ms bins
            n_bins = int(max_time / bin_size)
            
            print(f"  Binning spikes ({max_time:.1f}s, {bin_size*1000:.0f}ms bins)...")
            
            firing_rates = np.zeros((n_neurons, n_bins))
            
            for neuron_idx in range(min(n_neurons, 200)):
                spike_times = units['spike_times'][neuron_idx]
                spike_bins = (spike_times / bin_size).astype(int)
                spike_bins = spike_bins[spike_bins < n_bins]
                
                counts = np.bincount(spike_bins, minlength=n_bins)
                firing_rates[neuron_idx, :] = counts / bin_size
                
                if neuron_idx % 50 == 0:
                    print(f"    Processed {neuron_idx}/{min(n_neurons, 200)} neurons...")
            
            firing_rates = firing_rates[:min(n_neurons, 200), :]
            
            print(f"  ✓ Loaded {firing_rates.shape[0]} neurons × {firing_rates.shape[1]} bins")
            return firing_rates, "MC_Maze Motor Cortex"
            
    except Exception as e:
        print(f"  ✗ Error reading NWB file: {e}")
        return None, None


def load_steinmetz_visual():
    """Load Steinmetz Neuropixels dataset if available."""
    print("Attempting to load Steinmetz Neuropixels dataset...")
    
    data_path = "data/steinmetz_visual.npy"
    if not os.path.exists(data_path):
        print(f"  ✗ Data not found at {data_path}")
        print(f"  Download from: https://figshare.com/articles/dataset/steinmetz/9974357")
        return None, None
    
    print(f"  Loading from {data_path}")
    data = np.load(data_path)
    print(f"  ✓ Loaded {data.shape[0]} neurons × {data.shape[1]} timepoints")
    return data, "Steinmetz Visual Cortex"


def load_dream_dataset():
    """Load DREAM motor cortex dataset if available."""
    print("Attempting to load DREAM motor cortex dataset...")
    
    data_path = "data/dream_motor.npy"
    if not os.path.exists(data_path):
        print(f"  ✗ Data not found at {data_path}")
        print(f"  Download from: https://crcns.org/data-sets/motor-cortex/dream")
        return None, None
    
    print(f"  Loading from {data_path}")
    data = np.load(data_path)
    print(f"  ✓ Loaded {data.shape[0]} neurons × {data.shape[1]} timepoints")
    return data, "DREAM Motor Cortex"


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_neural_data(neural_data, smooth_sigma=2, z_score=True):
    """
    Preprocess neural data for RRMD analysis.
    
    Parameters:
    -----------
    neural_data : array (n_neurons, n_timepoints)
    smooth_sigma : float, temporal smoothing width (bins)
    z_score : bool, whether to z-score each neuron
    """
    print("\nPreprocessing data...")
    
    # Smooth
    if smooth_sigma > 0:
        smoothed = gaussian_filter1d(neural_data, sigma=smooth_sigma, axis=1)
        print(f"  ✓ Temporal smoothing (σ={smooth_sigma} bins)")
    else:
        smoothed = neural_data.copy()
    
    # Z-score
    if z_score:
        mean = np.mean(smoothed, axis=1, keepdims=True)
        std = np.std(smoothed, axis=1, keepdims=True) + 1e-8
        preprocessed = (smoothed - mean) / std
        print(f"  ✓ Z-score normalization")
    else:
        preprocessed = smoothed
    
    return preprocessed


# ============================================================================
# RRMD ANALYSIS WITH SOTA COMPARISON
# ============================================================================

def analyze_with_rrmd(neural_data, dataset_name, n_components=3):
    """
    Apply RRMD to neural data and compare to SOTA methods.
    """
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE ANALYSIS: {dataset_name}")
    print(f"{'='*70}\n")
    
    n_neurons, n_timepoints = neural_data.shape
    print(f"Data shape: {n_neurons} neurons × {n_timepoints} timepoints")
    
    # Step 1: Validate conditions
    print("\n[1/5] Checking RRMD applicability conditions...")
    conditions = validate_neural_conditions(neural_data)
    
    # Step 2: Dimensionality reduction
    print("\n[2/5] Reducing to 3D via PCA...")
    pca = PCA(n_components=n_components)
    neural_3d = pca.fit_transform(neural_data.T).T
    
    variance_explained = np.sum(pca.explained_variance_ratio_[:3]) * 100
    print(f"  Variance explained by 3 PCs: {variance_explained:.1f}%")
    
    # Step 3: RRMD analysis
    print("\n[3/5] RRMD: Detecting rotation axis...")
    rotation_axis = compute_pca_axis(neural_3d.T)
    symmetry_score = compute_rotational_symmetry(neural_3d.T, rotation_axis)
    
    print(f"  Rotation axis: [{rotation_axis[0]:6.3f}, {rotation_axis[1]:6.3f}, {rotation_axis[2]:6.3f}]")
    print(f"  Symmetry score: {symmetry_score:.4f} (lower = more symmetric)")
    
    # Step 4: jPCA comparison
    print("\n[4/5] jPCA: Finding rotational dynamics...")
    jpca_result = jpca_analysis(neural_data)
    print(f"  Rotation frequency: {jpca_result['rotation_freq']:.4f}")
    print(f"  Rotation strength: {jpca_result['rotation_strength']:.4f}")
    
    # Step 5: Compute metrics
    print("\n[5/5] Computing comparison metrics...")
    
    # Tangling
    tangling_rrmd = compute_tangling(neural_3d.T)
    tangling_jpca = compute_tangling(jpca_result['projected'])
    
    print(f"  Tangling (RRMD): {tangling_rrmd:.4f}")
    print(f"  Tangling (jPCA): {tangling_jpca:.4f}")
    
    # Visualization
    visualize_comparison(neural_3d, rotation_axis, jpca_result, dataset_name, pca,
                        symmetry_score, tangling_rrmd, tangling_jpca)
    
    return {
        'conditions': conditions,
        'pca': pca,
        'neural_3d': neural_3d,
        'rotation_axis': rotation_axis,
        'symmetry_score': symmetry_score,
        'variance_explained': variance_explained,
        'jpca': jpca_result,
        'tangling_rrmd': tangling_rrmd,
        'tangling_jpca': tangling_jpca
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_comparison(neural_3d, rotation_axis, jpca_result, dataset_name, pca,
                        symmetry_score, tangling_rrmd, tangling_jpca):
    """
    Create comprehensive visualization comparing RRMD and jPCA.
    """
    fig = plt.figure(figsize=(18, 12))
    times = np.arange(neural_3d.shape[1])
    
    # Row 1: RRMD Analysis
    # 1. RRMD 3D trajectory
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    scatter = ax1.scatter(neural_3d[0], neural_3d[1], neural_3d[2], 
                         c=times, cmap='viridis', s=5, alpha=0.5)
    
    # Plot rotation axis
    axis_scale = np.max(np.abs(neural_3d)) * 1.5
    ax1.plot([0, rotation_axis[0]*axis_scale], 
             [0, rotation_axis[1]*axis_scale],
             [0, rotation_axis[2]*axis_scale], 
             'r-', linewidth=3, label='Rotation Axis')
    
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.set_title('RRMD: 3D Trajectory')
    ax1.legend()
    
    # 2. RRMD rotation plane
    ax2 = fig.add_subplot(3, 3, 2)
    centered = neural_3d.T - np.mean(neural_3d.T, axis=0)
    projected = centered - np.outer(centered @ rotation_axis, rotation_axis)
    
    scatter2 = ax2.scatter(projected[:, 0], projected[:, 1], 
                          c=times, cmap='viridis', s=5, alpha=0.5)
    ax2.set_xlabel('Dim 1')
    ax2.set_ylabel('Dim 2')
    ax2.set_title(f'RRMD: Rotation Plane (Sym={symmetry_score:.3f})')
    ax2.axis('equal')
    
    # 3. RRMD metrics
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.text(0.1, 0.8, 'RRMD Metrics:', fontsize=14, fontweight='bold')
    ax3.text(0.1, 0.65, f'Symmetry: {symmetry_score:.4f}', fontsize=11)
    ax3.text(0.1, 0.55, f'Tangling: {tangling_rrmd:.4f}', fontsize=11)
    ax3.text(0.1, 0.45, f'Axis: [{rotation_axis[0]:.2f}, {rotation_axis[1]:.2f}, {rotation_axis[2]:.2f}]', 
             fontsize=10)
    ax3.text(0.1, 0.3, '✓ Lower symmetry = better', fontsize=9, color='green')
    ax3.text(0.1, 0.2, '✓ Lower tangling = better', fontsize=9, color='green')
    ax3.axis('off')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # Row 2: jPCA Analysis
    # 4. jPCA plane
    ax4 = fig.add_subplot(3, 3, 4)
    jpca_proj = jpca_result['projected']
    scatter4 = ax4.scatter(jpca_proj[:, 0], jpca_proj[:, 1], 
                          c=times, cmap='plasma', s=5, alpha=0.5)
    ax4.set_xlabel('jPC1')
    ax4.set_ylabel('jPC2')
    ax4.set_title(f'jPCA: Rotational Plane (ω={jpca_result["rotation_freq"]:.3f})')
    ax4.axis('equal')
    
    # 5. jPCA phase plot
    ax5 = fig.add_subplot(3, 3, 5)
    angles = np.arctan2(jpca_proj[:, 1], jpca_proj[:, 0])
    ax5.plot(angles, linewidth=1)
    ax5.set_xlabel('Time (bins)')
    ax5.set_ylabel('Phase (rad)')
    ax5.set_title('jPCA: Phase Evolution')
    ax5.grid(True, alpha=0.3)
    
    # 6. jPCA metrics
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.text(0.1, 0.8, 'jPCA Metrics:', fontsize=14, fontweight='bold')
    ax6.text(0.1, 0.65, f'Rot. Frequency: {jpca_result["rotation_freq"]:.4f}', fontsize=11)
    ax6.text(0.1, 0.55, f'Rot. Strength: {jpca_result["rotation_strength"]:.4f}', fontsize=11)
    ax6.text(0.1, 0.45, f'Tangling: {tangling_jpca:.4f}', fontsize=11)
    ax6.text(0.1, 0.25, '✓ Higher rotation strength = more rotational', fontsize=9, color='green')
    ax6.axis('off')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    
    # Row 3: Comparisons
    # 7. PCA spectrum
    ax7 = fig.add_subplot(3, 3, 7)
    variance_ratios = pca.explained_variance_ratio_[:15]
    ax7.bar(range(1, len(variance_ratios)+1), variance_ratios * 100)
    ax7.set_xlabel('Principal Component')
    ax7.set_ylabel('Variance Explained (%)')
    ax7.set_title('PCA Spectrum')
    ax7.grid(True, alpha=0.3)
    
    # 8. Method comparison
    ax8 = fig.add_subplot(3, 3, 8)
    methods = ['RRMD', 'jPCA']
    tangling_vals = [tangling_rrmd, tangling_jpca]
    colors = ['blue', 'orange']
    ax8.bar(methods, tangling_vals, color=colors, alpha=0.7)
    ax8.set_ylabel('Tangling')
    ax8.set_title('Tangling Comparison (Lower = Better)')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Summary
    ax9 = fig.add_subplot(3, 3, 9)
    winner_tangling = 'RRMD' if tangling_rrmd < tangling_jpca else 'jPCA'
    
    ax9.text(0.1, 0.9, 'COMPARISON SUMMARY', fontsize=14, fontweight='bold')
    ax9.text(0.1, 0.75, f'Dataset: {dataset_name}', fontsize=10)
    ax9.text(0.1, 0.65, f'Neurons: {pca.n_features_in_}', fontsize=10)
    ax9.text(0.1, 0.55, f'Timepoints: {neural_3d.shape[1]}', fontsize=10)
    ax9.text(0.1, 0.4, f'Winner (Tangling): {winner_tangling}', fontsize=12, 
             fontweight='bold', color='green')
    ax9.text(0.1, 0.25, f'  RRMD: {tangling_rrmd:.4f}', fontsize=10)
    ax9.text(0.1, 0.15, f'  jPCA: {tangling_jpca:.4f}', fontsize=10)
    ax9.axis('off')
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    
    plt.suptitle(f'RRMD vs jPCA: {dataset_name}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Use absolute path for saving
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    output_dir = os.path.join(project_root, 'outputs', 'figs')
    os.makedirs(output_dir, exist_ok=True)
    
    filename = dataset_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    output_path = os.path.join(output_dir, f'comparison_{filename}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison to 'outputs/figs/comparison_{filename}.png'")


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def main():
    """Run RRMD analysis and compare to SOTA methods."""
    print("\n" + "="*70)
    print("RRMD vs SOTA COMPARISON ON REAL NEURAL DATA")
    print("="*70 + "\n")
    
    datasets = [
        ("mc_maze", load_nlb_mc_maze),
        ("dream", load_dream_dataset),
        ("steinmetz", load_steinmetz_visual)
    ]
    
    results = {}
    analyzed_count = 0
    
    for name, loader_func in datasets:
        try:
            # Load data
            neural_data, dataset_name = loader_func()
            
            # Skip if data not available
            if neural_data is None:
                print(f"  → Skipping {name}\n")
                continue
            
            analyzed_count += 1
            
            # Preprocess
            preprocessed = preprocess_neural_data(neural_data, smooth_sigma=2)
            
            # Analyze
            result = analyze_with_rrmd(preprocessed, dataset_name)
            results[name] = result
            
            print(f"\n{'='*70}\n")
            
        except Exception as e:
            print(f"\n✗ Error analyzing {name}: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    if analyzed_count == 0:
        print("="*70)
        print("NO DATASETS AVAILABLE")
        print("="*70)
        print("\nDownload real data from:")
        print("  - MC_Maze: dandi download DANDI:000128")
        print("  - Steinmetz: https://figshare.com/articles/dataset/steinmetz/9974357")
        print("  - DREAM: https://crcns.org/data-sets/motor-cortex/dream")
        return {}
    
    # Summary comparison
    print("="*70)
    print("SUMMARY: RRMD vs jPCA")
    print("="*70)
    
    for name, result in results.items():
        rrmd_better = result['tangling_rrmd'] < result['tangling_jpca']
        winner = "RRMD ✓" if rrmd_better else "jPCA"
        
        print(f"\n{name.upper()}:")
        print(f"  Variance in 3D: {result['variance_explained']:.1f}%")
        print(f"  RRMD Symmetry: {result['symmetry_score']:.4f}")
        print(f"  RRMD Tangling: {result['tangling_rrmd']:.4f}")
        print(f"  jPCA Tangling: {result['tangling_jpca']:.4f}")
        print(f"  Winner: {winner}")
    
    # Overall statistics
    if len(results) > 0:
        rrmd_wins = sum(1 for r in results.values() if r['tangling_rrmd'] < r['tangling_jpca'])
        jpca_wins = len(results) - rrmd_wins
        
        print(f"\n{'='*70}")
        print(f"OVERALL: RRMD wins {rrmd_wins}/{len(results)} datasets")
        print(f"{'='*70}")
        
        # Save results to file
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '../..'))
        results_dir = os.path.join(project_root, 'outputs', 'results')
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, 'real_data_analysis_results.txt')
        
        with open(results_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RRMD vs jPCA: Real Neural Data Analysis\n")
            f.write("="*70 + "\n")
            f.write(f"Analysis Date: {timestamp}\n\n")
            
            for name, result in results.items():
                rrmd_better = result['tangling_rrmd'] < result['tangling_jpca']
                winner = "RRMD" if rrmd_better else "jPCA"
                improvement = abs(result['tangling_jpca'] - result['tangling_rrmd']) / result['tangling_jpca'] * 100
                
                f.write(f"Dataset: {name.upper()}\n")
                f.write("-" * 70 + "\n")
                f.write(f"  Neurons analyzed: {result['pca'].n_features_in_}\n")
                f.write(f"  Timepoints: {result['neural_3d'].shape[1]}\n")
                f.write(f"  Variance in 3D: {result['variance_explained']:.1f}%\n\n")
                f.write(f"  RRMD Results:\n")
                f.write(f"    - Symmetry Score: {result['symmetry_score']:.4f}\n")
                f.write(f"    - Tangling: {result['tangling_rrmd']:.4f}\n\n")
                f.write(f"  jPCA Results:\n")
                f.write(f"    - Rotation Frequency: {result['jpca']['rotation_freq']:.4f}\n")
                f.write(f"    - Tangling: {result['tangling_jpca']:.4f}\n\n")
                f.write(f"  Winner: {winner} ({improvement:.1f}% improvement)\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("OVERALL SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"Datasets Analyzed: {len(results)}\n")
            f.write(f"RRMD Wins: {rrmd_wins}\n")
            f.write(f"jPCA Wins: {jpca_wins}\n")
            f.write(f"Success Rate: {rrmd_wins/len(results)*100:.1f}%\n")
        
        print(f"\n✓ Results saved to: outputs/results/real_data_analysis_results.txt")
    
    plt.show()
    return results


if __name__ == "__main__":
    results = main()
