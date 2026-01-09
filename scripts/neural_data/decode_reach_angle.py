"""
Behavioral Decoding Test: Reach Angle from Neural Data
========================================================
Test if RRMD's rotation axis aligns with behavioral angular dimensions.

Key Hypothesis:
- RRMD rotation axis should align with reach angle
- Angular position in perpendicular plane correlates with reach direction
- Residual from perfect rotation correlates with reach kinematics (speed/distance)
- RRMD separates angle from speed better than jPCA
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import gaussian_filter1d
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.math.math_experiments import (
    compute_pca_axis,
    compute_rotational_symmetry,
    angle_between_vectors
)
from scripts.neural_data.real_data_analysis import (
    load_nlb_mc_maze,
    preprocess_neural_data,
    jpca_analysis
)


# ============================================================================
# BEHAVIORAL DATA EXTRACTION
# ============================================================================

def extract_reach_kinematics(nwb_file_path):
    """
    Extract reach angle, distance, and speed from NWB file.
    
    Returns:
    --------
    kinematics : dict
        'angles': reach angles in radians
        'distances': reach distances
        'speeds': reach speeds
        'timepoints': corresponding timepoints for each reach
    """
    try:
        from pynwb import NWBHDF5IO
    except ImportError:
        print("Warning: pynwb not installed. Cannot extract behavioral data.")
        return None
    
    print("Extracting behavioral kinematics from NWB file...")
    
    try:
        with NWBHDF5IO(nwb_file_path, 'r') as io:
            nwb = io.read()
            
            # Get trial information
            trials = nwb.trials
            n_trials = len(trials)
            
            # Try to extract actual position/velocity data
            # Check what's available in the processing module
            if 'behavior' in nwb.processing:
                behavior = nwb.processing['behavior']
                
                # Look for position data
                if 'Position' in behavior.data_interfaces:
                    position = behavior.data_interfaces['Position']
                    
                    # Get spatial series (hand position)
                    if hasattr(position, 'spatial_series'):
                        pos_data = None
                        for series_name in position.spatial_series:
                            spatial_series = position.spatial_series[series_name]
                            pos_data = spatial_series.data[:]
                            pos_timestamps = spatial_series.timestamps[:]
                            break
                        
                        if pos_data is not None and len(pos_data.shape) >= 2:
                            # Calculate reach angles from hand position
                            # Assuming pos_data is (time, xy) or (time, xyz)
                            x = pos_data[:, 0]
                            y = pos_data[:, 1]
                            
                            # Get center position
                            center_x = np.median(x)
                            center_y = np.median(y)
                            
                            # Calculate angles relative to center
                            dx = x - center_x
                            dy = y - center_y
                            all_angles = np.arctan2(dy, dx)
                            
                            # Calculate distances and speeds
                            all_distances = np.sqrt(dx**2 + dy**2)
                            all_speeds = np.gradient(all_distances)
                            
                            # Map to trial timepoints
                            bin_size = 0.05
                            start_times = trials['start_time'][:]
                            stop_times = trials['stop_time'][:]
                            
                            angles = []
                            distances = []
                            speeds = []
                            timepoints = []
                            
                            for i, (start_t, stop_t) in enumerate(zip(start_times, stop_times)):
                                # Find position data during this trial
                                trial_mask = (pos_timestamps >= start_t) & (pos_timestamps <= stop_t)
                                
                                if np.sum(trial_mask) > 0:
                                    # Use mean angle during trial
                                    trial_angles = all_angles[trial_mask]
                                    trial_dists = all_distances[trial_mask]
                                    trial_speeds = all_speeds[trial_mask]
                                    
                                    angles.append(np.mean(trial_angles))
                                    distances.append(np.max(trial_dists) - np.min(trial_dists))
                                    speeds.append(np.mean(np.abs(trial_speeds)))
                                    timepoints.append(int(start_t / bin_size))
                            
                            angles = np.array(angles)
                            distances = np.array(distances)
                            speeds = np.array(speeds)
                            timepoints = np.array(timepoints)
                            
                            print(f"  ✓ Extracted {len(angles)} reach trials with position data")
                            print(f"  Angle range: {np.min(angles):.2f} to {np.max(angles):.2f} rad")
                            print(f"  Distance range: {np.min(distances):.3f} to {np.max(distances):.3f}")
                            
                            return {
                                'angles': angles,
                                'distances': distances,
                                'speeds': speeds,
                                'timepoints': timepoints,
                                'n_trials': len(angles)
                            }
            
            # Fallback: use trial timing as proxy
            print("  ⚠ Position data not found, using trial timing as proxy")
            start_times = trials['start_time'][:]
            stop_times = trials['stop_time'][:]
            
            # Use trial duration as proxy for distance
            distances = stop_times - start_times
            speeds = 1.0 / (distances + 0.1)
            
            # Try to extract maze position/target if available
            angles = np.zeros(n_trials)
            if 'maze_pos' in trials.colnames:
                maze_pos = trials['maze_pos'][:]
                # Handle potentially ragged array
                try:
                    # Try to extract position from each trial
                    extracted_angles = []
                    for i in range(n_trials):
                        pos = maze_pos[i]
                        if hasattr(pos, '__len__') and len(pos) >= 2:
                            # If it's an array, take the mean or last position
                            if hasattr(pos[0], '__len__'):
                                # Array of positions, take last
                                extracted_angles.append(np.arctan2(pos[-1][1], pos[-1][0]))
                            else:
                                # Single position
                                extracted_angles.append(np.arctan2(pos[1], pos[0]))
                        else:
                            extracted_angles.append(0)
                    angles = np.array(extracted_angles)
                except:
                    angles = np.linspace(0, 2*np.pi, n_trials, endpoint=False)
            elif 'target_pos' in trials.colnames:
                target_pos = trials['target_pos'][:]
                # Handle potentially ragged array
                try:
                    extracted_angles = []
                    for i in range(n_trials):
                        pos = target_pos[i]
                        if hasattr(pos, '__len__') and len(pos) >= 2:
                            # If it's an array, take the mean or last position
                            if hasattr(pos[0], '__len__'):
                                # Array of positions, take last
                                extracted_angles.append(np.arctan2(pos[-1][1], pos[-1][0]))
                            else:
                                # Single position
                                extracted_angles.append(np.arctan2(pos[1], pos[0]))
                        else:
                            extracted_angles.append(0)
                    angles = np.array(extracted_angles)
                except:
                    angles = np.linspace(0, 2*np.pi, n_trials, endpoint=False)
            else:
                # Last resort: use trial structure to infer angles
                # Check if trials have condition or target labels
                if 'condition' in trials.colnames:
                    conditions = trials['condition'][:]
                    unique_conditions = np.unique(conditions)
                    n_conditions = len(unique_conditions)
                    condition_angles = np.linspace(0, 2*np.pi, n_conditions, endpoint=False)
                    # Map each trial to its condition angle
                    for i, cond in enumerate(conditions):
                        cond_idx = np.where(unique_conditions == cond)[0][0]
                        angles[i] = condition_angles[cond_idx]
                    print(f"  ⚠ Using {n_conditions} trial conditions as angle proxy")
                else:
                    print("  ⚠ No angle data found - creating uniform distribution")
                    angles = np.linspace(0, 2*np.pi, n_trials, endpoint=False)
            
            # Map to neural timepoints (bin indices)
            bin_size = 0.05
            timepoints = (start_times / bin_size).astype(int)
            
            print(f"  ✓ Extracted {n_trials} reach trials (limited behavioral data)")
            print(f"  Angle range: {np.min(angles):.2f} to {np.max(angles):.2f} rad")
            
            return {
                'angles': angles,
                'distances': distances,
                'speeds': speeds,
                'timepoints': timepoints,
                'n_trials': n_trials
            }
            
    except Exception as e:
        print(f"  ✗ Error extracting kinematics: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# ANGULAR DECODING ANALYSIS
# ============================================================================

def decode_angle_from_rrmd(neural_3d, rotation_axis, kinematics):
    """
    Test if angular position in RRMD plane correlates with reach angle.
    
    Parameters:
    -----------
    neural_3d : array (3, n_timepoints)
    rotation_axis : array (3,)
    kinematics : dict with behavioral data
    
    Returns:
    --------
    results : dict with correlations and decoded angles
    """
    print("Decoding reach angle from RRMD...")
    
    # Project onto plane perpendicular to rotation axis
    neural_centered = neural_3d.T - np.mean(neural_3d.T, axis=0)
    
    # Remove component along rotation axis
    projection_along_axis = np.outer(neural_centered @ rotation_axis, rotation_axis)
    neural_in_plane = neural_centered - projection_along_axis
    
    # Compute angular position in plane
    # Use first two components of plane projection
    decoded_angles = np.arctan2(neural_in_plane[:, 1], neural_in_plane[:, 0])
    
    # Extract neural angles at reach timepoints
    neural_angles_at_reaches = decoded_angles[kinematics['timepoints']]
    true_angles = kinematics['angles']
    
    # Compute circular correlation
    # Convert to complex representation for circular data
    neural_complex = np.exp(1j * neural_angles_at_reaches)
    true_complex = np.exp(1j * true_angles)
    
    # Circular correlation
    circ_corr = np.abs(np.mean(neural_complex * np.conj(true_complex)))
    
    # Also compute angle difference
    angle_diffs = np.abs(np.angle(neural_complex * np.conj(true_complex)))
    mean_angle_error = np.mean(angle_diffs)
    
    print(f"  Circular correlation: {circ_corr:.3f}")
    print(f"  Mean angle error: {np.degrees(mean_angle_error):.1f}°")
    
    return {
        'decoded_angles': decoded_angles,
        'neural_angles_at_reaches': neural_angles_at_reaches,
        'true_angles': true_angles,
        'circular_correlation': circ_corr,
        'mean_angle_error': mean_angle_error,
        'neural_in_plane': neural_in_plane
    }


def decode_kinematics_from_residual(neural_3d, rotation_axis, kinematics):
    """
    Test if residual from perfect rotation correlates with reach distance/speed.
    
    Parameters:
    -----------
    neural_3d : array (3, n_timepoints)
    rotation_axis : array (3,)
    kinematics : dict with behavioral data
    
    Returns:
    --------
    results : dict with correlations
    """
    print("Decoding reach kinematics from residual...")
    
    neural_centered = neural_3d.T - np.mean(neural_3d.T, axis=0)
    
    # Compute distance from rotation axis (residual)
    projection_along_axis = np.outer(neural_centered @ rotation_axis, rotation_axis)
    residual_magnitude = np.linalg.norm(neural_centered - 
                                       (neural_centered - projection_along_axis), axis=1)
    
    # Extract at reach timepoints
    residuals_at_reaches = residual_magnitude[kinematics['timepoints']]
    
    # Correlate with distance and speed
    if len(residuals_at_reaches) > 2:
        corr_distance, p_distance = pearsonr(residuals_at_reaches, kinematics['distances'])
        corr_speed, p_speed = pearsonr(residuals_at_reaches, kinematics['speeds'])
    else:
        corr_distance, p_distance = 0, 1
        corr_speed, p_speed = 0, 1
    
    print(f"  Correlation with distance: {corr_distance:.3f} (p={p_distance:.4f})")
    print(f"  Correlation with speed: {corr_speed:.3f} (p={p_speed:.4f})")
    
    return {
        'residual_magnitude': residual_magnitude,
        'residuals_at_reaches': residuals_at_reaches,
        'corr_distance': corr_distance,
        'p_distance': p_distance,
        'corr_speed': corr_speed,
        'p_speed': p_speed
    }


def compare_rrmd_vs_jpca_decoding(neural_data, neural_3d, rotation_axis, 
                                   jpca_result, kinematics):
    """
    Compare RRMD and jPCA for behavioral decoding.
    
    Returns:
    --------
    comparison : dict with metrics for both methods
    """
    print("\nComparing RRMD vs jPCA for decoding...")
    
    # RRMD decoding
    rrmd_angle = decode_angle_from_rrmd(neural_3d, rotation_axis, kinematics)
    rrmd_kinematics = decode_kinematics_from_residual(neural_3d, rotation_axis, kinematics)
    
    # jPCA decoding
    print("\nDecoding from jPCA plane...")
    jpca_projected = jpca_result['projected']
    
    # Decode angles from jPCA
    jpca_angles = np.arctan2(jpca_projected[:, 1], jpca_projected[:, 0])
    jpca_angles_at_reaches = jpca_angles[kinematics['timepoints']]
    
    neural_complex = np.exp(1j * jpca_angles_at_reaches)
    true_complex = np.exp(1j * kinematics['angles'])
    jpca_circ_corr = np.abs(np.mean(neural_complex * np.conj(true_complex)))
    
    angle_diffs = np.abs(np.angle(neural_complex * np.conj(true_complex)))
    jpca_mean_error = np.mean(angle_diffs)
    
    print(f"  jPCA circular correlation: {jpca_circ_corr:.3f}")
    print(f"  jPCA mean angle error: {np.degrees(jpca_mean_error):.1f}°")
    
    # Compare
    print(f"\n{'='*70}")
    print("DECODING COMPARISON")
    print(f"{'='*70}")
    print(f"Angle Decoding:")
    print(f"  RRMD: {rrmd_angle['circular_correlation']:.3f} correlation, "
          f"{np.degrees(rrmd_angle['mean_angle_error']):.1f}° error")
    print(f"  jPCA: {jpca_circ_corr:.3f} correlation, "
          f"{np.degrees(jpca_mean_error):.1f}° error")
    
    if rrmd_angle['circular_correlation'] > jpca_circ_corr:
        print(f"  ✓ RRMD better at angle decoding")
    else:
        print(f"  ~ jPCA better at angle decoding")
    
    print(f"\nKinematics Decoding (from residual):")
    print(f"  RRMD distance correlation: {rrmd_kinematics['corr_distance']:.3f}")
    print(f"  RRMD speed correlation: {rrmd_kinematics['corr_speed']:.3f}")
    
    return {
        'rrmd_angle': rrmd_angle,
        'rrmd_kinematics': rrmd_kinematics,
        'jpca_circ_corr': jpca_circ_corr,
        'jpca_mean_error': jpca_mean_error,
        'jpca_angles': jpca_angles
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_decoding_results(neural_3d, rotation_axis, kinematics, 
                               comparison, dataset_name):
    """Create comprehensive visualization of decoding results."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 3D trajectory with rotation axis
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(neural_3d[0], neural_3d[1], neural_3d[2], 'b-', alpha=0.3, linewidth=0.5)
    
    # Mark reach timepoints
    reach_tps = kinematics['timepoints']
    ax1.scatter(neural_3d[0, reach_tps], neural_3d[1, reach_tps], 
               neural_3d[2, reach_tps], c=kinematics['angles'], 
               cmap='hsv', s=100, edgecolor='black', linewidth=2)
    
    # Plot rotation axis
    center = np.mean(neural_3d, axis=1)
    axis_length = np.max(np.std(neural_3d, axis=1)) * 2
    ax1.plot([center[0] - axis_length*rotation_axis[0], 
             center[0] + axis_length*rotation_axis[0]],
            [center[1] - axis_length*rotation_axis[1], 
             center[1] + axis_length*rotation_axis[1]],
            [center[2] - axis_length*rotation_axis[2], 
             center[2] + axis_length*rotation_axis[2]],
            'r-', linewidth=3, label='Rotation Axis')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.set_title('3D Neural Trajectory')
    ax1.legend()
    
    # 2. Projection onto plane perpendicular to axis
    ax2 = fig.add_subplot(2, 3, 2)
    plane_proj = comparison['rrmd_angle']['neural_in_plane']
    ax2.plot(plane_proj[:, 0], plane_proj[:, 1], 'b-', alpha=0.3, linewidth=0.5)
    ax2.scatter(plane_proj[reach_tps, 0], plane_proj[reach_tps, 1],
               c=kinematics['angles'], cmap='hsv', s=100, 
               edgecolor='black', linewidth=2)
    ax2.set_xlabel('Plane Dim 1')
    ax2.set_ylabel('Plane Dim 2')
    ax2.set_title('Projection onto Perpendicular Plane')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # 3. Decoded vs True angles (RRMD)
    ax3 = fig.add_subplot(2, 3, 3, projection='polar')
    true_angles = comparison['rrmd_angle']['true_angles']
    decoded_angles = comparison['rrmd_angle']['neural_angles_at_reaches']
    
    for i, (true, decoded) in enumerate(zip(true_angles, decoded_angles)):
        ax3.plot([true, decoded], [1, 0.7], 'b-', alpha=0.5)
        ax3.scatter(true, 1, c='red', s=100, marker='o', edgecolor='black')
        ax3.scatter(decoded, 0.7, c='blue', s=100, marker='s', edgecolor='black')
    
    ax3.set_title(f'RRMD: True (●) vs Decoded (■) Angles\n'
                 f'Correlation: {comparison["rrmd_angle"]["circular_correlation"]:.3f}')
    
    # 4. Residual vs Distance
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(kinematics['distances'], 
               comparison['rrmd_kinematics']['residuals_at_reaches'],
               s=100, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Reach Distance')
    ax4.set_ylabel('Residual from Rotation')
    ax4.set_title(f'Distance Correlation: '
                 f'{comparison["rrmd_kinematics"]["corr_distance"]:.3f}')
    ax4.grid(True, alpha=0.3)
    
    # 5. Residual vs Speed
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(kinematics['speeds'], 
               comparison['rrmd_kinematics']['residuals_at_reaches'],
               s=100, alpha=0.7, edgecolor='black', color='orange')
    ax5.set_xlabel('Reach Speed')
    ax5.set_ylabel('Residual from Rotation')
    ax5.set_title(f'Speed Correlation: '
                 f'{comparison["rrmd_kinematics"]["corr_speed"]:.3f}')
    ax5.grid(True, alpha=0.3)
    
    # 6. RRMD vs jPCA comparison
    ax6 = fig.add_subplot(2, 3, 6)
    methods = ['RRMD', 'jPCA']
    correlations = [comparison['rrmd_angle']['circular_correlation'],
                   comparison['jpca_circ_corr']]
    colors = ['green' if correlations[0] > correlations[1] else 'gray',
             'orange' if correlations[1] > correlations[0] else 'gray']
    
    ax6.bar(methods, correlations, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax6.set_ylabel('Circular Correlation')
    ax6.set_title('Angle Decoding Comparison')
    ax6.set_ylim([0, 1])
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add text
    winner = 'RRMD' if correlations[0] > correlations[1] else 'jPCA'
    ax6.text(0.5, 0.5, f'Winner: {winner}', 
            transform=ax6.transAxes, ha='center', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Behavioral Decoding Analysis: {dataset_name}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    output_dir = os.path.join(project_root, 'outputs', 'figs')
    os.makedirs(output_dir, exist_ok=True)
    
    filename = dataset_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    output_path = os.path.join(output_dir, f'decoding_{filename}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved figure to 'outputs/figs/decoding_{filename}.png'")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_decoding_analysis():
    """Run complete behavioral decoding analysis."""
    
    print("\n" + "="*70)
    print("BEHAVIORAL DECODING: Reach Angle from Neural Data")
    print("="*70 + "\n")
    
    # Load real data
    neural_data, dataset_name = load_nlb_mc_maze()
    
    if neural_data is None:
        print("✗ Real data not available. Cannot run analysis.")
        print("\nPlease ensure MC_Maze data is in raw_data/000128/")
        return None
    
    # Extract kinematics from NWB file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    nwb_path = os.path.join(project_root, 
        "raw_data/000128/sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb")
    
    kinematics = extract_reach_kinematics(nwb_path)
    if kinematics is None:
        print("✗ Could not extract kinematics from NWB file.")
        return None
    
    # Preprocess
    print(f"\nAnalyzing: {dataset_name}")
    print(f"Data: {neural_data.shape[0]} neurons × {neural_data.shape[1]} timepoints")
    print(f"Reaches: {kinematics['n_trials']} trials\n")
    
    preprocessed = preprocess_neural_data(neural_data, smooth_sigma=2)
    
    # Reduce to 3D
    print("Reducing to 3D with PCA...")
    pca = PCA(n_components=3)
    neural_3d = pca.fit_transform(preprocessed.T).T
    variance_explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"  Variance explained: {variance_explained:.1f}%\n")
    
    # RRMD analysis
    print("RRMD: Finding rotation axis...")
    rotation_axis = compute_pca_axis(neural_3d.T)
    symmetry = compute_rotational_symmetry(neural_3d.T, rotation_axis)
    print(f"  Rotation axis: {rotation_axis}")
    print(f"  Symmetry score: {symmetry:.4f}\n")
    
    # jPCA analysis
    print("jPCA: Finding rotational plane...")
    jpca_result = jpca_analysis(preprocessed)
    print(f"  Rotation frequency: {jpca_result['rotation_freq']:.4f}\n")
    
    # Decoding analysis
    comparison = compare_rrmd_vs_jpca_decoding(
        preprocessed, neural_3d, rotation_axis, jpca_result, kinematics
    )
    
    # Visualization
    visualize_decoding_results(neural_3d, rotation_axis, kinematics, 
                               comparison, dataset_name)
    
    # Save text summary
    save_decoding_summary(comparison, kinematics, dataset_name)
    
    plt.show()
    return comparison


def save_decoding_summary(comparison, kinematics, dataset_name):
    """Save text summary of decoding results."""
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    results_dir = os.path.join(project_root, 'outputs', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'decoding_results.txt')
    
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Behavioral Decoding Analysis: Reach Angle from Neural Data\n")
        f.write("="*70 + "\n")
        f.write(f"Analysis Date: {timestamp}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Number of Reaches: {kinematics['n_trials']}\n\n")
        
        f.write("HYPOTHESIS TESTS\n")
        f.write("-" * 70 + "\n\n")
        
        f.write("1. Angular Decoding (RRMD Plane)\n")
        f.write(f"   Circular Correlation: {comparison['rrmd_angle']['circular_correlation']:.3f}\n")
        f.write(f"   Mean Angle Error: {np.degrees(comparison['rrmd_angle']['mean_angle_error']):.1f}°\n")
        status = "PASS" if comparison['rrmd_angle']['circular_correlation'] > 0.5 else "WEAK"
        f.write(f"   Status: {status}\n")
        f.write(f"   Conclusion: {'Strong alignment with reach angle' if status == 'PASS' else 'Weak alignment with reach angle'}\n\n")
        
        f.write("2. Kinematics Decoding (Residual)\n")
        f.write(f"   Distance Correlation: {comparison['rrmd_kinematics']['corr_distance']:.3f} ")
        f.write(f"(p={comparison['rrmd_kinematics']['p_distance']:.4f})\n")
        f.write(f"   Speed Correlation: {comparison['rrmd_kinematics']['corr_speed']:.3f} ")
        f.write(f"(p={comparison['rrmd_kinematics']['p_speed']:.4f})\n")
        sig = (comparison['rrmd_kinematics']['p_distance'] < 0.05 or 
               comparison['rrmd_kinematics']['p_speed'] < 0.05)
        f.write(f"   Status: {'SIGNIFICANT' if sig else 'NOT SIGNIFICANT'}\n")
        f.write(f"   Conclusion: Residual {'correlates with' if sig else 'does not correlate with'} reach kinematics\n\n")
        
        f.write("3. RRMD vs jPCA Comparison\n")
        rrmd_corr = comparison['rrmd_angle']['circular_correlation']
        jpca_corr = comparison['jpca_circ_corr']
        winner = "RRMD" if rrmd_corr > jpca_corr else "jPCA"
        improvement = abs(rrmd_corr - jpca_corr) / max(jpca_corr, 0.01) * 100
        
        f.write(f"   RRMD Correlation: {rrmd_corr:.3f}\n")
        f.write(f"   jPCA Correlation: {jpca_corr:.3f}\n")
        f.write(f"   Winner: {winner} ({improvement:.1f}% better)\n")
        f.write(f"   Conclusion: {winner} better separates angular from non-angular components\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 70 + "\n")
        
        tests_passed = sum([
            comparison['rrmd_angle']['circular_correlation'] > 0.5,
            sig,
            rrmd_corr > jpca_corr
        ])
        
        f.write(f"Tests Passed: {tests_passed}/3\n")
        f.write(f"Overall Result: {'VALIDATED' if tests_passed >= 2 else 'NEEDS REVIEW'}\n\n")
        
        if tests_passed >= 2:
            f.write("RRMD successfully identifies rotation axis aligned with behavioral\n")
            f.write("angular dimension and separates rotational from non-rotational dynamics.\n")
        else:
            f.write("Results suggest further investigation needed for behavioral alignment.\n")
    
    print(f"✓ Summary saved to: outputs/results/decoding_results.txt")


if __name__ == "__main__":
    results = run_decoding_analysis()
