#!/usr/bin/env python3
"""
Extract length/shape information from all .npy files recursively and save to table format

Usage:
  python3 tools/extract_npy_lengths.py
  python3 tools/extract_npy_lengths.py --input /path/to/data --output /path/to/output.txt
"""
import os
import argparse
import numpy as np
from datetime import datetime


def scan_npy_files_recursive(root_dir, output_file=None, sfreq=256):
    """
    Recursively scan directory for .npy files and extract shape/length information
    
    Parameters:
    root_dir: root directory to scan
    output_file: output .txt file path (optional)
    sfreq: sampling frequency for duration calculation
    """
    
    if output_file is None:
        output_file = os.path.join(root_dir, "npy_lengths_summary.txt")
    
    print(f"Scanning directory: {root_dir}")
    print(f"Output file: {output_file}")
    print("-" * 80)
    
    # Collect all .npy files
    npy_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                npy_files.append(file_path)
    
    print(f"Found {len(npy_files)} .npy files")
    
    if not npy_files:
        print("No .npy files found!")
        return
    
    # Analyze each file
    results = []
    errors = []
    
    for file_path in npy_files:
        try:
            # Get relative path for cleaner display
            rel_path = os.path.relpath(file_path, root_dir)
            
            # Load file (memory mapped to avoid loading huge files)
            arr = np.load(file_path, mmap_mode='r')
            
            # Extract information
            shape = arr.shape
            dtype = arr.dtype
            ndim = arr.ndim
            
            # Calculate file size
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Infer dimensions based on shape
            if ndim == 1:
                n_channels = 1
                n_samples = shape[0]
                n_epochs = 1
            elif ndim == 2:
                # Assume (n_channels, n_samples) or (n_samples, n_channels)
                # Use heuristic: longer dimension is likely samples
                if shape[0] > shape[1]:
                    n_samples, n_channels = shape
                else:
                    n_channels, n_samples = shape
                n_epochs = 1
            elif ndim == 3:
                # Assume (n_epochs, n_channels, n_samples)
                n_epochs, n_channels, epoch_length = shape
                n_samples = n_epochs * epoch_length
            else:
                # Fallback
                n_channels = "Unknown"
                n_samples = np.prod(shape)
                n_epochs = "Unknown"
            
            # Calculate duration if possible
            if isinstance(n_samples, int) and sfreq > 0:
                duration_s = n_samples / sfreq
                duration_min = duration_s / 60
            else:
                duration_s = "Unknown"
                duration_min = "Unknown"
            
            results.append({
                'file_path': rel_path,
                'shape': shape,
                'ndim': ndim,
                'dtype': dtype,
                'n_epochs': n_epochs,
                'n_channels': n_channels,
                'n_samples': n_samples,
                'duration_s': duration_s,
                'duration_min': duration_min,
                'file_size_mb': file_size_mb
            })
            
            print(f"✓ {rel_path}: {shape}")
            
        except Exception as e:
            error_msg = f"✗ {os.path.relpath(file_path, root_dir)}: {str(e)}"
            errors.append(error_msg)
            print(error_msg)
    
    # Create output table
    print(f"\nCreating output table: {output_file}")
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("=" * 120 + "\n")
        f.write("NPY FILES LENGTH/SHAPE ANALYSIS\n")
        f.write("=" * 120 + "\n")
        f.write(f"Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Root Directory: {root_dir}\n")
        f.write(f"Total Files Found: {len(npy_files)}\n")
        f.write(f"Successfully Analyzed: {len(results)}\n")
        f.write(f"Errors: {len(errors)}\n")
        f.write(f"Sampling Frequency: {sfreq} Hz\n")
        f.write("=" * 120 + "\n\n")
        
        # Write detailed table
        if results:
            f.write("DETAILED FILE ANALYSIS\n")
            f.write("-" * 120 + "\n")
            
            # Table header
            header = f"{'File Path':<50} {'Shape':<20} {'NDim':<5} {'Epochs':<8} {'Channels':<8} {'Samples':<10} {'Duration(s)':<12} {'Duration(min)':<12} {'Size(MB)':<10}\n"
            f.write(header)
            f.write("-" * 120 + "\n")
            
            # Table rows
            for result in results:
                shape_str = str(result['shape'])
                duration_s_str = f"{result['duration_s']:.2f}" if isinstance(result['duration_s'], (int, float)) else str(result['duration_s'])
                duration_min_str = f"{result['duration_min']:.2f}" if isinstance(result['duration_min'], (int, float)) else str(result['duration_min'])
                
                row = f"{result['file_path']:<50} {shape_str:<20} {result['ndim']:<5} {str(result['n_epochs']):<8} {str(result['n_channels']):<8} {str(result['n_samples']):<10} {duration_s_str:<12} {duration_min_str:<12} {result['file_size_mb']:<10.2f}\n"
                f.write(row)
        
        # Write summary statistics
        if results:
            f.write("\n" + "=" * 120 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 120 + "\n")
            
            # Calculate summary stats
            shapes = [r['shape'] for r in results]
            ndims = [r['ndim'] for r in results]
            durations = [r['duration_s'] for r in results if isinstance(r['duration_s'], (int, float))]
            file_sizes = [r['file_size_mb'] for r in results]
            
            f.write(f"Shape Statistics:\n")
            f.write(f"  Unique Shapes: {len(set(shapes))}\n")
            f.write(f"  Most Common Shape: {max(set(shapes), key=shapes.count)}\n")
            f.write(f"  Dimensions: {sorted(set(ndims))}\n")
            
            if durations:
                f.write(f"\nDuration Statistics (seconds):\n")
                f.write(f"  Mean: {np.mean(durations):.2f}\n")
                f.write(f"  Median: {np.median(durations):.2f}\n")
                f.write(f"  Min: {np.min(durations):.2f}\n")
                f.write(f"  Max: {np.max(durations):.2f}\n")
                f.write(f"  Std: {np.std(durations):.2f}\n")
            
            f.write(f"\nFile Size Statistics (MB):\n")
            f.write(f"  Mean: {np.mean(file_sizes):.2f}\n")
            f.write(f"  Median: {np.median(file_sizes):.2f}\n")
            f.write(f"  Min: {np.min(file_sizes):.2f}\n")
            f.write(f"  Max: {np.max(file_sizes):.2f}\n")
            f.write(f"  Total: {np.sum(file_sizes):.2f}\n")
        
        # Write errors if any
        if errors:
            f.write("\n" + "=" * 120 + "\n")
            f.write("ERRORS\n")
            f.write("=" * 120 + "\n")
            for error in errors:
                f.write(error + "\n")
        
        f.write("\n" + "=" * 120 + "\n")
        f.write("END OF ANALYSIS\n")
        f.write("=" * 120 + "\n")
    
    print(f"\n✓ Analysis complete!")
    print(f"✓ Results saved to: {output_file}")
    print(f"✓ Processed {len(results)} files successfully")
    if errors:
        print(f"✗ {len(errors)} files had errors")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract .npy file lengths/shapes recursively')
    parser.add_argument('--input', 
                       default="/Users/wachiii/Workschii/BCI/ASD/brain-asd/data/data_children_no_artifact",
                       help='Root directory to scan for .npy files')
    parser.add_argument('--output', 
                       default=None,
                       help='Output .txt file path (default: npy_lengths_summary.txt in input dir)')
    parser.add_argument('--sfreq', type=float, default=256.0, 
                       help='Sampling frequency for duration calculation (Hz)')
    
    args = parser.parse_args()
    
    scan_npy_files_recursive(args.input, args.output, args.sfreq)
