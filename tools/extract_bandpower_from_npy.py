#!/usr/bin/env python3
"""
Extract bandpower (Delta/Theta/Alpha/Beta/Gamma) from .npy EEG files
Only the first 19 channels are used. Saves per-file (5 x 19) numpy arrays.

Usage:
  python3 tools/extract_bandpower_from_npy.py \
    --input /path/to/trimedData \
    --output /path/to/features --sfreq 256

This script will mirror ASD/HC subfolders based on their location in the input tree.
"""
import os
import argparse
import numpy as np
from scipy.signal import welch

BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}


def calculate_bandpower_matrix(signal_data, sfreq, bands=BANDS):
    """Return bandpower matrix of shape (n_bands, n_channels).

    signal_data: (n_channels, n_samples)
    """
    if signal_data.ndim != 2:
        raise ValueError('signal_data must be 2D (n_channels, n_samples)')

    n_channels, n_samples = signal_data.shape
    n_bands = len(bands)
    bandpower = np.zeros((n_bands, n_channels), dtype=float)

    # set nperseg for Welch: use 4s window or smaller if signal short
    nperseg = int(sfreq * 4)
    if nperseg > n_samples:
        nperseg = n_samples
    if nperseg < 1:
        nperseg = 1

    for ch in range(n_channels):
        freqs, psd = welch(signal_data[ch], fs=sfreq, nperseg=nperseg)
        for i, (low, high) in enumerate(bands.values()):
            mask = (freqs >= low) & (freqs < high)
            if not np.any(mask):
                bandpower[i, ch] = 0.0
            else:
                bandpower[i, ch] = np.trapz(psd[mask], freqs[mask])

    return bandpower


def process_input_tree(input_root, output_root, sfreq=256, verbose=True):
    """Walk input_root, find .npy files and save bandpower to output_root preserving asd/hc subfolders."""
    # Create output folders
    os.makedirs(output_root, exist_ok=True)
    asd_out = os.path.join(output_root, 'asd')
    hc_out = os.path.join(output_root, 'hc')
    os.makedirs(asd_out, exist_ok=True)
    os.makedirs(hc_out, exist_ok=True)

    # Walk tree
    total = 0
    processed = 0
    for dirpath, dirnames, filenames in os.walk(input_root):
        for fname in filenames:
            if not fname.endswith('.npy'):
                continue
            total += 1
            fpath = os.path.join(dirpath, fname)
            # Infer group by checking if 'asd' or 'hc' is in relative path segments
            rel = os.path.relpath(fpath, input_root)
            parts = rel.split(os.sep)
            group = None
            if 'asd' in [p.lower() for p in parts]:
                group = 'asd'
            elif 'hc' in [p.lower() for p in parts]:
                group = 'hc'
            else:
                # fallback: if parent folder name contains asd/hc
                parent = os.path.basename(os.path.dirname(fpath)).lower()
                if parent in ('asd', 'hc'):
                    group = parent

            if group is None:
                if verbose:
                    print(f"Skipping {fpath}: cannot infer group (asd/hc) from path")
                continue

            try:
                arr = np.load(fpath)
            except Exception as e:
                print(f"Failed to load {fpath}: {e}")
                continue

            # Normalize shape to (n_channels, n_samples)
            if arr.ndim == 1:
                # treat as single channel
                arr2 = arr.reshape(1, -1)
            elif arr.ndim == 2:
                arr2 = arr
            elif arr.ndim == 3:
                # e.g., (n_epochs, n_channels, n_samples) -> concatenate along time
                n_epochs, n_ch, ep_len = arr.shape
                arr2 = arr.reshape(n_epochs * ep_len, n_ch).T
            else:
                # try to reshape so last dim is time
                arr2 = arr.reshape(arr.shape[0], -1)

            # select first 19 channels (or fewer if not available)
            n_ch_avail = arr2.shape[0]
            use_ch = min(19, n_ch_avail)
            if use_ch < 1:
                print(f"Skipping {fpath}: no channels found")
                continue
            arr_sel = arr2[:use_ch, :]

            # If fewer than 19 channels, pad columns? We'll keep the available channels.

            # compute bandpower
            try:
                bp = calculate_bandpower_matrix(arr_sel, sfreq)
            except Exception as e:
                print(f"Error computing bandpower for {fpath}: {e}")
                continue

            # If we computed for fewer channels, ensure saved shape is (5, use_ch)
            out_base = os.path.splitext(fname)[0]
            out_name = out_base + '_bandpower.npy'
            out_path = os.path.join(asd_out if group == 'asd' else hc_out, out_name)

            try:
                np.save(out_path, bp)
                processed += 1
                if verbose:
                    print(f"Saved bandpower for {fpath} -> {out_path} (shape {bp.shape})")
            except Exception as e:
                print(f"Failed to save {out_path}: {e}")

    if verbose:
        print(f"Done. Found {total} .npy files, processed {processed} bandpower outputs.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Extract bandpower from .npy EEG files (first 19 channels)')
    p.add_argument('--input', default="/Users/wachiii/Workschii/BCI/ASD/brain-asd/data/data_adult_eyeclose_no_artefacts/trimedData", 
                   help='Input root directory containing .npy files (e.g. trimedData)')
    p.add_argument('--output', default="/Users/wachiii/Workschii/BCI/ASD/brain-asd/data/data_adult_eyeclose_no_artefacts/features", 
                   help='Output features directory (will contain asd/ and hc/)')
    p.add_argument('--sfreq', type=float, default=256.0, help='Sampling frequency (Hz)')
    args = p.parse_args()

    process_input_tree(args.input, args.output, sfreq=args.sfreq)
