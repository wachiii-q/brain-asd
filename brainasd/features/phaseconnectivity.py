import numpy as np
import itertools
from scipy.signal import hilbert



class EEGPhaseConnectivity:
    def __init__(self, data, fs):
        """
        Initialize the Phase Connectivity class.

        :param data: EEG data (n_epochs, n_channels, n_samples)
        :param fs: Sampling frequency (Hz)
        """
        self.data = data
        self.fs = fs
        self.n_epochs, self.n_channels, self.n_samples = data.shape
        self.channel_pairs = list(itertools.combinations(range(self.n_channels), 2))


    def compute_phase(self, signal):
        """Compute the instantaneous phase using the Hilbert Transform."""
        analytic_signal = hilbert(signal)
        return np.angle(analytic_signal)  


    def compute_plv(self, phase1, phase2):
        """Compute Phase-Locking Value (PLV)."""
        phase_diff = np.exp(1j * (phase1 - phase2))
        return np.abs(np.mean(phase_diff))


    def compute_pli(self, phase1, phase2):
        """Compute Phase Lag Index (PLI)."""
        phase_diff = phase1 - phase2
        return np.abs(np.mean(np.sign(np.sin(phase_diff))))


    def compute_wpli(self, phase1, phase2):
        """Compute Weighted Phase Lag Index (wPLI)."""
        phase_diff = phase1 - phase2
        imag_part = np.imag(np.exp(1j * phase_diff))
        return np.abs(np.mean(np.sign(imag_part) * imag_part)) / (np.mean(np.abs(imag_part)) + 1e-10)


    def run(self):
        """
        Compute PLV, PLI, and wPLI for all channel pairs.
        
        :return: Dictionary containing connectivity measures.
                 Shape: (n_epochs, n_pairs)
        """
        results = {
            "PLV": np.zeros((self.n_epochs, len(self.channel_pairs))),
            "PLI": np.zeros((self.n_epochs, len(self.channel_pairs))),
            "wPLI": np.zeros((self.n_epochs, len(self.channel_pairs)))
        }
        for epoch_idx in range(self.n_epochs):
            epoch = self.data[epoch_idx]
            phases = np.array([self.compute_phase(epoch[ch]) for ch in range(self.n_channels)])
            for pair_idx, (ch1, ch2) in enumerate(self.channel_pairs):
                phase1, phase2 = phases[ch1], phases[ch2]
                results["PLV"][epoch_idx, pair_idx] = self.compute_plv(phase1, phase2)
                results["PLI"][epoch_idx, pair_idx] = self.compute_pli(phase1, phase2)
                results["wPLI"][epoch_idx, pair_idx] = self.compute_wpli(phase1, phase2)
        return results



if __name__ == "__main__":
    np.random.seed(42)
    n_epochs, n_channels, n_samples = 10, 16, 1000
    eeg_data = np.random.randn(n_epochs, n_channels, n_samples)
    fs = 250  # Hz

    phase_conn = PhaseConnectivity(eeg_data, fs)
    connectivity_results = phase_conn.run()

    # Print results
    print("PLV shape:", connectivity_results["PLV"].shape)  # Expected: (10, 120)
    print("PLI shape:", connectivity_results["PLI"].shape)  # Expected: (10, 120)
    print("wPLI shape:", connectivity_results["wPLI"].shape)  # Expected: (10, 120)
    print("Sample PLV values:", connectivity_results["PLV"][0, :5])  # First 5 values
