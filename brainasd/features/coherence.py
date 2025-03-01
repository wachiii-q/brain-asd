import numpy as np
import scipy.signal as signal
import itertools



class EEGCoherence:
    def __init__(self, data, fs, bands=None):
        """
        Initialize the EEGCoherence class.
        
        :param data: EEG data of shape (n_epochs, n_channels, n_samples)
        :param fs: Sampling frequency in Hz
        :param bands: Dictionary of frequency bands to compute coherence
        """
        self.data = data 
        self.fs = fs  
        self.bands = bands or {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta": (12, 30),
            "gamma": (30, 40)
        }
        self.channel_pairs = list(itertools.combinations(range(self.data.shape[1]), 2))


    def compute_coherence(self, sig1, sig2):
        """
        Compute coherence between two EEG signals.
        
        :param sig1: EEG signal from channel 1
        :param sig2: EEG signal from channel 2
        :return: Coherence values for defined frequency bands
        """
        f, Cxy = signal.coherence(sig1, sig2, fs=self.fs, nperseg=self.fs)
        band_coherence = {}
        for band, (low, high) in self.bands.items():
            band_coherence[band] = np.mean(Cxy[(f >= low) & (f < high)])
        return band_coherence


    def run(self):
        """
        Compute coherence for all channel pairs across all epochs.
        
        :return: Coherence results with shape (n_epochs, n_bands, n_pairs)
        """
        n_epochs, n_channels, _ = self.data.shape
        n_bands = len(self.bands)
        n_pairs = len(self.channel_pairs)
        results = np.zeros((n_epochs, n_bands, n_pairs)) 
        for epoch_idx, epoch in enumerate(self.data):
            for pair_idx, (ch1, ch2) in enumerate(self.channel_pairs):
                coherence_values = self.compute_coherence(epoch[ch1], epoch[ch2])
                results[epoch_idx, :, pair_idx] = list(coherence_values.values())
        return results



if __name__ == "__main__":
    fs = 250
    n_epochs, n_channels, n_samples = 1, 16, 1000
    eeg_data = np.random.randn(n_epochs, n_channels, n_samples)

    eeg_coherence = EEGCoherence(eeg_data, fs)
    coherence_results = eeg_coherence.run()
    
    print("Coherence results shape:", coherence_results.shape)  # Expected: (1, 5, 120)
    print("Sample coherence data:", coherence_results[0, :, 0])  # Coherence values for first channel pair
