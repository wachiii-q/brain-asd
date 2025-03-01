import numpy as np
from scipy.spatial.distance import pdist



class EEGCorrelationDimension:
    def __init__(self, data, r_min=0.01, r_max=2.0, num_r=50):
        """
        Initialize the Correlation Dimension computation.
        
        :param data: EEG data of shape (n_epochs, n_channels, n_samples)
        :param r_min: Minimum radius for correlation sum
        :param r_max: Maximum radius for correlation sum
        :param num_r: Number of radius values to test
        """
        self.data = data  
        self.r_values = np.logspace(np.log10(r_min), np.log10(r_max), num_r)


    def compute_d2(self, signal):
        """
        Compute Correlation Dimension (D2) for a single EEG signal.
        
        :param signal: 1D EEG signal (n_samples,)
        :return: Estimated correlation dimension D2
        """
        N = len(signal)
        distances = pdist(signal.reshape(-1, 1), metric='euclidean')
        C_r = np.zeros(len(self.r_values))
        for i, r in enumerate(self.r_values):
            C_r[i] = np.sum(distances < r) / (N * (N - 1) / 2)
        log_r = np.log(self.r_values)
        log_C_r = np.log(C_r + 1e-10)
        D2, _ = np.polyfit(log_r, log_C_r, 1)
        return D2


    def run(self):
        """
        Compute D2 for all EEG channels across all epochs.
        
        :return: D2 results with shape (n_epochs, n_channels)
        """
        n_epochs, n_channels, _ = self.data.shape
        results = np.zeros((n_epochs, n_channels))
        for epoch_idx, epoch in enumerate(self.data):
            for ch_idx, signal in enumerate(epoch):
                results[epoch_idx, ch_idx] = self.compute_d2(signal)
        return results  



if __name__ == "__main__":
    n_epochs, n_channels, n_samples = 1, 16, 1000
    eeg_data = np.random.randn(n_epochs, n_channels, n_samples)

    d2_computer = CorrelationDimension(eeg_data)
    d2_results = d2_computer.run()

    print("D2 results shape:", d2_results.shape)  # Expected: (1, 16)
    print("Sample D2 data:", d2_results[0, :])  # D2 values for all channels in first epoch
