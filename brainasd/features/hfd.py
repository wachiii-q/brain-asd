import numpy as np



class EEGHiguchiFractalDimension:
    def __init__(self, data, k_max=10):
        """
        Initialize the HFD computation.
        
        :param data: EEG data of shape (n_epochs, n_channels, n_samples)
        :param k_max: Maximum k value for the HFD algorithm (controls scale)
        """
        self.data = data  
        self.k_max = k_max  


    def compute_hfd(self, signal):
        """
        Compute Higuchi Fractal Dimension (HFD) for a single EEG signal.
        
        :param signal: 1D EEG signal (n_samples,)
        :return: HFD value
        """
        N = len(signal)
        L = np.zeros(self.k_max)
        k_values = np.arange(1, self.k_max + 1)
        for k in k_values:
            Lk = np.zeros(k)
            for m in range(k):
                idx = np.arange(m, N, k)  
                if len(idx) > 1:
                    Lm = np.sum(np.abs(np.diff(signal[idx])))
                    Lm /= (len(idx) * (N / k))  # Normalize length
                    Lk[m] = Lm
            L[k - 1] = np.mean(Lk)
        log_k = np.log(1.0 / k_values)
        log_L = np.log(L)
        HFD, _ = np.polyfit(log_k, log_L, 1)
        return HFD


    def run(self):
        """
        Compute HFD for all EEG channels across all epochs.
        
        :return: HFD results with shape (n_epochs, n_channels)
        """
        n_epochs, n_channels, _ = self.data.shape
        results = np.zeros((n_epochs, n_channels)) 
        for epoch_idx, epoch in enumerate(self.data):
            for ch_idx, signal in enumerate(epoch):
                results[epoch_idx, ch_idx] = self.compute_hfd(signal)
        return results  # Shape: (n_epochs, n_channels)



if __name__ == "__main__":
    n_epochs, n_channels, n_samples = 1, 16, 1000
    eeg_data = np.random.randn(n_epochs, n_channels, n_samples)

    # Create HFD analysis object and run
    hfd_computer = HiguchiFractalDimension(eeg_data, k_max=10)
    hfd_results = hfd_computer.run()
    
    print("HFD results shape:", hfd_results.shape)  # Expected: (1, 16)
    print("Sample HFD data:", hfd_results[0, :])  # HFD values for all channels in first epoch