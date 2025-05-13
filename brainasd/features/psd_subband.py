import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


class EEGSubPowerSpectrum:
    def __init__(self, data, fs, ifNormalize=True):
        """        
        :param data: EEG data of shape (n_epochs, n_channels, n_samples)
        :param fs: Sampling frequency in Hz
        :param ifNormalize: Normalize band power by total power
        """
        self.data = data
        self.fs = fs
        self.freqs = None
        self.psd = None
        self.alpha1 = None
        self.alpha2 = None
        self.beta1 = None
        self.beta2 = None
        self.beta3 = None
        self.eeg = None
        self.isNormalized = ifNormalize
    
    def compute_power_spectrum(self, data):
        self.freqs, self.psd = signal.welch(data, fs=self.fs, nperseg=self.fs, noverlap=self.fs / 2)
    
    def compute_band_power(self):
        self.alpha1 = self.psd[(self.freqs >= 8) & (self.freqs < 10)].sum()
        self.alpha2 = self.psd[(self.freqs >= 10) & (self.freqs < 12)].sum()
        self.beta1 = self.psd[(self.freqs >= 13) & (self.freqs < 17)].sum()
        self.beta2 = self.psd[(self.freqs >= 17) & (self.freqs < 25)].sum()
        self.beta3 = self.psd[(self.freqs >= 25) & (self.freqs < 30)].sum()
        self.eeg = self.psd[(self.freqs >= 8) & (self.freqs < 30)].sum()  # Total power in the subband range
        
        if self.isNormalized:
            self.alpha1 = self.alpha1 / self.eeg
            self.alpha2 = self.alpha2 / self.eeg
            self.beta1 = self.beta1 / self.eeg
            self.beta2 = self.beta2 / self.eeg
            self.beta3 = self.beta3 / self.eeg

    def run(self):
        # expected input.shape = (n_epochs, n_channels, n_samples)
        results = []
        for epoch in self.data:
            eachEpoch = []
            for channel in epoch:
                eegData = channel
                self.compute_power_spectrum(eegData)
                self.compute_band_power()
                eachEpoch.append([self.alpha1, self.alpha2, self.beta1, self.beta2, self.beta3])
            results.append(eachEpoch)
        results = np.array(results)
        results = np.transpose(results, (0, 2, 1))    
        return np.array(results)


if __name__ == '__main__':
    # Example usage
    sinWave = np.sin(2 * np.pi * 27 * np.linspace(0, 1, 1000))
    fs = 1000
    data = np.array([sinWave])
    data = np.expand_dims(data, axis=0)
    
    eegPowerSpectrum = EEGSubPowerSpectrum(data, fs)
    results = eegPowerSpectrum.run()    
    print(results)
    print(results.shape)
    
    # Plot power spectrum   
    plt.figure()
    plt.plot(eegPowerSpectrum.freqs, eegPowerSpectrum.psd)
    plt.title('Power spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.show()
    
    # Scatter plot normalized subband power
    plt.figure()
    plt.scatter(np.arange(5), results[0, 0, :])
    plt.title('Normalized subband power')
    plt.xlabel('Subband')
    plt.ylabel('Power')
    plt.show()