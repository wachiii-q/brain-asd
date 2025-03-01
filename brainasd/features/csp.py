import numpy as np
from scipy.linalg import eigh



class CSP:
    def __init__(self, n_components=4):
        """
        Initialize the CSP class.
        
        :param n_components: Number of spatial filters to keep per class
        """
        self.n_components = n_components
        self.filters = None


    def fit(self, X, y):
        """
        Compute CSP filters from EEG data.
        
        :param X: EEG data (n_epochs, n_channels, n_samples)
        :param y: Labels (0 or 1) for each epoch
        """
        class_0 = X[y == 0]  # EEG trials for class 0
        class_1 = X[y == 1]  # EEG trials for class 1
        cov_0 = np.mean([np.cov(epoch) for epoch in class_0], axis=0)
        cov_1 = np.mean([np.cov(epoch) for epoch in class_1], axis=0)
        eigenvalues, eigenvectors = eigh(cov_1, cov_0 + cov_1)
        top_components = eigenvectors[:, -self.n_components:]  # Largest eigenvalues
        bottom_components = eigenvectors[:, :self.n_components]  # Smallest eigenvalues
        self.filters = np.hstack([top_components, bottom_components])  # CSP filters


    def transform(self, X):
        """
        Apply CSP filters to EEG data and extract log-variance features.
        
        :param X: EEG data (n_epochs, n_channels, n_samples)
        :return: CSP features (n_epochs, 2 * n_components)
        """
        if self.filters is None:
            raise ValueError("CSP filters not computed. Call fit() first.")
        X_csp = np.array([self.filters.T @ epoch for epoch in X])  # Shape: (n_epochs, n_filters, n_samples)
        var_features = np.log(np.var(X_csp, axis=2))  # Shape: (n_epochs, 2 * n_components)
        return var_features



if __name__ == "__main__":
    # Simulated EEG data: (n_epochs=100, n_channels=16, n_samples=256)
    np.random.seed(42)
    n_epochs, n_channels, n_samples = 100, 16, 256
    eeg_data = np.random.randn(n_epochs, n_channels, n_samples)

    # Simulated binary labels (50 trials of each class)
    labels = np.array([0] * 50 + [1] * 50)

    # CSP Feature Extraction
    csp = CSP(n_components=4)
    csp.fit(eeg_data, labels)
    csp_features = csp.transform(eeg_data)

    print("CSP feature shape:", csp_features.shape)  # Expected: (100, 8)
    print("Sample CSP features:", csp_features[:5])  # First 5 trials
