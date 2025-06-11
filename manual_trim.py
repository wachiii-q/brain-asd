import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import mne

sampling_rate = 256
window_minutes = 5
window_samples = window_minutes * 60 * sampling_rate
which_save_asd_dir = "/Users/wachiii/Workschii/brain-asd/data/data_children_trimed_5min_3535/trimedData/asd"
which_save_hc_dir = "/Users/wachiii/Workschii/brain-asd/data/data_children_trimed_5min_3535/trimedData/hc"


class EEGTrimmer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EEG Trimmer")
        self.geometry("900x650")
        self.file_path = None
        self.signals = None
        self.channel = 0
        self.start = 0

        # Use DoubleVar for fractional minutes
        self.window_minutes_var = tk.DoubleVar(value=5.0)
        tk.Label(self, text="Window Length (minutes):").pack()
        self.window_minutes_spin = tk.Spinbox(
            self, from_=0.5, to=30, increment=0.5, textvariable=self.window_minutes_var, width=5, command=self.update_slider
        )
        self.window_minutes_spin.pack()

        tk.Button(self, text="Open .npy File", command=self.open_file).pack()
        self.channel_var = tk.IntVar(value=0)
        self.channel_menu = tk.OptionMenu(self, self.channel_var, ())
        self.channel_menu.pack()
        self.channel_var.trace('w', self.change_channel)

        self.slider = tk.Scale(self, from_=0, to=1, orient=tk.HORIZONTAL, label="Start Time (seconds)", command=self.update_plot)
        self.slider.pack(fill=tk.X)

        self.fig, self.ax = plt.subplots(figsize=(8,3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tk.Button(self, text="Save Segment", command=self.save_segment).pack()

    def get_window_samples(self):
        return int(self.window_minutes_var.get() * 60 * sampling_rate)


    def update_slider(self):
        if self.signals is not None:
            length = self.signals.shape[1] if self.signals.ndim > 1 else self.signals.shape[0]
            window_samples = self.get_window_samples()
            max_start = max(0, length - window_samples)
            self.slider.config(to=max_start // sampling_rate)
            self.slider.set(0)
            self.update_plot()

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
        if not path:
            return
        self.file_path = path
        self.signals = np.load(path, allow_pickle=True)
        if isinstance(self.signals, list):
            self.signals = np.array(self.signals)
        n_channels = self.signals.shape[0] if self.signals.ndim > 1 else 1
        self.channel_var.set(0)
        menu = self.channel_menu["menu"]
        menu.delete(0, "end")
        for i in range(n_channels):
            menu.add_command(label=f"Channel {i}", command=tk._setit(self.channel_var, i))
        self.update_slider()

    def change_channel(self, *args):
        self.channel = self.channel_var.get()
        self.update_plot()

    def update_plot(self, *args):
        if self.signals is None:
            return
        window_samples = self.get_window_samples()
        start_sec = self.slider.get()
        start = start_sec * sampling_rate
        end = start + window_samples
        self.ax.clear()
        # --- Preprocessing for plot using MNE ---
        if self.signals.ndim > 1:
            data = self.signals[:, start:end]
        else:
            data = self.signals[np.newaxis, start:end]
        n_channels = data.shape[0]
        ch_names = [f"Ch{i}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        raw.filter(0.4, 40, fir_design='firwin', verbose=False)
        raw.notch_filter(50, verbose=False)
        plot_data = raw.get_data(picks=self.channel).flatten()
        self.ax.plot(plot_data)
        self.ax.set_title(f"Channel {self.channel} | {start_sec}s to {start_sec+self.window_minutes_var.get()*60}s (preprocessed for plot)")
        self.canvas.draw()

    def save_segment(self):
        if self.signals is None:
            messagebox.showerror("Error", "No file loaded.")
            return
        window_samples = self.get_window_samples()
        start_sec = self.slider.get()
        start = start_sec * sampling_rate
        end = start + window_samples
        if self.signals.ndim > 1:
            trimmed = self.signals[:, start:end]
        else:
            trimmed = self.signals[start:end]
        out_name = os.path.basename(self.file_path)
        if out_name.endswith("_ASD.npy"):
            save_dir = which_save_asd_dir
        elif out_name.endswith("_HC.npy"):
            save_dir = which_save_hc_dir
        else:
            save_dir = filedialog.askdirectory(title="Select Output Directory")
            if not save_dir:
                return
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, out_name)
        np.save(out_path, trimmed)
        messagebox.showinfo("Saved", f"Saved trimmed file to:\n{out_path}")

if __name__ == "__main__":
    app = EEGTrimmer()
    app.mainloop()