import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import mne
from collections import deque

sampling_rate = 256
which_save_asd_dir = "/Users/wachiii/Workschii/brain-asd/data/data_children_trimed_5min_3535/trimedData/asd"
which_save_hc_dir = "/Users/wachiii/Workschii/brain-asd/data/data_children_trimed_5min_3535/trimedData/hc"

class EEGTrimmer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EEG Trimmer")
        self.geometry("12000x900")
        self.file_path = None
        self.signals = None
        self.channel = 0
        self.undo_stack = deque(maxlen=5)

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

        trim_frame = tk.Frame(self)
        trim_frame.pack(pady=5)
        tk.Label(trim_frame, text="Trim-Out Start (sec):").pack(side=tk.LEFT)
        self.trim_start_var = tk.DoubleVar(value=0)
        self.trim_start_slider = tk.Scale(
            trim_frame,
            from_=0,
            to=1,
            orient=tk.HORIZONTAL,
            variable=self.trim_start_var,
            command=self.update_plot,
            length=800,      
            resolution=0.1      
        )
        self.trim_start_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(trim_frame, text="Length (sec):").pack(side=tk.LEFT)
        self.trim_len_var = tk.IntVar(value=1)
        self.trim_len_spin = tk.Spinbox(trim_frame, from_=1, to=30, increment=1, textvariable=self.trim_len_var, width=3)
        self.trim_len_spin.pack(side=tk.LEFT)
        tk.Button(trim_frame, text="Trim Out", command=self.trim_out).pack(side=tk.LEFT, padx=5)
        tk.Button(trim_frame, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=5)

        self.fig, self.ax = plt.subplots(figsize=(10,3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.save_full_var = tk.BooleanVar(value=False)
        self.save_full_check = tk.Checkbutton(
            self, text="Save Full File (after artifact removal)", variable=self.save_full_var
        )
        self.save_full_check.pack()

        tk.Button(self, text="Save Segment", command=self.save_segment).pack()
        
        yaxis_frame = tk.Frame(self)
        yaxis_frame.pack(pady=5)
        self.auto_scale_var = tk.BooleanVar(value=True)
        self.auto_scale_check = tk.Checkbutton(
            yaxis_frame, text="Auto Y-Axis", variable=self.auto_scale_var, command=self.update_plot
        )
        self.auto_scale_check.pack(side=tk.LEFT)
        tk.Label(yaxis_frame, text="Ymin:").pack(side=tk.LEFT)
        self.ymin_entry = tk.Entry(yaxis_frame, width=6)
        self.ymin_entry.insert(0, "-100")
        self.ymin_entry.pack(side=tk.LEFT)
        tk.Label(yaxis_frame, text="Ymax:").pack(side=tk.LEFT)
        self.ymax_entry = tk.Entry(yaxis_frame, width=6)
        self.ymax_entry.insert(0, "100")
        self.ymax_entry.pack(side=tk.LEFT)


    def get_window_samples(self):
        return int(self.window_minutes_var.get() * 60 * sampling_rate)


    def update_slider(self):
        if self.signals is not None:
            length = self.signals.shape[1] if self.signals.ndim > 1 else self.signals.shape[0]
            window_samples = self.get_window_samples()
            max_start = max(0, length - window_samples)
            self.slider.config(to=max_start // sampling_rate)
            self.slider.set(0)
            self.trim_start_slider.config(to=(length // sampling_rate) - 1)
            self.trim_start_var.set(0)
            self.update_plot()


    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
        if not path:
            return
        self.file_path = path
        self.signals = np.load(path, allow_pickle=True)
        if isinstance(self.signals, list):
            self.signals = np.array(self.signals)
        self.undo_stack.clear()
        self.undo_stack.append(self.signals.copy())
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
        if self.signals.ndim > 1:
            data = self.signals[self.channel, start:end]
            ch_names = [f"Ch{self.channel}"]
        else:
            data = self.signals[start:end]
            ch_names = ["Ch0"]
        info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types="eeg")
        raw = mne.io.RawArray(data[np.newaxis, :], info, verbose=False)
        raw.notch_filter(50, verbose=False)
        plot_data = raw.get_data().flatten()
        self.ax.plot(np.arange(start, start+len(plot_data))/sampling_rate, plot_data)
        trim_start = self.trim_start_var.get()
        trim_len = self.trim_len_var.get()
        trim_end = trim_start + trim_len
        self.ax.axvspan(trim_start, trim_end, color='red', alpha=0.3, label='Trim-Out')
        self.ax.set_title(f"Channel {self.channel} | {start_sec}s to {start_sec+self.window_minutes_var.get()*60}s (notch 50Hz only)")
        self.ax.set_xlabel("Time (s)")
        self.ax.legend()
        if not self.auto_scale_var.get():
            try:
                ymin = float(self.ymin_entry.get())
                ymax = float(self.ymax_entry.get())
                self.ax.set_ylim(ymin, ymax)
            except Exception:
                pass 
        self.canvas.draw()


    def trim_out(self):
        if self.signals is None:
            return
        trim_start = self.trim_start_var.get()
        trim_len = self.trim_len_var.get()
        start_idx = int(trim_start * sampling_rate)
        end_idx = int((trim_start + trim_len) * sampling_rate)
        self.undo_stack.append(self.signals.copy())
        if self.signals.ndim > 1:
            self.signals = np.concatenate([self.signals[:, :start_idx], self.signals[:, end_idx:]], axis=1)
        else:
            self.signals = np.concatenate([self.signals[:start_idx], self.signals[end_idx:]])
        self.update_slider()


    def undo(self):
        if len(self.undo_stack) > 1:
            self.undo_stack.pop()
            self.signals = self.undo_stack[-1].copy()
            self.update_slider()
        else:
            messagebox.showinfo("Undo", "No more undo steps available.")


    def save_segment(self):
        if self.signals is None:
            messagebox.showerror("Error", "No file loaded.")
            return
        if self.save_full_var.get():
            trimmed = self.signals
        else:
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