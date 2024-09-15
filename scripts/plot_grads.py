import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings("ignore")

def plot_and_save_gradcam(ppg_signal, gradients, ax, title, pred_val):
    # Ensure input arrays are 1D
    ppg_signal = ppg_signal.flatten()
    gradients = gradients.flatten()
    
    #upsampling
    scale_factor = len(ppg_signal) / len(gradients)
    gradients_resampled = zoom(gradients, scale_factor, order=3)
    
    # Plot the PPG signal
    ax.plot(ppg_signal)
    ax.plot(gradients_resampled, alpha=0.7)

    # Superimpose the heatmap
    ax.imshow(np.expand_dims(gradients_resampled, axis=0), aspect='auto', 
              extent=[0, len(ppg_signal), np.min(ppg_signal), np.max(ppg_signal)], 
              cmap='jet', alpha=0.3)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title(f"visualization for {title}")

def visualize_vital_params(input, model, path, denorm_pred):
    fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Grad-CAM Visualizations")

    params = [
        ('HR', "conv1d_1", "out_hr", denorm_pred[0][0]),
        ('RR', "conv1d_1", "out_rr", denorm_pred[1][0]),
        ('SpO2', "conv1d_11", "out_spo2", denorm_pred[2][0]),
        ('SBP', "conv1d_7", "out_sbp", denorm_pred[3][0]),
        ('DBP', "conv1d_7", "out_dbp", denorm_pred[4][0]),
    ]
    
    for ax, (title, last_conv_layer, out_layer, pred_val) in zip(axs, params):
        heatmap = calculate_heatmap(input, model, last_conv_layer, out_layer)
        plot_and_save_gradcam(input, heatmap, ax, title, pred_val)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    fig.savefig(path)
    print(f"Combined plot saved..")
