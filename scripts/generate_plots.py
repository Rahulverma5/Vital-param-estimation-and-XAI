import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings("ignore")

def plot_and_save_gradcam(ppg_signal, gradients, ax, title, pred_val, true_val=None):
    # Ensure input arrays are 1D
    ppg_signal = ppg_signal.flatten()
    gradients = gradients.flatten()

    # Interpolate the gradients to match the length of the PPG signal
    # x_original = np.linspace(0, 1, len(gradients))
    # x_resampled = np.linspace(0, 1, len(ppg_signal))

    # interpolator = interp1d(x_original, gradients, kind='linear')
    # gradients_resampled = interpolator(x_resampled)
    
    #upsampling
    scale_factor = len(ppg_signal) / len(gradients)
    gradients_resampled = zoom(gradients, scale_factor, order=2)

    # Plot the PPG signal
    ax.plot(ppg_signal)
    #ax.plot(gradients_resampled[0:400], alpha=0.7)

    # Superimpose the heatmap
    ax.imshow(np.expand_dims(gradients_resampled, axis=0), aspect='auto', 
              extent=[0, len(ppg_signal), np.min(ppg_signal), np.max(ppg_signal)], 
              cmap='jet', alpha=0.3)
    
    
    ax.text(1.0, 1.0, f"Predicted: {pred_val[0]:.2f}", ha="right", va="top", transform=ax.transAxes)
    ax.text(1.0, 0.8, f"True: {true_val:.2f}", ha="right", va="top", transform=ax.transAxes)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title(f"visualization for {title}")

def visualize_vital_params(input, model, path, denorm_pred, true_val=None):
    fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Grad-CAM Visualizations")

    params = [
        ('HR', "conv1d_1", "out_hr", denorm_pred[0][0], true_val[0]),
        ('RR', "conv1d_1", "out_rr", denorm_pred[1][0], true_val[1]),
        ('SpO2', "conv1d_11", "out_spo2", denorm_pred[2][0], true_val[2]),
        ('SBP', "conv1d_7", "out_sbp", denorm_pred[3][0],true_val[3]),
        ('DBP', "conv1d_7", "out_dbp", denorm_pred[4][0], true_val[4]),
    ]
    
    for ax, (title, last_conv_layer, out_layer, pred_val, true_v) in zip(axs, params):
        heatmap = calculate_heatmap(input, model, last_conv_layer, out_layer)
        plot_and_save_gradcam(input, heatmap, ax, title, pred_val, true_v)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    fig.savefig(path)
    print(f"Combined plot saved..")
