#return an array that feedable to the grad cam
def get_input_array(X, index=1):
    input = X[index]
    input = input.expand_dims(input, axis=0)
    return input

def visualize_input_signal(input):
    plt.figure(figsize=(12,2))
    plt.plot(input)
    plt.title("PPG Signal")
    plt.xlabel("time steps")
    plt.grid(True)
    
def visualize_heatmap(heatmap):
    plt.figure(figsize=(12,2))
    plt.plot(heatmap)
    plt.title("Weighted sum of Feature maps")
    plt.grid(True)
    
def get_stats(input_data):
    means = np.mean(input_data, axis=0)
    vars = np.std(input_data, axis=0)
    return means, vars

def denormalize(arr, mean, std):
  return (arr*std) + mean

def normalize_values(arr):
  mean = np.mean(arr)
  std = np.std(arr)
  arr_normalized = (arr - mean)/std
  return arr_normalized
