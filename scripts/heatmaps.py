
warnings.filterwarnings("ignore")

def calculate_heatmap(input, model, last_conv_layer_name, out_layer_name, pred_index=None):
    grad_model = tensorflow.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.get_layer(out_layer_name).output]
    )
    input = tf.convert_to_tensor(input)
    
    with tf.GradientTape() as tape:
        tape.watch(input)
        last_conv_layer_output, preds = grad_model(input)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1))
  
    # Multiply each channel in the feature map array
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
  
    # Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
