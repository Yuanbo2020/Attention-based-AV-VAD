def vis_cam(model, audio_x, videos_x, prefix):
                intermediate_layer_model1= Model(inputs=model.input, 	outputs=model.get_layer("video_final_out").output)
                model = intermediate_layer_model1 
                loss = K.sum(model.output)
                conv_output = model.get_layer("re_lu_13").output
                grads = normalize(tf.gradients(loss, [conv_output]))
                # grads = tf.gradients(loss, [conv_output])
                gradient_function = K.function([model.input], [conv_output, grads])
                output, grads_val = gradient_function([audio_x, videos_x])
                output, grads_val = output[21].mean(-1), grads_val[0][21].mean(-1)
                for out_idx, heatmap in enumerate([output, grads_val, output*grads_val]):
                    heatmap = heatmap / heatmap.max()
                    heatmap = np.uint8(255 * heatmap)
                    import matplotlib.cm as cm
                    from PIL import Image

                    jet = cm.get_cmap("jet")
                    jet_colors = jet(np.arange(256))[:, :3]
                    jet_heatmap = jet_colors[heatmap]
                    # jet_heatmap_img = Image.fromarray(jet_heatmap, "RGB").resize(videos_x[0].shape[:-1][::-1]).rotate(180)
                    jet_heatmap_img = Image.fromarray(jet_heatmap, "RGB").resize(videos_x[0].shape[:-1][::-1])
                    video_frame = np.uint8(255 * videos_x[21])
                    superimposed_arr = np.array(jet_heatmap_img) * 0.8 + video_frame
                    superimposed_arr = superimposed_arr / superimposed_arr.max()
                    superimposed_arr = np.uint8(255 * superimposed_arr)
                    from scipy.misc import imsave
                    imsave(f"{prefix}_{out_idx}.png", superimposed_arr)


weights = np.mean(grads_val, axis = (0, 1))
cam = np.ones(output.shape[0 : 2], dtype = np.float32)
