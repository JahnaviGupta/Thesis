import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import GIDC_model_Unet
from PIL import Image
import os

# Ensure compatibility with TensorFlow 2.x
tf.compat.v1.disable_eager_execution()

# --- MODIFICATIONS START HERE ---

# Define optimization parameters early so they are accessible
img_W = 256
img_H = 256
batch_size = 1
lr0 = 0.005                                   # learning rate
TV_strength = 1e-9                            # regularization parameter of Total Variation
Steps = 1001                                  # optimization steps

# Define the phases and their corresponding data files and folders
phases = [
    ('linked_0.mat', 'pattern_360/pattern_cir_0', 'phase0'),
    ('linked_2pibi3.mat', 'pattern_360/pattern_cir_2pibi3', 'phase2pibi3'),
    ('linked_4pibi3.mat', 'pattern_360/pattern_cir_4pibi3', 'phase4pibi3')
]

A_real_combined = []
y_real_combined = []

for mat_file, pattern_folder, tag in phases:
    print(f"\n📂 Loading data from {mat_file}")
    data = loadmat(mat_file)
    
    # Assuming the variable names are 'intensities' for measurements
    # and 'pattern_paths' for the pattern file paths.
    measurements = data['A'].squeeze()
    paths = data['pattern_paths'].squeeze()

    patterns = []
    for p in paths:
        # Handling different data types for paths in the .mat file
        path = str(p[0] if isinstance(p[0], str) else p[0][0])
        full_path = os.path.join(pattern_folder, os.path.basename(path))
       
        p = 'pattern_360/' + p
        p = p.strip()              
        p = os.path.normpath(p)  
        print(f"🖼️  Loading pattern: {p}")
        img = Image.open(p).convert('L')
        img = img.resize((img_W, img_H))
        img_np = np.array(img, dtype=np.float32) / 255.0
        patterns.append(img_np)

    patterns = np.stack(patterns, axis=-1)  # shape (H, W, N)
    A_real_combined.append(patterns)
    y_real_combined.append(measurements)

# Combine all patterns and measurements
A_real = np.concatenate(A_real_combined, axis=-1)
y_real = np.concatenate(y_real_combined, axis=0)


num_patterns = np.shape(A_real)[-1]  # Total number of patterns loaded

# --- MODIFICATIONS END HERE ---

result_save_path = './r/'

# create results save path
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)

# --- MODIFICATION: Using a pre-loaded image as initial input (as per your previous request) ---
print('Using sample_input.png as the initial input for GIDC...')
try:
    # img_path = '21 Feb, 2025/hgmSTAR, static ground glass, 3phase shift/cxyabs, IC.bmp'
    img_path = 'sample_star_two.png'
    extracted_star_img = Image.open(img_path).convert('L')
    extracted_star_img = extracted_star_img.resize((img_W, img_H))
    
    #GI = np.array(extracted_star_img)
    GI = np.array(extracted_star_img, dtype=np.float32) / 255.0
    GI = np.reshape(GI, (img_W, img_H))
    # ... after GI = np.reshape(GI, (img_W, img_H))
    temp_img = Image.fromarray(GI.astype('uint8')).convert('L')
    temp_img.save('initial_GI.bmp')
    # ...
    print('Finished')
    
except FileNotFoundError:
    print(f"Error: The file {img_path} was not found.")
    print("Please make sure sample_input.jpg is in the same directory as the script.")
    exit()

# --- MODIFICATION ENDS HERE ---


with tf.compat.v1.variable_scope('input'):           
    inpt = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, img_W, img_H, 1], name='inpt')
    y = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, 1, 1, num_patterns], name='y') 
    A = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, img_W, img_H, num_patterns], name='A')                
    x = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, img_W, img_H, 1], name='x')
                
    isTrain = tf.compat.v1.placeholder(tf.bool, name='isTrain')
    lr = tf.compat.v1.placeholder(tf.float32, name='learning_rate')
    groable = tf.Variable(tf.constant(0))
    lrate = tf.compat.v1.train.exponential_decay(lr0, groable, 100, 0.90)


x_pred, y_pred = GIDC_model_Unet.inference(inpt, A, batch_size, img_W, img_H, num_patterns)

# define the loss function
TV_reg = TV_strength * tf.image.total_variation(tf.reshape(x_pred, [batch_size, img_W, img_H, 1]))
loss_y = tf.reduce_mean(tf.square(y - y_pred))
loss = loss_y + TV_reg

update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
with tf.compat.v1.variable_scope('train_step'):
    with tf.control_dependencies(update_ops):
        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9, epsilon=1e-08).minimize(loss)

init_op = (tf.compat.v1.local_variables_initializer(), tf.compat.v1.global_variables_initializer())

with tf.compat.v1.Session() as sess:
    sess.run(init_op)
    y_real = np.reshape(y_real, [batch_size, 1, 1, num_patterns])
    A_real = np.reshape(A_real, [batch_size, img_W, img_H, num_patterns])
    GI = np.reshape(GI, [batch_size, img_W, img_H, 1])

    # Re-adding the normalization lines you removed
    # GI = (GI - np.mean(GI)) / np.std(GI)
    # y_real = (y_real - np.mean(y_real)) / np.std(y_real)
    # A_real = (A_real - np.mean(A_real)) / np.std(A_real)
    GI = (GI - np.mean(GI)) / (np.std(GI) + 1e-8)
    y_real = (y_real - np.mean(y_real)) / (np.std(y_real) + 1e-8)
    
    GI_temp0 = np.reshape(GI, [img_W, img_H], order='F')
    GI_temp = np.transpose(GI_temp0)
    y_real_temp = np.reshape(y_real, [num_patterns])
    inpt_temp = GI

    print('Single-pixel correlation(Three Phase) reconstruction...')

    for step in range(Steps):
        lr_temp = sess.run(lrate, feed_dict={groable: step})

        if step % 100 == 0:
            train_y_loss = sess.run(loss_y, feed_dict={inpt: inpt_temp, y: y_real, A: A_real, isTrain: True, lr: lr_temp})
            print('step:%d----y loss:%f----learning rate:%f----num of patterns:%d' % (step, train_y_loss, lr_temp, num_patterns))

            [y_out, x_out] = sess.run([y_pred, x_pred], feed_dict={inpt: inpt_temp, y: y_real, A: A_real, isTrain: True, lr: lr_temp})
            x_out = np.reshape(x_out, [img_W, img_H], order='F')
            y_out = np.reshape(y_out, [num_patterns], order='F')

        
            plt.figure(figsize=(12, 10))  # full-page figure

            # -------- Top-left: Initial Image (GI) --------
            ax1 = plt.subplot(2, 2, 1)
            im1 = ax1.imshow(GI_temp0)
            ax1.set_title('Single Pixel(Three)', fontsize=12)
            ax1.set_xticks([])
            ax1.set_yticks([])
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
            # -------- Top-right: Refined Image (GIDC) --------
            ax2 = plt.subplot(2, 2, 2)
            im2 = ax2.imshow(x_out)
            ax2.set_title('DNC', fontsize=12)
            ax2.set_xticks([])
            ax2.set_yticks([])
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            # -------- Bottom-left: Predicted y --------
            ax3 = plt.subplot(2, 2, 3)
            ax3.plot(y_out, linewidth=1)
            ax3.set_title('Predicted y', fontsize=12)
            ax3.set_xlabel('Pattern Index')
            ax3.set_ylabel('Intensity')
            ax3.grid(True)
            
            # -------- Bottom-right: Real y --------
            ax4 = plt.subplot(2, 2, 4)
            ax4.plot(y_real_temp, linewidth=1)
            ax4.set_title('Real y', fontsize=12)
            ax4.set_xlabel('Pattern Index')
            ax4.set_ylabel('Intensity')
            ax4.grid(True)
            
            plt.tight_layout()
            plt.show()

            x_out = x_out - np.min(x_out)
            x_out = x_out * 255 / np.max(np.max(x_out))
            x_out = Image.fromarray(x_out.astype('uint8')).convert('L')
            x_out.save(result_save_path + 'GIDC_%d_%d.bmp' % (num_patterns, step))

        # optimize the weights in the DNN
        sess.run([train_op], feed_dict={inpt: inpt_temp, y: y_real, A: A_real, isTrain: True, lr: lr_temp})

print('Finished!')