# -*- coding: utf-8 -*-
"""
TensorFlow 2.10 compatible version of GIDC_main(GI).py
Original author: Fei Wang (Jan 2022)
Adapted for TF 2.10 by preserving TF1-style graph execution
"""

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import GIDC_model_Unet
from PIL import Image
import os
import h5py

# ================= TF 2.10 COMPATIBILITY =================
tf.compat.v1.disable_eager_execution()
tf.random.set_seed(1234)
np.random.seed(1234)

# ================= LOAD DATA =================
file_path = 'GI_data(speckle)_1000_64.mat'
data = h5py.File(file_path, 'r')

print("Keys:", list(data.keys()))

result_save_path = './output_1000_64/'
os.makedirs(result_save_path, exist_ok=True)
# ================= PARAMETERS =================
img_W = 64
img_H = 64
SR = 0.1
batch_size = 1
lr0 = 0.05
TV_strength = 1e-9
Steps = 1001

num_patterns = 1000

A_real = np.array(data['patterns'])
y_real = np.array(data['measurements'])

# Fix dimensions
A_real = np.transpose(A_real, (2, 1, 0))
y_real = y_real.reshape(-1)               # (1,1000) → (1000,)


if num_patterns > A_real.shape[-1]:
    raise ValueError(f"num_patterns ({num_patterns}) > available patterns ({A_real.shape[-1]})")
# ================= CLASSICAL GHOST IMAGING RECONSTRUCTION =================
print('Classical Ghost Imaging reconstruction...')

# Compute mean values
measurements_mean = np.mean(y_real)                # mean of measurements
pattern_mean = np.mean(A_real, axis=2)            # mean of patterns along the 3rd dimension

# Initialize reconstruction
CGI = np.zeros((img_H, img_W), dtype=np.float64)

# Ghost imaging correlation reconstruction
for k in range(num_patterns):
    CGI += (y_real[k] - measurements_mean) * (A_real[:, :, k] - pattern_mean)

# Normalize
CGI /= num_patterns


print('CGI reconstruction finished')

# ================= TF GRAPH =================
with tf.compat.v1.variable_scope('input'):
    inpt = tf.compat.v1.placeholder(tf.float32, [batch_size, img_W, img_H, 1])
    y = tf.compat.v1.placeholder(tf.float32, [batch_size, 1, 1, num_patterns])
    A = tf.compat.v1.placeholder(tf.float32, [batch_size, img_W, img_H, num_patterns])
    isTrain = tf.compat.v1.placeholder(tf.bool)
    lr = tf.compat.v1.placeholder(tf.float32)

    global_step = tf.compat.v1.Variable(0, trainable=False)

    lrate = tf.compat.v1.train.exponential_decay(lr0, global_step, decay_steps=100, decay_rate=0.9, staircase=True)
    

# ================= MODEL =================
x_pred, y_pred = GIDC_model_Unet.inference(
    inpt, A, batch_size, img_W, img_H, num_patterns, isTrain
)

# ================= LOSS =================
TV_reg = TV_strength * tf.reduce_mean(
    tf.image.total_variation(x_pred)
)

loss_y = tf.reduce_mean(tf.square(y - y_pred))
loss = loss_y + TV_reg

# ================= OPTIMIZER =================
update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    train_op = tf.compat.v1.train.AdamOptimizer(
        learning_rate=lrate, beta1=0.5, beta2=0.9
    ).minimize(loss, global_step=global_step)

# ================= SESSION =================
init_op = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init_op)

    y_real = y_real.reshape(batch_size, 1, 1, num_patterns)
    A_real = A_real.reshape(batch_size, img_W, img_H, num_patterns)
    CGI = CGI.reshape(batch_size, img_W, img_H, 1)

    # Normalize
    CGI = (CGI - CGI.mean()) / CGI.std()
    y_real = (y_real - y_real.mean()) / y_real.std()
    A_real = (A_real - A_real.mean()) / A_real.std()

    CGI_temp = CGI[0, :, :, 0]
    y_real_temp = y_real.flatten()

    print('Classical Ghost Imaging reconstruction...')

    for step in range(Steps):
        if step % 100 == 0:
            train_loss = sess.run(
                loss_y,
                feed_dict={inpt: CGI, y: y_real, A: A_real, isTrain: True}
            )
            lr_val = sess.run(lrate)
            # print(f"step:{step}  y_loss:{train_loss:.6f}  lr:{lr_val:.6f}")
            print("step:{}  y_loss:{:.6f}  lr:{:.6f}".format(step, train_loss, lr_val))

            y_out, x_out = sess.run(
                [y_pred, x_pred],
                feed_dict={inpt: CGI, y: y_real, A: A_real, isTrain: False}
            )

            # -------- SAFE CONVERSION --------
            x_out = np.asarray(x_out, dtype=np.float64)
            y_out = np.asarray(y_out, dtype=np.float64)
            y_real_temp = np.asarray(y_real_temp, dtype=np.float64)
            CGI_temp = np.asarray(CGI_temp, dtype=np.float64)
            
            # Remove NaN / Inf (CRITICAL)
            x_out = np.nan_to_num(x_out)
            y_out = np.nan_to_num(y_out)
            y_real_temp = np.nan_to_num(y_real_temp)
            CGI_temp = np.nan_to_num(CGI_temp)
            
            # Reshape safely
            # x_out = x_out.reshape(img_W, img_H)
            x_out = x_out.reshape(img_H, img_W)
            # print("y_out shape:", y_out.shape)
            
            plt.figure(figsize=(12, 10))

            # ---- CGI ----
            ax1 = plt.subplot(2, 2, 1)
            im1 = ax1.imshow(CGI_temp, cmap='gray')
            ax1.set_title('(a)')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1)
            
            # ---- GIDC ----
            ax2 = plt.subplot(2, 2, 2)
            im2 = ax2.imshow(x_out, cmap='gray')
            ax2.set_title('(b)')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2)
            
            # ---- Predicted y ----
            ax3 = plt.subplot(2, 2, 3)
            y_out = y_out.reshape(-1)  # 🔥 FIX
            ax3.plot(np.arange(len(y_out)), y_out)
            ax3.set_title('(c)')
            
            # ---- Real y ----
            ax4 = plt.subplot(2, 2, 4)
            y_real_temp = y_real_temp.reshape(-1)  # 🔥 FIX
            ax4.plot(np.arange(len(y_real_temp)), y_real_temp)
            ax4.set_title('(d)')
            
            plt.tight_layout()
            plt.savefig(f"{result_save_path}/plot_{step}.png")
            plt.close()
            
            
            # -------- SAVE IMAGE --------
            x_img = (x_out - x_out.min()) / (x_out.max() - x_out.min() + 1e-8) * 255
            x_img = np.nan_to_num(x_img)
            
            Image.fromarray(x_img.astype(np.uint8)).save(
                f"{result_save_path}/GIDC_{num_patterns}_{step}.bmp"
            )

        sess.run(train_op, feed_dict={inpt: CGI, y: y_real, A: A_real, isTrain: True})

print('Finished!')
