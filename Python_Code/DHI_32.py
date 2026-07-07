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
file_path = 'DHSI_dataset_32.mat'
data = h5py.File(file_path, 'r')

print("Keys:", list(data.keys()))

result_save_path = './output_32/'
os.makedirs(result_save_path, exist_ok=True)

# ================= PARAMETERS =================
img_W = 32
img_H = 32
SR = 0.1
batch_size = 1
lr0 = 0.05
TV_strength = 1e-9
Steps = 1001

num_patterns = 32*32

A_real = np.array(data['patterns'])
y_real = np.array(data['measurements'])

A_real = np.transpose(A_real, (2, 1, 0))
y_real = y_real.reshape(-1) 


if num_patterns > A_real.shape[-1]:
    raise ValueError(f"num_patterns ({num_patterns}) > available patterns ({A_real.shape[-1]})")
# ================= SINGLE-PIXEL CORRELATION RECONSTRUCTION =================
print('Single-pixel correlation reconstruction...')

# Initialize reconstruction
recon = np.zeros((img_H, img_W), dtype=np.float64)

for i in range(num_patterns):
    pattern = A_real[:, :, i]
    measurement = y_real[i]

    recon += pattern * measurement

# Normalize (important!)
recon /= num_patterns

print('Single-pixel correlation finished')

# ================= TF GRAPH =================
with tf.compat.v1.variable_scope('input'):
    inpt = tf.compat.v1.placeholder(tf.float32, [batch_size, img_W, img_H, 1])
    y = tf.compat.v1.placeholder(tf.float32, [batch_size, 1, 1, num_patterns])
    A = tf.compat.v1.placeholder(tf.float32, [batch_size, img_W, img_H, num_patterns])
    isTrain = tf.compat.v1.placeholder(tf.bool)
    lr = tf.compat.v1.placeholder(tf.float32)

    global_step = tf.compat.v1.Variable(0, trainable=False)

    lrate = tf.compat.v1.train.exponential_decay(
        lr0, global_step, decay_steps=100, decay_rate=0.9, staircase=True
    )

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
    recon = recon.reshape(batch_size, img_W, img_H, 1)

    # Normalize
    recon = (recon - recon.mean()) / recon.std()
    y_real = (y_real - y_real.mean()) / y_real.std()
    A_real = (A_real - A_real.mean()) / A_real.std()

    recon_temp = recon[0, :, :, 0]
    y_real_temp = y_real.flatten()

    print('Single-pixel correlation reconstruction...')

    for step in range(Steps):
        if step % 100 == 0:
            train_loss = sess.run(
                loss_y,
                feed_dict={inpt: recon, y: y_real, A: A_real, isTrain: True}
            )
            lr_val = sess.run(lrate)
            print(f"step:{step}  y_loss:{train_loss:.6f}  lr:{lr_val:.6f}")

            y_out, x_out = sess.run(
                [y_pred, x_pred],
                feed_dict={inpt: recon, y: y_real, A: A_real, isTrain: False}
            )

            x_out = x_out.reshape(img_H, img_W)
            y_out = y_out.flatten()

            plt.figure(figsize=(12, 10))  # full-page figure

            # -------- Top-left: Initial Image (SP) --------
            ax1 = plt.subplot(2, 2, 1)
            im1 = ax1.imshow(recon_temp, cmap='gray')
            ax1.set_title('(a)', fontsize=12)
            ax1.set_xticks([])
            ax1.set_yticks([])
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
            # -------- Top-right: Refined Image (DNC) --------
            ax2 = plt.subplot(2, 2, 2)
            im2 = ax2.imshow(x_out, cmap='gray')
            ax2.set_title('(b)', fontsize=12)
            ax2.set_xticks([])
            ax2.set_yticks([])
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            # -------- Bottom-left: Predicted y --------
            ax3 = plt.subplot(2, 2, 3)
            ax3.plot(y_out, linewidth=1)
            ax3.set_title('(c)', fontsize=12)
            ax3.set_xlabel('Pattern Index')
            ax3.set_ylabel('Intensity')
            ax3.grid(True)
            
            # -------- Bottom-right: Real y --------
            ax4 = plt.subplot(2, 2, 4)
            ax4.plot(y_real_temp, linewidth=1)
            ax4.set_title('(d)', fontsize=12)
            ax4.set_xlabel('Pattern Index')
            ax4.set_ylabel('Intensity')
            ax4.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{result_save_path}/plot_{step}.png")
            plt.close()

            x_img = (x_out - x_out.min()) / x_out.max() * 255
            Image.fromarray(x_img.astype(np.uint8)).save(
                f"{result_save_path}/DNC_{num_patterns}_{step}.bmp"
            )

        sess.run(train_op, feed_dict={inpt: recon, y: y_real, A: A_real, isTrain: True})

print('Finished!')
