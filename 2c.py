import numpy as np
import matplotlib.pyplot as plt
import functions as fn  
from scipy.signal import medfilt  
import os

N = 50
sig = np.zeros(N)
sig[10:20] = 1.0

M = 15
h = np.array(fn.hamming(M)) 
h = h / np.sum(h) 

conv_time = np.convolve(sig, h, mode='full')

L = len(sig) + len(h) - 1
SIG_fft = fn.my_fft(sig, n=L)
H_fft   = fn.my_fft(h, n=L)

Y_fft = SIG_fft * H_fft

conv_freq = np.real(fn.my_ifft(Y_fft, n=L))

diff = np.max(np.abs(conv_time - conv_freq))

plt.figure(figsize=(10, 5))
plt.title(f"Twierdzenie o splocie: max różnica = {diff:.2e}")
plt.plot(conv_time, 'k-', linewidth=3, label='Splot w czasie (np.convolve)', alpha=0.5)
plt.plot(conv_freq, 'r--', label='Mnożenie widm (my_fft * my_fft)')
plt.legend()
plt.grid(True)
plt.savefig('output_img/2c.png')