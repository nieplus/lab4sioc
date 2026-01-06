import numpy as np
import matplotlib.pyplot as plt
import functions as fn  
from scipy.signal import medfilt  
import os
def generate_signal(n, fs):
    t = np.arange(n) / fs
    sig = 1.0 * np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    return t, sig

def add_gaussian_noise(signal, sigma=0.5):
    noise = np.random.normal(0, sigma, signal.shape)
    return signal + noise

def add_impulse_noise(signal, probability=0.05, amplitude=3.0):
    noisy_signal = signal.copy()
    num_impulses = int(probability * len(signal))
    indices = np.random.choice(len(signal), num_impulses, replace=False)
    impulses = np.random.choice([-amplitude, amplitude], num_impulses)
    noisy_signal[indices] += impulses
    return noisy_signal

if __name__ == "__main__":
    fs = 100.0
    n = 200 
    t, clean_sig = generate_signal(n, fs)

    print("1. Szum Gaussowski: Porównanie LowPass (Freq) vs Uśrednianie (Time)")
    noisy_gauss = add_gaussian_noise(clean_sig)

    F = fn.my_fft(noisy_gauss)
    freqs = fn.my_fftfreq(n, d=1/fs)
    cutoff = 25.0 
    mask = fn.create_ideal_lowpass(freqs, cutoff)
    filtered_freq_spec = F * mask
    filtered_gauss_fft = np.real(fn.my_ifft(filtered_freq_spec))

    window_size = 5
    kernel = np.array(fn.ones(window_size)) / window_size
    filtered_gauss_time = np.convolve(noisy_gauss, kernel, mode='same')

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Szum Gaussowski")
    plt.plot(t, clean_sig, 'k--', label='Czysty', alpha=0.6)
    plt.plot(t, noisy_gauss, 'gray', label='Zaszumiony', alpha=0.4)
    plt.plot(t, filtered_gauss_fft, 'r', label='FFT LowPass')
    plt.plot(t, filtered_gauss_time, 'b', label='Czas: Średnia')
    plt.legend()
    plt.grid(True)

    noisy_impulse = add_impulse_noise(clean_sig)

    F_imp = fn.my_fft(noisy_impulse)

    mask_gauss = fn.create_gaussian_lowpass(freqs, sigma=15.0) 
    filtered_imp_fft = np.real(fn.my_ifft(F_imp * mask_gauss))

    filtered_imp_time = medfilt(noisy_impulse, kernel_size=5)

    plt.subplot(1, 2, 2)
    plt.title("Szum Impulsowy")
    plt.plot(t, clean_sig, 'k--', label='Czysty', alpha=0.6)
    plt.plot(t, noisy_impulse, 'gray', label='Zaszumiony', alpha=0.4)
    plt.plot(t, filtered_imp_fft, 'r', label='FFT Gaussian LP')
    plt.plot(t, filtered_imp_time, 'g', label='Czas: Mediana')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if not os.path.exists('output_img'):
        os.makedirs('output_img')
    plt.savefig('output_img/zadanie_b.png')