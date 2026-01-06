import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

fs = 1000       
T = 1.0         
N = int(fs * T) 
t = np.linspace(0, T, N, endpoint=False)


sig_sin = np.sin(2 * np.pi * 50 * t)


sig_sum = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

sig_sq = np.sign(np.sin(2 * np.pi * 10 * t))

sygnaly = [("Sinus", sig_sin), ("Suma", sig_sum), ("Prostokąt", sig_sq)]


def manual_analysis(signal, fs):
    freqs = np.arange(0, fs//2, 1) 
    amps = []
    for f in freqs:

        ref_sin = np.sin(2 * np.pi * f * t)
        ref_cos = np.cos(2 * np.pi * f * t)
        
        re = np.sum(signal * ref_cos)
        im = np.sum(signal * ref_sin)
        
        magnitude = np.sqrt(re**2 + im**2) * (2/N) 
        amps.append(magnitude)
    return freqs, amps

plt.figure(figsize=(12, 10))


for i, (name, sig) in enumerate(sygnaly):
    fr_man, amp_man = manual_analysis(sig, fs)
    
    yf = fft(sig)
    xf = fftfreq(N, 1/fs)[:N//2]
    amp_fft = 2.0/N * np.abs(yf[:N//2]) 
    
    plt.subplot(3, 2, i*2 + 1)
    plt.plot(fr_man, amp_man, 'r--', label='Ręczny "Splot"', linewidth=2)
    plt.plot(xf, amp_fft, 'k-', alpha=0.5, label='FFT')
    plt.title(f"{name} - Porównanie")
    plt.legend()
    plt.grid(True, alpha=0.3)

sig_sum_shifted = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t + np.pi/2)

yf_orig = np.abs(fft(sig_sum))[:N//2]
yf_shift = np.abs(fft(sig_sum_shifted))[:N//2]

plt.subplot(3, 2, 2)
plt.plot(xf, yf_orig, 'b', label='Oryginał')
plt.plot(xf, yf_shift, 'r--', label='Przesunięty w fazie')
plt.title("Wpływ Fazy na Moduł FFT")
plt.legend()
plt.grid(True)

N_short = int(fs * 0.1) 
t_short = t[:N_short]
sig_short = np.sin(2 * np.pi * 50 * t_short) + 0.5 * np.sin(2 * np.pi * 60 * t_short) 

yf_short = fft(sig_short)
xf_short = fftfreq(N_short, 1/fs)[:N_short//2]

sig_long = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)
yf_long = fft(sig_long)

plt.subplot(3, 2, 4)
plt.plot(xf_short, 2.0/N_short * np.abs(yf_short)[:N_short//2], 'r-o', label='Krótki (0.1s)')
plt.plot(xf, 2.0/N * np.abs(yf_long)[:N//2], 'b-', alpha=0.3, label='Długi (1.0s)')
plt.title("Rozdzielczość: Krótki vs Długi sygnał")
plt.xlim(0, 100) 
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("output_img/signal_analysis_comparison.png")