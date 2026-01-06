import numpy as np

def _fft_recursive(x):
    N = x.shape[0]
    if N <= 1:
        return x
    even = _fft_recursive(x[0::2])
    odd  = _fft_recursive(x[1::2])
    k = np.arange(N // 2)
    factor = np.exp(-2j * np.pi * k / N)
    return np.concatenate([even + factor * odd, even - factor * odd])

def my_fft(x, n=None):
    x = np.asarray(x, dtype=complex)
    if n is None:
        n = x.shape[0]
    if x.shape[0] < n:
        x = np.concatenate([x, np.zeros(n - x.shape[0])])
    else:
        x = x[:n]
    N_power2 = 2**int(np.ceil(np.log2(n)))
    if N_power2 > n:
        x = np.concatenate([x, np.zeros(N_power2 - n)])
    result = _fft_recursive(x)
    return result[:n]

def my_ifft(x, n=None):
    x = np.asarray(x, dtype=complex)
    if n is None:
        n = x.shape[0]
    x_conj = np.conjugate(x)
    X = my_fft(x_conj, n)
    return np.conjugate(X) / X.shape[0]

def my_fftshift(x):
    x = np.asarray(x)
    mid = (x.shape[0] + 1) // 2
    return np.concatenate((x[mid:], x[:mid]))

def my_ifftshift(x):
    x = np.asarray(x)
    mid = x.shape[0] // 2
    return np.concatenate((x[mid:], x[:mid]))

def my_fftfreq(n, d=1.0):
    val = 1.0 / (n * d)
    results = np.arange(n, dtype=float)
    N_half = (n - 1) // 2 + 1
    results[N_half:] -= n
    return results * val

def create_ideal_lowpass(freqs, cutoff):
    return (np.abs(freqs) <= cutoff).astype(float)

def create_ideal_highpass(freqs, cutoff):
    return (np.abs(freqs) > cutoff).astype(float)

def create_gaussian_lowpass(freqs, sigma):
    return np.exp(-(freqs**2) / (2 * sigma**2))

def create_bandpass(freqs, low, high):
    return ((np.abs(freqs) >= low) & (np.abs(freqs) <= high)).astype(float)

import math

def ones(shape):
    if isinstance(shape, int):
        return [1.0] * shape
    else:
        rows, cols = shape
        return [[1.0] * cols for _ in range(rows)]

def hamming(M):
    if M < 1:
        return []
    if M == 1:
        return [1.0]
    
    return [
        0.54 - 0.46 * math.cos(2 * math.pi * n / (M - 1)) 
        for n in range(M)
    ]

def hanning(M):
    if M < 1:
        return []
    if M == 1:
        return [1.0]
        
    return [
        0.5 - 0.5 * math.cos(2 * math.pi * n / (M - 1)) 
        for n in range(M)
    ]


