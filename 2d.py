import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import os




plt.rcParams['figure.figsize'] = [12, 10]
plt.rcParams['font.size'] = 10
plt.style.use('bmh') 


def plot_windows():
    M = 64  
    fft_size = 2048 
    
    windows = {
        'Prostokątne': fn.ones(M),
        'Hamming': fn.hamming(M),
        'Hann': fn.hanning(M)
    }

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle('Wpływ okien na widmo (Amplituda i Faza)', fontsize=16)

    for i, (name, win) in enumerate(windows.items()):
        axs[i, 0].plot(win, '.-', color='tab:blue')
        axs[i, 0].set_title(f'Okno {name} (czas)')
        axs[i, 0].set_ylabel('Amplituda')
        axs[i, 0].grid(True)

        W = fn.my_fft(win, fft_size)
        W_shifted = fn.my_fftshift(W)
        freqs = np.linspace(-0.5, 0.5, fft_size)
        mag_db = 20 * np.log10(np.abs(W_shifted) / np.max(np.abs(W_shifted)) + 1e-10)
        
        phase = np.angle(W_shifted)

        color = 'tab:red'
        ax2 = axs[i, 1]
        ax2.plot(freqs, mag_db, color=color, label='Amplituda (dB)')
        ax2.set_ylabel('Magnituda [dB]', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim([-100, 5])
        ax2.set_title(f'Widmo okna {name}')
        
        ax3 = ax2.twinx()  
        color = 'tab:green'
        ax3.plot(freqs, phase, color=color, alpha=0.3, label='Faza (rad)')
        ax3.set_ylabel('Faza [rad]', color=color)
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.set_ylim([-np.pi, np.pi])
        
        axs[i, 1].grid(True)

    plt.tight_layout()
    plt.savefig('output_img/windows.png')


def plot_aliasing():
    t_end = 1.0
    f_signal = 9.0 
    
    fs_high = 1000 
    t_high = np.linspace(0, t_end, int(t_end * fs_high))
    y_high = np.cos(2 * np.pi * f_signal * t_high)

    fs_bad = 14.0 
    t_bad = np.linspace(0, t_end, int(t_end * fs_bad))
    y_bad = np.cos(2 * np.pi * f_signal * t_bad)

    f_alias = abs(f_signal - fs_bad)
    y_alias = np.cos(2 * np.pi * f_alias * t_high) 

    plt.figure(figsize=(12, 6))
    plt.title(f'Aliasing: Sygnał {f_signal} Hz próbkowany z {fs_bad} Hz udaje {f_alias} Hz')
    
    plt.plot(t_high, y_high, 'g-', alpha=0.5, label=f'Oryginał ({f_signal} Hz)')
    
    plt.plot(t_bad, y_bad, 'ro', label=f'Próbki (fs={fs_bad} Hz)')
    
    plt.plot(t_high, y_alias, 'b--', label=f'Alias / Pozorny sygnał ({f_alias} Hz)')
    
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.legend()
    plt.grid(True)
    plt.savefig('output_img/aliasing.png')

if __name__ == "__main__":
    if not os.path.exists("output_img"):
        os.makedirs("output_img")
    print("Generuję wykresy okien...")
    plot_windows()
    print("Generuję demonstrację aliasingu...")
    plot_aliasing()