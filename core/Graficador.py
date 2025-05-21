import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

class GraficadorSismico:
    def __init__(self, evento):
        self.evento = evento

    def procesar_trazas(self, aplicar_filtro=False, recortar=False,t_init=0,t_end =0):
        self.trace_x = self.evento.get_traza('X').copy().detrend(type='constant')
        self.trace_y = self.evento.get_traza('Y').copy().detrend(type='constant')
        self.trace_z = self.evento.get_traza('Z').copy().detrend(type='constant')
        
        if aplicar_filtro:
            self.trace_x.filter('highpass', freq=0.8, corners=2, zerophase=True)
            self.trace_y.filter('highpass', freq=0.8, corners=2, zerophase=True)
            self.trace_z.filter('highpass', freq=0.8, corners=2, zerophase=True)
        if recortar:
            start_time_x=self.evento.componentes['X'].starttime
            start_time_y=self.evento.componentes['Y'].starttime
            start_time_z=self.evento.componentes['Z'].starttime
            self.trace_x = self.trace_x.slice(start_time_x+t_init, start_time_x + t_end)
            self.trace_y = self.trace_y.slice(start_time_y+t_init, start_time_y + t_end)
            self.trace_z = self.trace_z.slice(start_time_z+t_init, start_time_z + t_end)

        self.t = np.arange(
            0,
            self.evento.componentes['Z'].data_long / self.evento.componentes['Z'].fs,
            self.evento.componentes['Z'].dt
        )
        # Ajustar longitudes si es necesario
        min_len = min(len(self.t), len(self.trace_x), len(self.trace_y), len(self.trace_z))
        self.t = self.t[:min_len]
        self.trace_x = self.trace_x[:min_len]
        self.trace_y = self.trace_y[:min_len]
        self.trace_z = self.trace_z[:min_len]

    def graficar_trazas(self, aplicar_filtro=False, recortar=False,t_init=0,t_end =0):
        self.procesar_trazas(aplicar_filtro, recortar,t_init,t_end)

        fs = self.evento.componentes['Z'].fs
        N = len(self.t)
        df = fs / N

        # FFT y espectros normalizados
        fx = fft(self.trace_x)
        fy = fft(self.trace_y)
        fz = fft(self.trace_z)

        self.freqs = fftfreq(N, 1/fs)[:N//2]
        self.amp_x = 2.0 / N * np.abs(fx[:N//2])
        self.amp_y = 2.0 / N * np.abs(fy[:N//2])
        self.amp_z = 2.0 / N * np.abs(fz[:N//2])

        self.amp_x /= self.amp_x.max()
        self.amp_y /= self.amp_y.max()
        self.amp_z /= self.amp_z.max()

        title = self.evento.componentes['Z'].titulo + '\nTriaxial velocity recording'

        fig = plt.figure(figsize=(12, 10), dpi=70)
        plt.get_current_fig_manager().window.geometry("800x700+0+0")

        plt.subplot(321)
        plt.plot(self.t, self.trace_z, 'b', linewidth=0.5)
        plt.ylabel('Z [nm/s]')
        plt.xlim([0, self.t[-1]])

        plt.subplot(322)
        plt.specgram(self.trace_z, NFFT=512, Fs=fs, cmap='jet')
        plt.ylabel('f [Hz]')
        plt.ylim(0, 20)

        plt.subplot(323)
        plt.plot(self.t, self.trace_y, 'r', linewidth=0.5)
        plt.ylabel('N [nm/s]')
        plt.xlim([0, self.t[-1]])

        plt.subplot(324)
        plt.specgram(self.trace_y, NFFT=512, Fs=fs, cmap='jet')
        plt.ylabel('f [Hz]')
        plt.ylim(0, 20)

        plt.subplot(325)
        plt.plot(self.t, self.trace_x, 'k', linewidth=0.5)
        plt.ylabel('E [nm/s]')
        plt.xlabel('tiempo [s]')
        plt.xlim([0, self.t[-1]])

        plt.subplot(326)
        plt.specgram(self.trace_x, NFFT=512, Fs=fs, cmap='jet')
        plt.ylabel('f [Hz]')
        plt.xlabel('tiempo [s]')
        plt.ylim(0, 20)
        fig.suptitle(title, fontsize=12)
    def graficar_espectros(self ):
        
        self.stacked_espectral = self.amp_x+self.amp_y+self.amp_z
        self.stacked_espectral /= self.stacked_espectral.max()
        

        title = self.evento.componentes['Z'].titulo + '\nStacked spectrum (triaxial)'
        fig = plt.figure(figsize=(12, 10), dpi=70)
        plt.get_current_fig_manager().window.geometry("800x700+0+0")

        plt.subplot(411)
        plt.plot(self.freqs, self.amp_z, 'b', linewidth=0.7)
        plt.ylabel('Z ')
        plt.xlim(0,20)
        plt.ylim(0,1.2)
        
        plt.subplot(412)
        plt.plot(self.freqs, self.amp_y, 'r', linewidth=0.7)
        plt.ylabel('N')
        plt.xlim(0,20)
        plt.ylim(0,1.2)
        
        plt.subplot(413)
        plt.plot(self.freqs, self.amp_x, 'k', linewidth=0.7)
        plt.ylabel('E')
        plt.xlim(0,20)
        plt.ylim(0,1.2)
        plt.subplot(414)
        
        plt.plot(self.freqs, self.stacked_espectral, 'g', linewidth=0.7)
        plt.ylabel('stacked Spec. Amp')
        plt.xlabel('f [Hz]')
        plt.xlim(0,20)
        plt.ylim(0,1.2)
        fig.suptitle(title, fontsize=12)

    def picar_frecuencias_espectro(self, freq_x, stacked_espectral):
        def find_nearest(array, array2, value, value2): 
            array = np.asarray(array)
            array2 =  np.asarray(array2)
            idx = (np.abs(array - value) + np.abs(array2 - value2)).argmin()
            return idx
        def tellme(message):
            print(message)
            plt.title(message, fontsize=10)
            plt.draw()
        picking = True
        pt_aux = []
        fig = plt.figure(figsize=(12, 10), dpi=100)
        plt.get_current_fig_manager().window.geometry("800x700+850+0")
        plt.plot(freq_x, stacked_espectral, 'g', linewidth=0.8)
        plt.xlim(0, 20)
        plt.ylim(0, 1.1)
        plt.ylabel('Spec. Amp [counts/Hz]')
        plt.xlabel('f [Hz]')
        tellme('Click on this figure')
        plt.waitforbuttonpress()
        
        while True:
            pts = []
            while picking:
                tellme('Select frequency')
                pts = np.asarray(plt.ginput(1, timeout=-1))
                punto = pts[0, :]
                pt = find_nearest(freq_x, stacked_espectral, punto[0], punto[1])
                pt_aux.append(pt)
                plt.plot(freq_x[pt], stacked_espectral[pt], 'o', markersize=5)
                plt.annotate(str(round(freq_x[pt], 2)), (freq_x[pt], stacked_espectral[pt]))
                tellme('1.Finalize: Enter\n2.Continue selecting frequencies: Mouse click')
                if plt.waitforbuttonpress():
                    break
            # Confirmación
            plt.plot(freq_x[pt_aux], stacked_espectral[pt_aux], 'o', markersize=5)
            for i in pt_aux:
                plt.annotate(str(round(freq_x[i], 2)), (freq_x[i], stacked_espectral[i]))
            tellme('Correct? → Enter.\nRedo? → Mouse click')
            if plt.waitforbuttonpress():
                break
            # Reiniciar selección
            plt.cla()
            plt.plot(freq_x, stacked_espectral, 'g', linewidth=0.8)
            plt.xlim(0, 20)
            plt.ylim(0, 1.1)
            plt.ylabel('Spec. Amp [counts/Hz]')
            plt.xlabel('f [Hz]')
        plt.show

        self.frecuencias_pickeadas = [freq_x[i] for i in pt_aux]
        title = 'Picked frequencies'
        plt.clf()
        plt.get_current_fig_manager().window.geometry("800x700+850+0")
        plt.plot(freq_x, stacked_espectral, 'r', linewidth=0.8)
        for i in pt_aux:
            plt.axvline(freq_x[i], color="b", linewidth=0.8, linestyle="dashed")
            plt.annotate(str(round(freq_x[i], 2)), (freq_x[i], stacked_espectral[i]))
        plt.xlabel('f [Hz]')
        plt.ylabel('Spec. Amp [counts/Hz]')
        plt.xlim(0, 20)
        plt.ylim(0, 1.1)
        plt.title(title)
    
        
        