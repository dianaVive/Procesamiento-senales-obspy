

import numpy as np
import matplotlib.pyplot as plt
import obspy
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from tkinter import filedialog
from obspy import read
from itertools import islice
from obspy import UTCDateTime, read, Trace, Stream
from scipy.fft import fft, fftfreq
from pylab import *
from obspy.signal.tf_misfit import cwt
import math
import scipy.optimize as opt
from scipy.optimize import minimize
from scipy.signal import hilbert
from scipy.interpolate import InterpolatedUnivariateSpline
from numpy.linalg import eig
from mpl_toolkits import mplot3d
import os
from mpl_toolkits.mplot3d import axes3d
import pandas as pd

class lector_archivos:
    def configura_Ascii(self,file_name):
        lines_number_head =5
        with open(file_name) as input_file:
            head = list(islice(input_file, lines_number_head))
            data = np.genfromtxt(input_file, dtype=None,encoding=None,usecols=(0))
        Fecha_hora_list = list(map(str.strip, head[2].split(' ')))
        Fs_list = list(map(str.strip, head[3].split(' ')))
        mmddHHSS_texto = head[0]
        hora_texto = Fecha_hora_list[1] 
        fecha_texto = Fecha_hora_list[0]
        station_chanel = head[1]
        self.component = station_chanel[-1]
        self.station = station_chanel[0:3]
        fs_texto = Fs_list[0]
        yyyy = fecha_texto[0:4]
        mm = fecha_texto[5:7]
        day = fecha_texto[8:10]
        HH = hora_texto[0:2]
        MM = hora_texto[3:5]
        self.senal = yyyy+mm+day+HH+MM
        self.titulo = self.senal+ ' ' + self.station.strip()
        self.fs = float(fs_texto)    #Sampling frequency [Hz]
        self.dt = 1/self.fs               #Sampling time interval [samples/s]
        self.fn = self.fs/2               #Nyquist frequency [Hz]
        self.data_long=len(data);      #Total lenght of the trace [samples]
        stats = {'network': 'OP', 'station': self.station, 'location': '',
                'channel': '', 'npts': len(data), 'sampling_rate': self.fs,
                'mseed': {'dataquality': 'D'},'starttime':UTCDateTime()}
                # set current time
        stats['starttime'] = UTCDateTime()
        self.starttime = UTCDateTime()
        self.st = Stream([Trace(data=data, header=stats)])
        st_cpy = Stream([Trace(data=data, header=stats)])
        #self.st.write('sismo.mseed', format='MSEED')  
        #st1 = read("sismo.mseed")    
    def Configura_mseed(self,file_name):
        self.st = read(file_name,format="mseed")
        traza =self.st[0]
        self.fs = traza.stats.sampling_rate
        self.data_long = traza.stats.npts
        self.component = traza.stats.channel
        #self.data_long = len(traza.data)
        fecha = str(traza.stats.starttime)[0:16]
        fecha = fecha.replace('-', '')
        fecha = fecha.replace('T', '')
        fecha = fecha.replace(':', '')
        self.titulo = fecha+ ' ' + traza.stats.station
        self.senal = fecha
        self.dt = 1/self.fs
        self.fn = self.fs/2
        self.station = traza.stats.station+self.component[2]
        self.starttime= traza.stats.starttime
def find_nearest2(array, array2, value, value2): 
    array = np.asarray(array)
    array2 =  np.asarray(array2)
    idx = (np.abs(array - value)+np.abs(array2-value2)).argmin() 
    return idx
def tellme(s):
    print(s)
    plt.title(s, fontsize=10)
    plt.draw()
def calcular_Coseno_metodo(A_Inicial,frequency,Q_metodo,t_exponencial_new): 
    Acos = np.array(list(map(lambda t_exponencial_new: A_Inicial*math.cos(2*math.pi*frequency*t_exponencial_new)*math.exp(-(math.pi*frequency*t_exponencial_new/Q_metodo)),t_exponencial_new)))
    return Acos
def calcular_residual(sx_filt,Acos): 
    square1 = np.array(list(map(lambda x:x**2,(sx_filt-Acos))))
    square2 = np.array(list(map(lambda x:x**2,sx_filt)))
    residual = sum(square1)/sum(square2)
    return residual
def save_data(tecnica,Q,h,r):
    lista_tecnica.append(tecnica) 
    listaQ.append(Q) 
    lista_h.append(h) 
    lista_residual.append(r) 
def find_max(k,index,N,data):
    vector_busqueda = np.arange(index-k, N, 1)
    param_busqueda = data[vector_busqueda]
    index_max= vector_busqueda[np.array(param_busqueda).argmax()]
    return index_max
def rotate_ne_rt(n, e, ba):
    """
    Rotates horizontal components of a seismogram.

    The North- and East-Component of a seismogram will be rotated in Radial
    and Transversal Component. The angle is given as the back-azimuth, that is
    defined as the angle measured between the vector pointing from the station
    to the source and the vector pointing from the station to the North.

    :type n: :class:`~numpy.ndarray`
    :param n: Data of the North component of the seismogram.
    :type e: :class:`~numpy.ndarray`
    :param e: Data of the East component of the seismogram.
    :type ba: float
    :param ba: The back azimuth from station to source in degrees.
    :return: Radial and Transversal component of seismogram.
    """
    if len(n) != len(e):
        raise TypeError("North and East component have different length.")
    if ba < 0 or ba > 360:
        raise ValueError("Back Azimuth should be between 0 and 360 degrees.")
    ba = radians(ba)
    r = - e * sin(ba) - n * cos(ba)
    t = - e * cos(ba) + n * sin(ba)
    return r, t
def rotate_zne_lqt(z, n, e, ba, inc):
    """
    Rotates all components of a seismogram.

    The components will be rotated from ZNE (Z, North, East, left-handed) to
    LQT (e.g. ray coordinate system, right-handed). The rotation angles are
    given as the back-azimuth and inclination.

    The transformation consists of 3 steps::

        1. mirroring of E-component at ZN plain: ZNE -> ZNW
        2. negative rotation of coordinate system around Z-axis with angle ba:
           ZNW -> ZRT
        3. negative rotation of coordinate system around T-axis with angle inc:
           ZRT -> LQT

    :type z: :class:`~numpy.ndarray`
    :param z: Data of the Z component of the seismogram.
    :type n: :class:`~numpy.ndarray`
    :param n: Data of the North component of the seismogram.
    :type e: :class:`~numpy.ndarray`
    :param e: Data of the East component of the seismogram.
    :type ba: float
    :param ba: The back azimuth from station to source in degrees.
    :type inc: float
    :param inc: The inclination of the ray at the station in degrees.
    :return: L-, Q- and T-component of seismogram.
    """
    if len(z) != len(n) or len(z) != len(e):
        raise TypeError("Z, North and East component have different length!?!")
    if ba < 0 or ba > 360:
        raise ValueError("Back Azimuth should be between 0 and 360 degrees!")
    if inc < 0 or inc > 360:
        raise ValueError("Inclination should be between 0 and 360 degrees!")
    ba = radians(ba)
    inc = radians(inc)
    l = z * cos(inc) - n * sin(inc) * cos(ba) - e * sin(inc) * sin(ba)  # NOQA
    q = z * sin(inc) + n * cos(inc) * cos(ba) + e * cos(inc) * sin(ba)  # NOQA
    t = n * sin(ba) - e * cos(ba)  # NOQA
    return l, q, t
def plot_windows(numero_ventanas,data_x,data_y,figx,maxMin):
    n1=0
    n2=n1+npw-1
    left_b=0.05
    bottom_b=0.8
    width_b=0.1
    height_b=0.1
    spacing =0.05
    width_subplot=0.11
    contador=0
    contador_h=0
    for i in range(numero_ventanas):
        if left_b>1:
            contador =0
            contador_h = contador_h+1
            left_b=0.05
            bottom_b=1-(spacing*(contador_h)+(contador_h)*width_subplot)-0.2
        pos_plot = [left_b,bottom_b,width_b,height_b]
        data_subplot_x = data_x[n1:n2]
        data_subplot_y = data_y[n1:n2]
        figx.add_axes(pos_plot)
        plt.plot(data_subplot_x,data_subplot_y,linewidth=0.5)
        plt.xticks(fontsize=4)
        plt.yticks(fontsize=4)
        plt.subplots_adjust(wspace=1.5, hspace=3.5)
        plt.xlim(-maxMin,maxMin)
        plt.ylim(-maxMin,maxMin)
        #plt.tight_layout()
        n1 = n2+1
        n2 = n1+npw-1
        left_b =spacing*(contador +2)+(contador+1)*width_subplot
        contador = contador+1
def Max_concatenated_data(x,y,z):
    concatenate_data = np.concatenate((x, y, z), axis=None)
    concatenated_data_max = concatenate_data[concatenate_data.argmax()]
    concatenated_data_min = concatenate_data[concatenate_data.argmin()]
    concatenate_data = np.concatenate((concatenated_data_max,concatenated_data_min), axis=None)
    scal= abs(concatenate_data)
    ej=scal[scal.argmax()]
    return ej
def Data_respuesta_instrumental(volcan,senal,componente):
    if volcan.upper()=='CHILES':
        tabla_respuesta = 'respuestasLakiy_Chiles_movitopar.csv'
    elif volcan.upper()=='GALERAS':
        tabla_respuesta = 'respuestasLakiy_Galeras_movitopar.csv'
    elif volcan.upper()=='CUMBAL':
        tabla_respuesta = 'respuestasLakiy_Cumbal_movitopar.csv'
    else:
        tabla_respuesta =[]
    factor_respuesta_csv = pd.read_csv(tabla_respuesta, sep=';')
    fecha_especifica = pd.to_datetime(senal,format='%Y%m%d%H%M')
    resultado = factor_respuesta_csv[(factor_respuesta_csv['componente'] == componente)]
    rango_encontrado =[]
    Resultado_busqueda = pd.DataFrame()
    nombre_columna_1 = 'Respuesta lakiy (nm/s)cta'  # Reemplaza esto con el nombre de la columna deseada
    nombre_columna_2 = 'Latitud'  # Reemplaza esto con el nombre de la columna deseada
    nombre_columna_3 = 'Longitud'  # Reemplaza esto con el nombre de la columna deseada
    for i in range(len(resultado)):
        vigencia_Inicial = pd.to_datetime(resultado['Vigencia Inicial'].iloc[i],dayfirst=True)
        vigencia_Final = pd.to_datetime(resultado['Vigencia Final'].iloc[i],dayfirst=True)
        if vigencia_Inicial <= fecha_especifica <=vigencia_Final:
            rango_encontrado = i
            break
    if rango_encontrado !=[]:
        respuesta_instrumental = resultado[nombre_columna_1].iloc[rango_encontrado]
        Latitud = resultado[nombre_columna_2].iloc[rango_encontrado]
        Longitud = resultado[nombre_columna_3].iloc[rango_encontrado]
        Resultado_busqueda = resultado.iloc[rango_encontrado]
    else:
        respuesta_instrumental = 1
        Latitud = resultado[nombre_columna_2].iloc[0]
        Longitud = resultado[nombre_columna_3].iloc[0]
        print('Validity or seismological station not found, the instrumental response factors are 1')
    return respuesta_instrumental, Latitud, Longitud, Resultado_busqueda


wd=os.getcwd()
print('Ingrese componente X (Oeste-Este)')
x_component = filedialog.askopenfilename()
print('Ingrese componente Y (Sur-Norte)')
y_component = filedialog.askopenfilename()
print('Ingrese componente Z (Vertical)')
z_component = filedialog.askopenfilename()


datos_evento_z = lector_archivos()
datos_evento_x = lector_archivos()
datos_evento_y = lector_archivos()

if z_component.endswith('txt'):
    datos_evento_x.configura_Ascii(x_component)
    datos_evento_y.configura_Ascii(y_component)
    datos_evento_z.configura_Ascii(z_component)
if z_component.endswith('mseed'):
    datos_evento_x.Configura_mseed(x_component)
    datos_evento_y.Configura_mseed(y_component)
    datos_evento_z.Configura_mseed(z_component)
volcan = input('Enter volcano: ')
estacion = str(datos_evento_z.station).strip()
titulo_directorio = str(datos_evento_z.senal)
data_out = wd+'/../results/'+volcan+'/'+titulo_directorio+'/'
save_fig_title = data_out
try:
    os.mkdir(data_out)
except:
    print("The folder already exists")
lista_tecnica = []
listaQ = []
lista_h =[]
lista_residual=[]

cadena_aux = datos_evento_x.station[0:2]
if cadena_aux =='UR':
    factor_respuesta_X, Latitud, Longitud,Resultado_busqueda_X = Data_respuesta_instrumental(volcan,datos_evento_x.senal,datos_evento_x.station+'W')
    factor_respuesta_Y, Latitud, Longitud,Resultado_busqueda_Y= Data_respuesta_instrumental(volcan,datos_evento_x.senal,datos_evento_y.station+'S')
    factor_respuesta_Z, Latitud, Longitud,Resultado_busqueda_Z= Data_respuesta_instrumental(volcan,datos_evento_x.senal,datos_evento_z.station+'R')
elif cadena_aux =='OB':
    factor_respuesta_X, Latitud, Longitud,Resultado_busqueda_X = Data_respuesta_instrumental(volcan,datos_evento_x.senal,datos_evento_x.station+'R')
    factor_respuesta_Y, Latitud, Longitud,Resultado_busqueda_Y= Data_respuesta_instrumental(volcan,datos_evento_x.senal,datos_evento_y.station+'R')
    factor_respuesta_Z, Latitud, Longitud,Resultado_busqueda_Z= Data_respuesta_instrumental(volcan,datos_evento_x.senal,datos_evento_z.station+'R')
else:
    factor_respuesta_X, Latitud, Longitud,Resultado_busqueda_X = Data_respuesta_instrumental(volcan,datos_evento_x.senal,datos_evento_x.station+'E')
    factor_respuesta_Y, Latitud, Longitud,Resultado_busqueda_Y= Data_respuesta_instrumental(volcan,datos_evento_x.senal,datos_evento_x.station+'N')
    factor_respuesta_Z, Latitud, Longitud,Resultado_busqueda_Z= Data_respuesta_instrumental(volcan,datos_evento_x.senal,datos_evento_x.station+'Z')
print('The found response factors are:'+ '\n'+ 'Este = '+ str(factor_respuesta_X) + '\n'+ 'Norte = '+str(factor_respuesta_Y) + '\n'+ 'Vertical = '+str(factor_respuesta_Z))


stream_read_x = datos_evento_x.st
stream_read_y = datos_evento_y.st
stream_read_z = datos_evento_z.st

data_original_x = stream_read_x[0].copy().data  #Aux x
data_original_y = stream_read_y[0].copy().data  #Aux y
data_original_z = stream_read_z[0].copy().data  #Aux z

trace_x = stream_read_x[0].detrend(type='constant')
trace_y = stream_read_y[0].detrend(type='constant')
trace_z = stream_read_z[0].detrend(type='constant')

tr_filt_x = trace_x.copy() 
tr_filt_y = trace_y.copy() 
tr_filt_z = trace_z.copy()  

datos_detrended_x = factor_respuesta_X*trace_x.data
datos_detrended_y = factor_respuesta_Y*trace_y.data
datos_detrended_z = factor_respuesta_Z*trace_z.data

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
time_vector = np.arange(0, datos_evento_z.data_long / datos_evento_z.fs, datos_evento_z.dt)
if len(datos_detrended_z) <len(time_vector):
    time_vector = time_vector[:len(datos_detrended_z)] # se agrega para igualar tamaño de vectores
elif len(time_vector) <len(datos_detrended_z):
    datos_detrended_x = datos_detrended_x[:len(time_vector)]
    datos_detrended_y = datos_detrended_y[:len(time_vector)]   
    datos_detrended_z = datos_detrended_z[:len(time_vector)]      
print('---------------------------------------------------------------') 
t = time_vector
data_fourier_x = fft(datos_detrended_x)
data_fourier_y = fft(datos_detrended_y)
data_fourier_z = fft(datos_detrended_z)

N = datos_evento_z.data_long
df = datos_evento_z.fs/N

freq_x = fftfreq(N, datos_evento_x.dt)[:N//2]   #x
freq_y = fftfreq(N, datos_evento_y.dt)[:N//2]
freq_z = fftfreq(N, datos_evento_z.dt)[:N//2]

amp_espectral_x = 2.0/N * np.abs(data_fourier_x[0:N//2])
amp_espectral_y = 2.0/N * np.abs(data_fourier_y[0:N//2])
amp_espectral_z = 2.0/N * np.abs(data_fourier_z[0:N//2])

amp_espectral_x=amp_espectral_x/amp_espectral_x[amp_espectral_x.argmax()]
amp_espectral_y=amp_espectral_y/amp_espectral_y[amp_espectral_y.argmax()]
amp_espectral_z=amp_espectral_z/amp_espectral_z[amp_espectral_z.argmax()]

print('----------------------------------------------------------------')

titleFig = datos_evento_z.titulo + '\n' + 'Registro triaxial en velocidad'
fig1 = plt.figure(figsize=(12,10), dpi=70) 
plt.get_current_fig_manager().window.geometry("800x700+0+0")#"widthxheight+Xposition+Yposition"
plt.subplot(321)
plt.plot(time_vector, datos_detrended_z, 'b',linewidth=0.5)
plt.ylabel('Z [nm/s]')
plt.xlim([0, datos_evento_z.data_long/datos_evento_z.fs])
plt.subplot(322)
plt.specgram(datos_detrended_z, NFFT=512, Fs=datos_evento_z.fs, cmap = 'jet')
plt.ylabel('f [Hz]')
plt.ylim(0,20)
plt.subplot(323)
plt.plot(time_vector, datos_detrended_y, 'r',linewidth=0.5)
plt.ylabel('N [nm/s]')
plt.xlim([0, datos_evento_y.data_long/datos_evento_y.fs])
plt.subplot(324)
plt.specgram(datos_detrended_y, NFFT=512, Fs=datos_evento_y.fs, cmap = 'jet')
plt.ylabel('f [Hz]')
plt.ylim(0,20)
plt.subplot(325)
plt.plot(time_vector, datos_detrended_x, 'k',linewidth=0.5)
plt.ylabel('E [n/s]')
plt.xlabel('tiempo [s]')
plt.xlim([0, datos_evento_x.data_long/datos_evento_x.fs])
plt.subplot(326)
plt.specgram(datos_detrended_x, NFFT=512, Fs=datos_evento_x.fs, cmap = 'jet')
plt.ylabel('f [Hz]')
plt.xlabel('tiempo [s]')
plt.ylim(0,20)
fig1.suptitle(titleFig, fontsize=12)
plt.show(block=False)

filtro_cp = input('Do you want to filter the trace as a short period? (y/n) ')
if filtro_cp == 'y':
    trace_x.filter('highpass',  freq=0.8,corners=2, zerophase=True)
    trace_y.filter('highpass',  freq=0.8,corners=2, zerophase=True)
    trace_z.filter('highpass',  freq=0.8,corners=2, zerophase=True)

    datos_detrended_x = factor_respuesta_X*trace_x.data
    datos_detrended_y = factor_respuesta_Y*trace_y.data
    datos_detrended_z = factor_respuesta_Z*trace_z.data

    data_fourier_x = fft(datos_detrended_x)
    data_fourier_y = fft(datos_detrended_y)
    data_fourier_z = fft(datos_detrended_z)

    N = datos_evento_z.data_long
    df = datos_evento_z.fs/N


    freq_x = fftfreq(N, datos_evento_x.dt)[:N//2]   #x
    freq_y = fftfreq(N, datos_evento_y.dt)[:N//2]
    freq_z = fftfreq(N, datos_evento_z.dt)[:N//2]

    amp_espectral_x = 2.0/N * np.abs(data_fourier_x[0:N//2])
    amp_espectral_y = 2.0/N * np.abs(data_fourier_y[0:N//2])
    amp_espectral_z = 2.0/N * np.abs(data_fourier_z[0:N//2])

    amp_espectral_x=amp_espectral_x/amp_espectral_x[amp_espectral_x.argmax()]
    amp_espectral_y=amp_espectral_y/amp_espectral_y[amp_espectral_y.argmax()]
    amp_espectral_z=amp_espectral_z/amp_espectral_z[amp_espectral_z.argmax()]

    #fig1 = plt.figure(figsize=(12,10), dpi=70) 
    plt.clf()
    plt.get_current_fig_manager().window.geometry("800x700+0+0")#"widthxheight+Xposition+Yposition"
    plt.subplot(321)
    plt.plot(time_vector, datos_detrended_z, 'b',linewidth=0.5)
    plt.ylabel('Z [nm/s]')
    plt.xlim([0, datos_evento_z.data_long/datos_evento_z.fs])
    plt.subplot(322)
    plt.specgram(datos_detrended_z, NFFT=512, Fs=datos_evento_z.fs, cmap = 'jet')
    plt.ylabel('f [Hz]')
    plt.ylim(0,20)
    plt.subplot(323)
    plt.plot(time_vector, datos_detrended_y, 'r',linewidth=0.5)
    plt.ylabel('N [nm/s]')
    plt.xlim([0, datos_evento_y.data_long/datos_evento_y.fs])
    plt.subplot(324)
    plt.specgram(datos_detrended_y, NFFT=512, Fs=datos_evento_y.fs, cmap = 'jet')
    plt.ylabel('f [Hz]')
    plt.ylim(0,20)
    plt.subplot(325)
    plt.plot(time_vector, datos_detrended_x, 'k',linewidth=0.5)
    plt.ylabel('E [n/s]')
    plt.xlabel('tiempo [s]')
    plt.xlim([0, datos_evento_x.data_long/datos_evento_x.fs])
    plt.subplot(326)
    plt.specgram(datos_detrended_x, NFFT=512, Fs=datos_evento_x.fs, cmap = 'jet')
    plt.ylabel('f [Hz]')
    plt.xlabel('tiempo [s]')
    plt.ylim(0,20)
    fig1.suptitle(titleFig, fontsize=12)
    plt.show(block=False)
recortar = input('Do you want to cut the seismic signal? (y/n) ')
if recortar == 'y':
    start_time_x=datos_evento_x.starttime
    start_time_y=datos_evento_y.starttime
    start_time_z=datos_evento_z.starttime
    t_init = int(input('Initial time (s): '))
    t_end = int(input('Final time (s): '))
    t_intervalo = t_end-t_init 


    trace_x = trace_x.slice(start_time_x+t_init, start_time_x + t_end)
    trace_y = trace_y.slice(start_time_y+t_init, start_time_y + t_end)
    trace_z = trace_z.slice(start_time_z+t_init, start_time_z + t_end)

    tr_filt_x = trace_x.copy() 
    tr_filt_y = trace_y.copy() 
    tr_filt_z = trace_z.copy() 

    time_vector = np.arange(0, len(trace_z.data) / datos_evento_z.fs, datos_evento_z.dt)
    datos_detrended_x = factor_respuesta_X*trace_x.data
    datos_detrended_y = factor_respuesta_Y*trace_y.data
    datos_detrended_z = factor_respuesta_Z*trace_z.data


    if len(datos_detrended_z) <len(time_vector):
        time_vector = time_vector[:len(datos_detrended_z)] # se agrega para igualar tamaño de vectores
    elif len(time_vector) <len(datos_detrended_z):
        datos_detrended_x = datos_detrended_x[:len(time_vector)]
        datos_detrended_y = datos_detrended_y[:len(time_vector)]   
        datos_detrended_z = datos_detrended_z[:len(time_vector)]  

    t = time_vector
    data_fourier_x = fft(datos_detrended_x)
    data_fourier_y = fft(datos_detrended_y)
    data_fourier_z = fft(datos_detrended_z)

    N = len(trace_z.data)
    df = datos_evento_z.fs/N

    freq_x = fftfreq(N, datos_evento_x.dt)[:N//2]   #x
    freq_y = fftfreq(N, datos_evento_y.dt)[:N//2]
    freq_z = fftfreq(N, datos_evento_z.dt)[:N//2]

    amp_espectral_x = 2.0/N * np.abs(data_fourier_x[0:N//2])
    amp_espectral_y = 2.0/N * np.abs(data_fourier_y[0:N//2])
    amp_espectral_z = 2.0/N * np.abs(data_fourier_z[0:N//2])

    amp_espectral_x=amp_espectral_x/amp_espectral_x[amp_espectral_x.argmax()]
    amp_espectral_y=amp_espectral_y/amp_espectral_y[amp_espectral_y.argmax()]
    amp_espectral_z=amp_espectral_z/amp_espectral_z[amp_espectral_z.argmax()]

    #fig1 = plt.figure(figsize=(12,10), dpi=80) 
    plt.clf()
    plt.get_current_fig_manager().window.geometry("800x700+0+0")#"widthxheight+Xposition+Yposition"
    plt.subplot(321)
    plt.plot(time_vector, datos_detrended_z, 'b',linewidth=0.5)
    plt.ylabel('Z [nm/s]')
    plt.xlim([0, len(trace_z.data)/datos_evento_z.fs])
    plt.subplot(322)
    plt.specgram(datos_detrended_z, NFFT=512, Fs=datos_evento_z.fs, cmap = 'jet')
    plt.ylabel('f [Hz]')
    plt.ylim(0,20)
    plt.subplot(323)
    plt.plot(time_vector, datos_detrended_y, 'r',linewidth=0.5)
    plt.ylabel('N [nm/s]')
    plt.xlim([0, len(trace_y.data)/datos_evento_y.fs])
    plt.subplot(324)
    plt.specgram(datos_detrended_y, NFFT=512, Fs=datos_evento_y.fs, cmap = 'jet')
    plt.ylabel('f [Hz]')
    plt.ylim(0,20)
    plt.subplot(325)
    plt.plot(time_vector, datos_detrended_x, 'k',linewidth=0.5)
    plt.ylabel('E [n/s]')
    plt.xlabel('tiempo [s]')
    plt.xlim([0, len(trace_z.data)/datos_evento_z.fs])
    plt.subplot(326)
    plt.specgram(datos_detrended_x, NFFT=512, Fs=datos_evento_x.fs, cmap = 'jet')
    plt.ylabel('f [Hz]')
    plt.xlabel('tiempo [s]')
    plt.ylim(0,20)
    fig1.suptitle(titleFig+'\n'+'Trimmed signal', fontsize=12)
    plt.show(block=False)

#Apilamiento de espectros
stacked_espectral = amp_espectral_x+amp_espectral_y+amp_espectral_z
stacked_espectral=stacked_espectral/stacked_espectral[stacked_espectral.argmax()]

titleFig2 = datos_evento_z.titulo+'\n'+ 'Espectro Triaxial y Apilado (Normalizado)'
fig2 = plt.figure(figsize=(12,10), dpi=80) 
plt.get_current_fig_manager().window.geometry("800x700+0+0")#"widthxheight+Xposition+Yposition"
plt.subplot(411)
plt.plot(freq_z, amp_espectral_z, 'b', linewidth=0.7)
plt.ylabel('Z ')
plt.xlim(0,20)
plt.ylim(0,1.2)
plt.subplot(412)
plt.plot(freq_y, amp_espectral_y, 'r', linewidth=0.7)
plt.ylabel('N')
plt.xlim(0,20)
plt.ylim(0,1.2)
plt.subplot(413)
plt.plot(freq_x, amp_espectral_x, 'k', linewidth=0.7)
plt.ylabel('E')
plt.xlim(0,20)
plt.ylim(0,1.2)
plt.subplot(414)
plt.plot(freq_x, stacked_espectral, 'g', linewidth=0.7)
plt.ylabel('stacked Spec. Amp')
plt.xlabel('f [Hz]')
plt.xlim(0,20)
plt.ylim(0,1.2)
fig2.suptitle(titleFig2, fontsize=12)
plt.show(block=False)
#--------------------------------------------------------
picking = True
fig3 = plt.figure(figsize=(12,10), dpi=100) 
plt.get_current_fig_manager().window.geometry("800x700+850+0")#"widthxheight+Xposition+Yposition"
plt.plot(freq_x,stacked_espectral, 'g',linewidth=0.8)
plt.xlim(0,20)
plt.ylim(0,1.1) 
plt.ylabel('Spec. Amp [counts/Hz]')
plt.xlabel('f [Hz]')  
tellme('Click on this figure')
plt.waitforbuttonpress()
while True:
    pts = []
    pt_aux = []
    while picking ==True:
        tellme('Select frequency')
        pts = np.asarray(plt.ginput(1, timeout=-1))
        punto = pts[0,:]
        pt = find_nearest2(freq_x,stacked_espectral, punto[0],punto[1])
        pt_aux.append(pt)
        Frecuencia_pt_aux = freq_x[pt]
        amplitud_frecuencia =stacked_espectral[pt]
        plt.plot(freq_x[pt], stacked_espectral[pt], 'o', markersize='5')
        plt.annotate(str(round(freq_x[pt],2)), (freq_x[pt],stacked_espectral[pt]))
        tellme('1.Finalize:--------------------------------->Enter \n' '2.Continue selecting frequencies-----------> Mouse click')
        if plt.waitforbuttonpress():
            break
    plt.plot(freq_x[pt_aux], stacked_espectral[pt_aux], 'o', markersize='5')
    for i in pt_aux:
        plt.annotate(str(round(freq_x[i],2)), (freq_x[i],stacked_espectral[i]))
    tellme('Correct?-------> Enter. \n' 'Re-enter frequencies------> mouse click')
    if plt.waitforbuttonpress():
        break
    plt.cla()
    plt.plot(freq_x,stacked_espectral, 'g',linewidth=0.8)
    plt.xlim(0,20)
    plt.ylim(0,1.1) 
    plt.ylabel('Spec. Amp [counts/Hz]')
    plt.xlabel('f [Hz]')  
plt.show

titleFig3 = datos_evento_z.titulo +'\n'+ 'Picked frequencies'
plt.clf()
plt.get_current_fig_manager().window.geometry("800x700+850+0")#"widthxheight+Xposition+Yposition"
plt.plot(freq_x,stacked_espectral, 'r',linewidth=0.8)
for i in pt_aux:
    plt.axvline(freq_x[i], color = "b", linewidth = 0.8, linestyle = "dashed")
    plt.annotate(str(round(freq_x[i],2)), (freq_x[i],stacked_espectral[i]))
plt.xlabel('f [Hz]')
plt.ylabel('Spec. Amp [counts/Hz]')
plt.xlim(0,20)
plt.ylim(0,1.1) 
fig3.suptitle(titleFig3, fontsize=12)
plt.show(block=False)

tipo_procesamiento = int(input('What kind of processing do you want? Select the Process \n' '1. Minimum_frequency \n' '2. Dominant_frequency\n' '3. First_picking \n' '4. Full_spectrum \n'))
if tipo_procesamiento==1:
    processing = 'Minimum_frequency'
    picked_frequencies = freq_x[pt_aux]
    min_frequency_index = pt_aux[np.array(picked_frequencies).argmin()]
    indl = min_frequency_index
if tipo_procesamiento==2:
    processing = 'Dominant_frequency'
    picked_frequencies = stacked_espectral[pt_aux]
    dom_frequency_index = pt_aux[np.array(picked_frequencies).argmax()]
    indl = dom_frequency_index
if tipo_procesamiento==3:
    processing = 'First_picking'
    picked_frequencies = stacked_espectral[pt_aux]
    first_picking = pt_aux[0]
    indl = first_picking

k = round(0.1*N/datos_evento_z.fs)  #k = f*N/fs
if tipo_procesamiento!=4:
    index_max= find_max(k,indl,indl+k,stacked_espectral)
    frequency = round(freq_x[index_max],2)
    cadena_filt=str(round(frequency,2))
    frequency_doc_results = round(freq_x[index_max],4)
    frequency_fig=round(freq_x[index_max],2)
    T_period =round(1/frequency_doc_results,4)
    unid='Hz'
    f_str='f='
    print('*********************************************************Factor de amplificación******************************************************')
    frequency_factor_ampl = str(math.ceil(frequency))
    if not Resultado_busqueda_X.empty:
        columna = Resultado_busqueda_X.get(frequency_factor_ampl)
        if columna is not None:
            factor_Ampl_X = Resultado_busqueda_X[frequency_factor_ampl]
            factor_Ampl_Y = Resultado_busqueda_Y[frequency_factor_ampl]
            factor_Ampl_Z = Resultado_busqueda_Z[frequency_factor_ampl]
            print('The found amplification factors are:'+ '\n'+ 'Este = '+ str(factor_Ampl_X) + '\n'+ 'Norte = '+str(factor_Ampl_Y) + '\n'+ 'Vertical = '+str(factor_Ampl_Z))

        else:
            print('Frequency ', frequency_factor_ampl,'not found. The amplification factors are 1')
            factor_Ampl_X=1
            factor_Ampl_Y=1
            factor_Ampl_Z=1
    else:
        print('Validity or seismological station not found, the amplification factors are 1')
        factor_Ampl_X=1
        factor_Ampl_Y=1
        factor_Ampl_Z=1
    print('**************************************************************************************************************************************')
    tr_filt_x.filter('bandpass', freqmin = frequency-0.1, freqmax = frequency+0.1, corners=4, zerophase=True)
    tr_filt_y.filter('bandpass', freqmin = frequency-0.1, freqmax = frequency+0.1, corners=4, zerophase=True)
    tr_filt_z.filter('bandpass', freqmin = frequency-0.1, freqmax = frequency+0.1, corners=4, zerophase=True)

    filtered_data_x = (factor_respuesta_X/factor_Ampl_X)*tr_filt_x.data  ##Revisar inge Roberto
    filtered_data_y = (factor_respuesta_Y/factor_Ampl_Y)*tr_filt_y.data
    filtered_data_z = (factor_respuesta_Z/factor_Ampl_Z)*tr_filt_z.data

    filtered_data_fourier_x = fft(filtered_data_x)
    filtered_data_fourier_y = fft(filtered_data_y)
    filtered_data_fourier_z = fft(filtered_data_z)

    filter_freq = fftfreq(N, datos_evento_z.dt)[:N//2]

    filter_spectr_x = 2.0/N * np.abs(filtered_data_fourier_x[0:N//2])
    filter_spectr_y = 2.0/N * np.abs(filtered_data_fourier_y[0:N//2])
    filter_spectr_z = 2.0/N * np.abs(filtered_data_fourier_z[0:N//2])

    filter_spectr_x=filter_spectr_x/filter_spectr_x[filter_spectr_x.argmax()]
    filter_spectr_y=filter_spectr_y/filter_spectr_y[filter_spectr_y.argmax()]
    filter_spectr_z=filter_spectr_z/filter_spectr_z[filter_spectr_z.argmax()]

    titleFig4 = datos_evento_z.titulo+'\n'+ 'sx trialxial filtered  around of '+ str(round(frequency,2)) + ' Hz'
    fig4 = plt.figure(figsize=(12,10), dpi=80) 
    plt.get_current_fig_manager().window.geometry("800x700+0+0")#"widthxheight+Xposition+Yposition"
    plt.subplot(321)
    plt.plot(time_vector, filtered_data_z, 'b', linewidth=0.5)
    plt.ylabel('Z [nm/s]')
    plt.xlim([0, N/datos_evento_z.fs])
    plt.subplot(322)
    plt.plot(filter_freq, filter_spectr_z, 'b', linewidth=0.5)
    plt.ylabel('Spec. Amp')
    plt.xlim(0,20)
    plt.ylim(0,1.2) 
    plt.subplot(323)
    plt.plot(time_vector, filtered_data_y, 'r', linewidth=0.5)
    plt.ylabel('N [nm/s]')
    plt.xlim([0, N/datos_evento_y.fs])
    plt.subplot(324)
    plt.plot(filter_freq, filter_spectr_y, 'r', linewidth=0.5)
    plt.ylabel('Spec. Amp')
    plt.xlim(0,20)
    plt.ylim(0,1.2) 
    plt.subplot(325)
    plt.plot(time_vector, filtered_data_x, 'k', linewidth=0.5)
    plt.ylabel('E [nm/s]')
    plt.xlabel('Time [s]')
    plt.xlim([0, N/datos_evento_x.fs])
    plt.subplot(326)
    plt.plot(filter_freq, filter_spectr_x, 'k', linewidth=0.5)
    plt.ylabel('Spec. Amp')
    plt.xlim(0,20)
    plt.ylim(0,1.2) 
    plt.xlabel('f [Hz]')
    fig4.suptitle(titleFig4, fontsize=12)
    plt.show(block=False)
else:
    processing = 'Full_spectrum'
    filtered_data_x = trace_x.data
    filtered_data_y = trace_y.data
    filtered_data_z = trace_z.data
    cadena_filt='NO_filter'
    frequency_doc_results =''
    frequency_fig=''
    T_period = ''
    unid=''
    f_str=''

#Grafica 3d centro
ej = Max_concatenated_data(filtered_data_x, filtered_data_y, filtered_data_z)

titleFig5 = datos_evento_z.titulo+'\n'+ 'Movimiento de particulas 3D'+'\n'+f_str+str(frequency_fig)+unid
fig51d = plt.figure(figsize=(12,10), dpi=80)
fig5 = fig51d.add_subplot(projection='3d') 
plt.get_current_fig_manager().window.geometry("800x700+0+0")#"widthxheight+Xposition+Yposition"
plt.plot(filtered_data_x, filtered_data_y, filtered_data_z,  'g', linewidth=0.5)
fig5.plot3D([-ej, ej], [0,0], [0,0],  'k', linewidth=0.7)
fig5.plot3D([0,0], [-ej, ej], [0,0],  'k', linewidth=0.7)
fig5.plot3D([0,0], [0,0], [-ej, ej], 'k', linewidth=0.7)
fig5.plot3D(0, 0, 0,  'o', markersize='5')
fig5.set_ylabel('Y (sur-norte)')
fig5.set_xlabel('X (oeste-este)')
fig5.set_zlabel('Z (vertical)')
fig5.set_title(titleFig5, fontsize=12)
fig5.view_init(30,-37.5)
plt.show(block=False)

empty_vector = []
max_este_norte = Max_concatenated_data(filtered_data_x, filtered_data_y, empty_vector)
titleFig6 = datos_evento_z.titulo+'\n'+ 'Movimiento de particulas 2D, Este - Norte'+'\n'+f_str+str(frequency_fig)+unid
fig6 = plt.figure(figsize=(12,10), dpi=80)
plt.get_current_fig_manager().window.geometry("800x700+850+0")#"widthxheight+Xposition+Yposition"
plt.plot(filtered_data_x, filtered_data_y,  'g', linewidth=0.5)
plt.plot([-ej, ej], [0,0],  'k', linewidth=0.7)
plt.plot([0,0], [-ej, ej],  'k', linewidth=0.7)
plt.plot(0, 0,  'o', markersize='5')
plt.ylabel('Y (sur-norte)')
plt.xlabel('X (oeste-este)')
plt.xlim(-max_este_norte,max_este_norte)
plt.ylim(-max_este_norte,max_este_norte) 
fig6.suptitle(titleFig6, fontsize=12)
plt.show(block=False)

#????????????????????????????????????????????????????????????????????????????????????????????????
if tipo_procesamiento!=4:
    print('The period of the filtered signal is: '+ str(T_period) + ' s')
ns = float(input('Enter window duration:'))
len_filtered_data_x =len(filtered_data_x)
npw = int(ns/datos_evento_z.dt)
numero_ventanas = math.floor(len_filtered_data_x/npw)
titleFig7_1 = datos_evento_z.titulo+'\n'+ 'Movimiento de particulas 2D, Este-Norte ' +f_str+str(frequency_fig)+unid + '\n' + 'Duración de la ventana '+str(int(ns)) + 's'
fig7_1 = plt.figure(figsize=(15, 15), dpi=100)
plt.get_current_fig_manager().window.geometry("800x700+850+0")#"widthxheight+Xposition+Yposition"
plot_windows(numero_ventanas,filtered_data_x,filtered_data_y,fig7_1,max_este_norte)
fig7_1.suptitle(titleFig7_1, fontsize=6)
plt.show(block=False)
##????????????????????????????????????????????????????????????????????????????????????????????????
#plt.show(block=False)
#????????????????????????????????????????????????????????????????????????????????????????????????

# Determinación de matriz de covarianza y vectores propios
triaxial_data = np.array([filtered_data_x, filtered_data_y, filtered_data_z])
cov_matrix = np.cov(triaxial_data)
eigenvalues,eigenvectors=np.linalg.eigh(cov_matrix)
sorted_eigenvalues_index = np.argsort(eigenvalues)
print('********************************************************************** Rectilinearidad y planaridad ***************************')
rectilinearidad = 1-(eigenvalues[sorted_eigenvalues_index[1]]/eigenvalues[sorted_eigenvalues_index[2]])
planaridad = 1-(2*eigenvalues[sorted_eigenvalues_index[0]]/(eigenvalues[sorted_eigenvalues_index[2]]+eigenvalues[sorted_eigenvalues_index[1]]))
print('rectinileradidad',rectilinearidad)
print('planaridad',planaridad)

print('*******************************************************************************************************************************')
azimut1 = math.acos(eigenvectors[1,sorted_eigenvalues_index[2]]/(math.sqrt(math.pow(eigenvectors[0,sorted_eigenvalues_index[2]],2)+math.pow(eigenvectors[1,sorted_eigenvalues_index[2]],2))))*(180/math.pi)
azimut2 = 180-azimut1
print('azimut',azimut1)
print('azimut2',azimut2)
azimut_selec = float(input('Defina el azimut en grados: '))
inclination = math.acos(eigenvectors[2,2])*(180/math.pi)
print('Inclinación: ',inclination)
radial,tang = rotate_ne_rt(filtered_data_y, filtered_data_x, azimut_selec)


titleFig7 = datos_evento_z.titulo+'\n'+ 'Movimiento de particulas 2D' +'\n'+f_str+str(frequency_fig)+unid
fig7 = plt.figure(figsize=(15,15), dpi=90)

positions = {
    'ax1': [0.1, 0.3, 0.35, 0.4],  # [left, bottom, width, height]
    'ax2': [0.55, 0.3, 0.35, 0.4]
}

plt.get_current_fig_manager().window.geometry("800x700+850+0")#"widthxheight+Xposition+Yposition"
#ax7 = plt.subplot(121)
ax7 = fig7.add_axes(positions['ax1'])
plt.plot(filtered_data_x, filtered_data_y,  'g', linewidth=0.5)
plt.plot([-ej, ej], [0,0],  'k', linewidth=0.7)
plt.plot([0,0], [-ej, ej],  'k', linewidth=0.7)
plt.plot(0, 0,  'o', markersize='5')
#ax7.set_aspect(1)
plt.ylabel('Y (sur-norte)',fontsize=9)
plt.xlabel('X (oeste-este)')
ax7.set_title('azimut= '+str(round(azimut_selec,3)))
#ax7_b = plt.subplot(122)
ax7_b = fig7.add_axes(positions['ax2'])
plt.plot(radial, filtered_data_z,  'g', linewidth=0.5)
plt.plot([-ej, ej], [0,0],  'k', linewidth=0.7)
plt.plot([0,0], [-ej, ej],  'k', linewidth=0.7)
plt.plot(0, 0,  'o', markersize='5')
plt.ylabel('Z (Vertical)', fontsize=9)
plt.xlabel('R (Radial)')
#ax7_b.set_aspect(1.35)
ax7_b.yaxis.tick_right()
ax7_b.yaxis.set_label_position("right")
ax7_b.set_title('inclination= '+str(round(inclination,3)))
fig7.subplots_adjust(hspace=10)
fig7.suptitle(titleFig7, fontsize=12)
plt.show(block=False)

l_rotate,q_rotate,tang2 = rotate_zne_lqt(filtered_data_z, filtered_data_y, filtered_data_x, azimut_selec, inclination)

titleFig8 = datos_evento_z.titulo+'\n'+ 'Movimiento de Particulas 3D, L-T-Q'+'\n'+ 'Rectilinearidad = '+ str(round(rectilinearidad,2))+', '+ 'Planaridad = '+ str(round(planaridad,2))+'\n'+f_str+str(frequency_fig)+unid
fig83d = plt.figure(figsize=(12,10), dpi=80)
fig8 = fig83d.add_subplot(projection='3d') 
plt.get_current_fig_manager().window.geometry("800x700+0+0")#"widthxheight+Xposition+Yposition"
plt.plot(l_rotate, tang2, q_rotate,   'r', linewidth=0.5)
fig8.plot3D([-ej, ej], [0,0], [0,0],  'k', linewidth=0.7)
fig8.plot3D([0,0], [-ej, ej], [0,0],  'k', linewidth=0.7)
fig8.plot3D([0,0], [0,0], [-ej, ej], 'k', linewidth=0.7)
fig8.plot3D(0, 0, 0,  'o', markersize='5')
fig8.set_ylabel('T (Tangencial)')
fig8.set_xlabel('L (Longitudinal)')
fig8.set_zlabel('Q (vertical)')
fig8.set_title(titleFig8, fontsize=12)
fig8.view_init(30,-37.5)
plt.show(block=False)

#################################################################################################
max_l_q_t = Max_concatenated_data(l_rotate,q_rotate,tang2)
imagen_fondo = plt.imread('lqtsystem.png')
titleFig9 = datos_evento_z.titulo+'\n'+ 'Movimiento de Particulas 2D, L-T-Q'+'\n'+ 'Rectilinearidad = '+ str(round(rectilinearidad,2))+', '+ 'Planaridad = '+ str(round(planaridad,2))+'\n'+f_str+str(frequency_fig)+unid
fig9 = plt.figure(figsize=(12,10), dpi=80) 
plt.get_current_fig_manager().window.geometry("800x700+0+0")#"widthxheight+Xposition+Yposition"
ax9_1= plt.subplot(221)
plt.plot(l_rotate, q_rotate,  'g', linewidth=0.5)
plt.plot([-ej, ej], [0,0],  'k', linewidth=0.7)
plt.plot([0,0], [-ej, ej],  'k', linewidth=0.7)
plt.plot(0, 0,  'o', markersize='5')
plt.ylabel('Q (vertical)')
plt.xlim(-max_l_q_t,max_l_q_t)
plt.ylim(-max_l_q_t,max_l_q_t)
plt.tight_layout()
ax9_1.set_title('Frontal',fontsize=9)
ax9_2 =plt.subplot(223)
plt.plot(l_rotate, tang2,  'g', linewidth=0.5)
plt.plot([-ej, ej], [0,0],  'k', linewidth=0.7)
plt.plot([0,0], [-ej, ej],  'k', linewidth=0.7)
plt.plot(0, 0,  'o', markersize='5')
plt.ylabel('T (Tangencial)')
plt.xlabel('L (Longitunal)')
plt.xlim(-max_l_q_t,max_l_q_t)
plt.ylim(-max_l_q_t,max_l_q_t)
plt.tight_layout()
ax9_2.set_title('Base',fontsize=9)
ax9_3=plt.subplot(222)
plt.plot(tang2, q_rotate,  'g', linewidth=0.5)
plt.plot([-ej, ej], [0,0],  'k', linewidth=0.7)
plt.plot([0,0], [-ej, ej],  'k', linewidth=0.7)
plt.plot(0, 0,  'o', markersize='5')
plt.xlabel('T (Tangencial)')
plt.xlim(-max_l_q_t,max_l_q_t)
plt.ylim(-max_l_q_t,max_l_q_t)
ax9_3.set_title('Lateral',fontsize=9)
ax9_4= plt.subplot(224)
plt.imshow(imagen_fondo, extent=[ -78.41, -77.335, 0.41, 0.971], aspect='auto',alpha=0.8)
ax9_4.axis("off")
fig9.suptitle(titleFig9, fontsize=12)
plt.tight_layout()
plt.show(block=False)



titleFig10_1 = datos_evento_z.titulo + '\n' + 'movimiento de particulas Q-L (Frontal), '+ f_str+str(frequency_fig)+unid + '\n' + 'Duración de la ventana '+ str(int(ns)) + 's'
fig10_1 = plt.figure(figsize=(15,15), dpi=100) 
plt.get_current_fig_manager().window.geometry("800x700+0+0")#"widthxheight+Xposition+Yposition"
plot_windows(numero_ventanas,l_rotate,q_rotate,fig10_1,max_l_q_t)
fig10_1.suptitle(titleFig10_1, fontsize=6)
plt.show(block=False)

max_q_t = Max_concatenated_data(tang2,q_rotate,empty_vector)
titleFig10_2 = datos_evento_z.titulo + '\n' + 'movimiento de particulas Q-T (Lateral), '+ f_str+str(frequency_fig)+unid + '\n' + 'Duración de la ventana '+ str(int(ns)) + 's'
fig10_2 = plt.figure(figsize=(15,15), dpi=100) 
plt.get_current_fig_manager().window.geometry("800x700+0+0")#"widthxheight+Xposition+Yposition"
plot_windows(numero_ventanas,tang2,q_rotate,fig10_2,max_l_q_t)
fig10_2.suptitle(titleFig10_2, fontsize=6)
plt.show(block=False)

max_q_t = Max_concatenated_data(tang2,q_rotate,empty_vector)
titleFig10_3 = datos_evento_z.titulo + '\n' + 'movimiento de particulas T-L (Base), '+ f_str+str(frequency_fig)+unid + '\n' + 'Duración de la ventana '+ str(int(ns)) + 's'
fig10_3 = plt.figure(figsize=(15,15), dpi=100) 
plt.get_current_fig_manager().window.geometry("800x700+0+0")#"widthxheight+Xposition+Yposition"
plot_windows(numero_ventanas,l_rotate,tang2,fig10_3,max_l_q_t)
fig10_3.suptitle(titleFig10_3, fontsize=6)
plt.show(block=False)
 
#################################################################################################
# Sismograma en los ejes del sistema de la onda (L,Q, T) 
triaxial_data_lqt = np.concatenate([l_rotate, q_rotate, tang2])
max_amplitud_rotate = triaxial_data_lqt[triaxial_data_lqt.argmax()]

titleFig10 = datos_evento_z.titulo + '\n' + 'Registro triaxial en velocidad L-Q-T'+'\n'+f_str+str(frequency_fig)+unid
fig10 = plt.figure(figsize=(12,10), dpi=70) 
plt.get_current_fig_manager().window.geometry("800x700+0+0")#"widthxheight+Xposition+Yposition"
plt.subplot(311)
plt.plot(time_vector, l_rotate, 'b',linewidth=0.5)
plt.ylabel('L [nm/s]')
plt.ylim(-1.2*max_amplitud_rotate,1.2*max_amplitud_rotate) 
plt.xlim(0,len(trace_z.data)/datos_evento_z.fs) 
plt.subplot(312)
plt.plot(time_vector, q_rotate, 'r',linewidth=0.5)
plt.ylabel('Q [nm/s]')
plt.ylim(-1.2*max_amplitud_rotate,1.2*max_amplitud_rotate) 
plt.xlim(0,len(trace_z.data)/datos_evento_z.fs) 
plt.subplot(313)
plt.plot(time_vector, tang2, 'k',linewidth=0.5)
plt.ylabel('T [n/s]')
plt.xlabel('tiempo [s]')
plt.ylim(-1.2*max_amplitud_rotate,1.2*max_amplitud_rotate) 
plt.xlim(0,len(trace_z.data)/datos_evento_z.fs) 
fig10.suptitle(titleFig10, fontsize=12)
plt.show(block=False)

#('********************* vectores propios de la matriz de covarianza ***************************')
filtered_data_z, filtered_data_y, 
X1 = filtered_data_x*eigenvectors[0,sorted_eigenvalues_index[2]]+filtered_data_y*eigenvectors[1,sorted_eigenvalues_index[2]]+filtered_data_z*eigenvectors[2,sorted_eigenvalues_index[2]]
X2 = filtered_data_x*eigenvectors[0,sorted_eigenvalues_index[1]]+filtered_data_y*eigenvectors[1,sorted_eigenvalues_index[1]]+filtered_data_z*eigenvectors[2,sorted_eigenvalues_index[1]]
X3 = filtered_data_x*eigenvectors[0,sorted_eigenvalues_index[0]]+filtered_data_y*eigenvectors[1,sorted_eigenvalues_index[0]]+filtered_data_z*eigenvectors[2,sorted_eigenvalues_index[0]]


titleFig11 = datos_evento_z.titulo+'\n'+ 'Mov partículas 3D - sistema vectores propios'+'\n'+f_str+str(frequency_fig)+unid
fig11 = plt.figure(figsize=(12,10), dpi=80).add_subplot(projection='3d') 
plt.get_current_fig_manager().window.geometry("800x700+0+0")#"widthxheight+Xposition+Yposition"
plt.plot(X1, X2, X3,  'g', linewidth=0.5)
fig11.plot3D([-ej, ej], [0,0], [0,0],  'k', linewidth=0.7)
fig11.plot3D([0,0], [-ej, ej], [0,0],  'k', linewidth=0.7)
fig11.plot3D([0,0], [0,0], [-ej, ej], 'k', linewidth=0.7)
fig11.plot3D(0, 0, 0,  'o', markersize='5')
fig11.set_ylabel('X2')
fig11.set_xlabel('X1')
fig11.set_zlabel('X3')
fig11.set_title(titleFig11, fontsize=12)
fig11.view_init(30,-37.5)
plt.show(block=False)

titleFig12 = datos_evento_z.titulo + '\n' + 'Registro Triaxial En Velocidad (Sistema Propio)'+'\n'+f_str+str(frequency_fig)+unid
fig12 = plt.figure(figsize=(12,10), dpi=70) 
plt.get_current_fig_manager().window.geometry("800x700+0+0")#"widthxheight+Xposition+Yposition"
plt.subplot(311)
plt.plot(time_vector, X1, 'b',linewidth=0.5)
plt.ylabel('X1 [nm/s]')
plt.ylim(-1.2*max_amplitud_rotate,1.2*max_amplitud_rotate) 
plt.xlim(0,len(trace_z.data)/datos_evento_z.fs) 
plt.subplot(312)
plt.plot(time_vector, X2, 'r',linewidth=0.5)
plt.ylabel('X2 [nm/s]')
plt.ylim(-1.2*max_amplitud_rotate,1.2*max_amplitud_rotate) 
plt.xlim(0,len(trace_z.data)/datos_evento_z.fs) 
plt.subplot(313)
plt.plot(time_vector, X3, 'k',linewidth=0.5)
plt.ylabel('X3 [n/s]')
plt.xlabel('tiempo [s]')
plt.ylim(-1.2*max_amplitud_rotate,1.2*max_amplitud_rotate) 
plt.xlim(0,len(trace_z.data)/datos_evento_z.fs) 
fig12.suptitle(titleFig12, fontsize=12)
plt.show(block=False)

#################################################################################################
titleFig13 = datos_evento_z.titulo+'\n'+ 'Movimiento de Particulas 2D, X1-X2-X3'+'\n'+ 'Rectilinearidad = '+ str(round(rectilinearidad,2))+', '+ 'Planaridad = '+ str(round(planaridad,2))+'\n'+f_str+str(frequency_fig)+unid
fig13 = plt.figure(figsize=(12,10), dpi=80) 
plt.get_current_fig_manager().window.geometry("800x700+850+0")#"widthxheight+Xposition+Yposition"
ax13_1= plt.subplot(221)
plt.plot(X1, X3,  'r', linewidth=0.5)
plt.plot([-ej, ej], [0,0],  'k', linewidth=0.7)
plt.plot([0,0], [-ej, ej],  'k', linewidth=0.7)
plt.plot(0, 0,  'o', markersize='5')
plt.ylabel('X3')
#ax13_1.set_title('Frontal',fontsize=9)
ax13_2 =plt.subplot(223)
plt.plot(X1, X2,  'r', linewidth=0.5)
plt.plot([-ej, ej], [0,0],  'k', linewidth=0.7)
plt.plot([0,0], [-ej, ej],  'k', linewidth=0.7)
plt.plot(0, 0,  'o', markersize='5')
plt.ylabel('X2')
plt.xlabel('X1')
#ax13_2.set_title('Base',fontsize=9)
ax13_3=plt.subplot(222)
plt.plot(X2, X3,  'r', linewidth=0.5)
plt.plot([-ej, ej], [0,0],  'k', linewidth=0.7)
plt.plot([0,0], [-ej, ej],  'k', linewidth=0.7)
plt.plot(0, 0,  'o', markersize='5')
plt.xlabel('X2')
#ax13_3.set_title('Lateral',fontsize=9)
ax13_4= plt.subplot(224)
ax13_4.axis("off")
fig13.suptitle(titleFig13, fontsize=12)
plt.show(block=False)

if tipo_procesamiento!=4:
    print('****************************************************Transformada de Hilbert******************************************************')
    analytic_signal_C = hilbert(X1)
    log_envol = np.array(list(map(lambda analytic_signal_C : math.log(abs(analytic_signal_C)),analytic_signal_C))) 


    fig14 = plt.figure(figsize=(12,10), dpi=80) 
    plt.get_current_fig_manager().window.geometry("800x700+850+0")#"widthxheight+Xposition+Yposition"
    plt.plot(time_vector,log_envol, 'r')
    plt.ylabel('Log(Amp_envelope(X1))')
    plt.xlabel('Time [s]')
    tellme('Click on the graph')
    plt.waitforbuttonpress()
    while True:
        pts = []
        while len(pts) < 2:
            tellme('Select start point and end point in the linear part of the decay')
            pts = np.asarray(plt.ginput(2, timeout=-1))  
        pt1 = pts[0,:]
        pt2 = pts[1,:]
        index_max_value_t1 = find_nearest2(time_vector,log_envol, pt1[0],pt1[1])
        index_max_value_t2 = find_nearest2(time_vector,log_envol, pt2[0],pt2[1])
        aH, bH = np.polyfit(time_vector[index_max_value_t1:index_max_value_t2], log_envol[index_max_value_t1:index_max_value_t2], deg=1)
        y_est = aH * time_vector[index_max_value_t1:index_max_value_t2] + bH
        plt.plot(time_vector[index_max_value_t1:index_max_value_t2],y_est, 'g')
        tellme('Correct? \n' 'YES------------>Enter  \n' 'NO------------>Mouse click')
        if plt.waitforbuttonpress():
            break
        plt.cla()
        plt.plot(t,log_envol, 'r')
    plt.show
    index_max_t_init= find_max(k,index_max_value_t1,N,X1)
    A_Inicial = X1[index_max_t_init]
    t_Inicial = time_vector[index_max_t_init]
    index_max_t_end= find_max(k,index_max_value_t2,N,X1)
    A_final = X1[index_max_t_end]
    t_final = time_vector[index_max_t_end]
    aH_aux, bH_aux = np.polyfit(time_vector[index_max_t_init:index_max_t_end], log_envol[index_max_t_init:index_max_t_end], deg=1)
    y_est_final = aH_aux * time_vector[index_max_t_init:index_max_t_end] + bH_aux
    q = abs(aH_aux)
    Q_Hilbert = math.pi*frequency/q
    t_0 = time_vector[index_max_t_init]
    t_exponencial = time_vector-t_0
    t_exponencial_new = t_exponencial[index_max_t_init:index_max_t_end]
    sx_filt = X1[index_max_t_init:index_max_t_end] 
    Acos_Hilbert = calcular_Coseno_metodo(A_Inicial,frequency,Q_Hilbert,t_exponencial_new)
    res_Hilbert = calcular_residual(sx_filt,Acos_Hilbert)
    h_hilbert =1/(2*Q_Hilbert)    

    plt.clf()
    plt.get_current_fig_manager().window.geometry("800x700+0+0")#"widthxheight+Xposition+Yposition"
    plt.subplot(211)
    plt.plot(time_vector, X1, 'b', linewidth=0.5, label='O sx, '+ 'f='+ str(round(frequency,2))+ ' Hz')
    plt.ylabel('X1 [nm/s]')
    plt.xlabel('Time [s]')
    plt.xlim([0, len(X1)/datos_evento_z.fs])
    plt.plot(time_vector[index_max_t_init:index_max_t_end], Acos_Hilbert, 'r',linestyle='dashed',linewidth=0.5,label='C sx, '+ 'Q ='+ str(int(Q_Hilbert))+ ', ' +'h='+ str(round(h_hilbert,4)))
    fig14.suptitle(datos_evento_z.titulo+'\n'+'Envelope decay using Hilbert transform', fontsize=12)
    plt.legend(loc='upper right')
    plt.subplot(212)
    plt.plot(time_vector,log_envol, 'r', label='Log(Envelope)')
    plt.plot(time_vector[index_max_t_init:index_max_t_end], y_est_final, 'g', label='Polynomial curve fitting')
    plt.ylabel('Log(Amp_envelope(X1))')
    plt.xlabel('Time [s]')
    plt.legend(loc='upper right')
    plt.show(block=False)

    fig15 = plt.figure(figsize=(12,10), dpi=80) 
    plt.get_current_fig_manager().window.geometry("800x700+450+300")#"widthxheight+Xposition+Yposition"
    ax15_1= plt.subplot(222)
    plt.plot(time_vector,log_envol, 'r', label='Log(Envelope)')
    plt.plot(time_vector[index_max_t_init:index_max_t_end], y_est_final, 'g', label='Polynomial curve fitting')
    plt.ylabel('Log(Amp_envelope(X1))')
    plt.xlabel('Time [s]')
    plt.legend(loc='upper right', fontsize=8)
    ax15_1.set_title('f='+ str(round(frequency,2))+ ' Hz,'+' Q='+ str(round(Q_Hilbert))+','+' h='+ str(round(h_hilbert,4)), fontsize=16)
    plt.subplot(223)
    plt.specgram(X1, NFFT=512, Fs=datos_evento_z.fs, cmap = 'jet')
    plt.xlabel('Time [s]')
    plt.ylabel('frequency [Hz]')
    plt.ylim(0,20)
    ax15_2=plt.subplot(221)
    plt.plot(time_vector, X1, 'b', linewidth=0.5, label='O sx')
    plt.plot(time_vector[index_max_t_init:index_max_t_end], Acos_Hilbert, 'r',linestyle='dashed',linewidth=0.5,label='C sx')
    plt.xlabel('Time [s]')
    plt.xlim([0, len(X1)/datos_evento_z.fs])
    plt.legend(loc='upper right', fontsize=8)
    ax15_2.set_title(datos_evento_z.titulo, fontsize=16)
    plt.subplot(224)
    for i in pt_aux:
        plt.axhline(freq_x[i], color = "b", linewidth = 0.8, linestyle = "dashed") 
        plt.annotate(str(round(freq_x[i],2)), (stacked_espectral[i],freq_x[i]))
    plt.plot(stacked_espectral, freq_x, 'r',linewidth=0.5)
    plt.ylim(0,20)
    plt.xlim([0,stacked_espectral[stacked_espectral.argmax()]+0.5]) 
    plt.xlabel('Spec. Amp [counts/Hz]')  

#*******************************************save all data*******************************************************************
fig1.savefig(save_fig_title+'fig_1_'+'sx_observada_triaxial_'+datos_evento_z.senal+'_'+estacion+'.png') #'.svg'
fig2.savefig(save_fig_title+'fig_2_'+'sx_triaxial_spectrum_'+datos_evento_z.senal+'_'+estacion+'.png') #'.svg'
fig3.savefig(save_fig_title+'fig_3_'+'picked_freq_'+datos_evento_z.senal+'_'+estacion+'.png')
fig7.savefig(save_fig_title+'fig_4_'+'movitopar_2D_'+datos_evento_z.senal+'_'+estacion+'_'+cadena_filt+'.png')
fig83d.savefig(save_fig_title+'fig_5_'+'movitopar_LQT_3D_'+datos_evento_z.senal+'_'+estacion+'_'+cadena_filt+'.png')
fig9.savefig(save_fig_title+'fig_6_'+'movitopar_LQT_2D_'+datos_evento_z.senal+'-'+estacion+'_'+cadena_filt+'.png')
fig12.savefig(save_fig_title+'fig_7_'+'sistema_propio_triaxial_'+datos_evento_z.senal+'_'+estacion+'-'+cadena_filt+'.png')
fig13.savefig(save_fig_title+'fig_8_'+'movitopar_X1X2X3_2D_'+datos_evento_z.senal+'-'+estacion+'_'+cadena_filt+'.png')
fig7_1.savefig(save_fig_title+'fig_9_'+'movitopar_2D_window_'+datos_evento_z.senal+'_'+estacion+'_'+cadena_filt+'.png')

fig10_1.savefig(save_fig_title+'fig_10_'+'movitopar_2D_window_QL_'+datos_evento_z.senal+'_'+estacion+'-'+cadena_filt+'.png')
fig10_2.savefig(save_fig_title+'fig_11_'+'movitopar_2D_window_QT_'+datos_evento_z.senal+'-'+estacion+'_'+cadena_filt+'.png')
fig10_3.savefig(save_fig_title+'fig_12_'+'movitopar_2D_window_TL_'+datos_evento_z.senal+'_'+estacion+'_'+cadena_filt+'.png')
if tipo_procesamiento!=4:
    fig4.savefig(save_fig_title+'fig_13_'+'sx_triaxial_'+datos_evento_z.senal+'_'+estacion+'_'+cadena_filt+'.png')
    fig14.savefig(save_fig_title+'fig_14_Hilbert_'+datos_evento_z.senal+'_'+estacion+'_'+cadena_filt+'.png')
    fig15.savefig(save_fig_title+'fig_15_mosaic_'+datos_evento_z.senal+'_'+estacion+'_'+cadena_filt+'.png')

file_results = open(data_out+'results_'+datos_evento_z.senal+'_'+estacion+'_'+ cadena_filt+'.txt', 'w')
file_head = [datos_evento_z.senal, estacion, cadena_filt,'\n'] 
file_results.writelines(file_head)
file_results.writelines('picked frequencies:'+ str(np.round(freq_x[pt_aux],2))+'\n')
file_results.writelines('processing: '+ processing + ' with frequency: '+ cadena_filt +'\n')
file_results.writelines('Azimut= '+ str(round(azimut_selec,4))+'\n')
file_results.writelines('Inclination= '+ str(round(inclination,4))+'\n')
file_results.writelines('Rectilinearidad = '+ str(round(rectilinearidad,2))+'\n')
file_results.writelines('Planaridad= '+ str(round(planaridad,2))+'\n')
file_results.writelines('Duración de la ventana= '+ str(int(ns))+ 's'+'\n')
if recortar == 'y':
    file_results.writelines('Start time='+ str(t_init)+'\n')
    file_results.writelines('End time='+ str(t_end)+'\n')
if tipo_procesamiento!=4:
    file_results.writelines('Q Hilbert = '+ str(round(Q_Hilbert))+'\n')
    file_results.writelines('h Hilbert='+ str(round(h_hilbert,4))+'\n')
    file_results.writelines('r Hilbert ='+ str(round(res_Hilbert,4))+'\n')
    cadena_Hilbert = str(round(Q_Hilbert))+ ' '+ str(round(h_hilbert,4))+ ' '+ str(round(res_Hilbert,4))
else:
    cadena_Hilbert=''
fecha_evento = str(datos_evento_z.senal)
tiempo_evento=(str(fecha_evento[0:4])+'/'+str(fecha_evento[4:6])+'/'+str(fecha_evento[6:8])+' '+str(fecha_evento[8:10])+':'+str(fecha_evento[10:12]))
file_results.writelines(estacion +': '+ datos_evento_z.senal+ ' ' + tiempo_evento + ' ' + str(T_period) +' '+ str(frequency_doc_results)+' '+cadena_Hilbert+'\n')
file_results.close()

file_results_total = open(data_out+'results_total_'+datos_evento_z.senal+'.txt', 'a')
file_results_total.writelines(estacion +': '+ datos_evento_z.senal+ ' ' + tiempo_evento + ' ' + str(T_period) +' '+ str(frequency_doc_results)+' '+cadena_Hilbert+'\n')
file_results_total.close()

if not os.path.isfile(data_out+'results_'+datos_evento_z.senal+'_'+'Azimuth'+'.txt'):
    file_Azimuth = open(data_out+'results_'+datos_evento_z.senal+'_'+'Azimuth'+'.txt', 'a')
    file_Azimuth.writelines('Estacion;lat;lon;angulo'+'\n')
    file_Azimuth.close()

file_Azimuth = open(data_out+'results_'+datos_evento_z.senal+'_'+'Azimuth'+'.txt', 'a')
file_Azimuth.writelines(estacion+';'+str(Latitud)+';'+str(Longitud)+';' + str(azimut_selec)+';' + str(round(rectilinearidad,2)) +'\n')
file_Azimuth.close()

print('**************************************************************** Fin **************************************************************')
plt.show(block=False)
close = input('Press c if you want to close all windows ')
if  close == 'c':
    plt.close('all')




