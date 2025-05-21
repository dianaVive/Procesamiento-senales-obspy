from core.EventoSismico import EventoSismico
from core.Graficador import GraficadorSismico
import matplotlib.pyplot as plt


evento = EventoSismico()
evento.cargar_componentes()
graficador = GraficadorSismico(evento)
graficador.graficar_trazas(aplicar_filtro=False)
plt.show(block=False)
filtro_cp = input("Apply short-period filter? (y/n): ")
if filtro_cp.lower() == 'y':
   aplicar_filtro=True
   graficador.graficar_trazas(aplicar_filtro)
else:
   aplicar_filtro=False 
plt.show(block=False)
   
recortar = input("Trim the signal? (y/n): ")
if recortar.lower() == 'y':
   t_init = int(input('Initial time (s): '))
   t_end = int(input('Final time (s): '))
   recortar=True
   graficador.graficar_trazas(aplicar_filtro,recortar,t_init,t_end) 
plt.show(block=False)

graficador.graficar_espectros()
plt.show(block=False)


graficador.picar_frecuencias_espectro(graficador.freqs, graficador.stacked_espectral)
plt.show(block=False)

print('---------------------------')
print('Selected frequencies: ')
print(", ".join(f"{float(f):.2f}" for f in graficador.frecuencias_pickeadas))
close = input('Press 'c' to close all windows ')
if  close == 'c':
    plt.close('all')


