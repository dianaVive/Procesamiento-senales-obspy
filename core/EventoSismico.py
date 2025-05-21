from tkinter import filedialog
from core.file_loader import FileLoader

class EventoSismico:
    def __init__(self):
        self.componentes = {}

    def cargar_componentes(self):
        print('Select the X component (West–East)')
        x_file = filedialog.askopenfilename()
        print('Select the Y component (South–North)')
        y_file = filedialog.askopenfilename()
        print('Select the Z component (Vertical)')
        z_file = filedialog.askopenfilename()

        archivos = {'X': x_file, 'Y': y_file, 'Z': z_file}

        for comp, path in archivos.items():
            lector = FileLoader(path)
            if path.endswith('.txt'):
                lector.load_ascii()
            elif path.endswith('.mseed'):
                lector.load_mseed()
            else:
                raise ValueError(f'Unsupported file format: {path}')
            self.componentes[comp] = lector

    def get_traza(self, componente):
        return self.componentes[componente].st[0]