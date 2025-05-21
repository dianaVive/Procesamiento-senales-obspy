from obspy import read, Stream, Trace, UTCDateTime
import numpy as np
from itertools import islice
from pathlib import Path

class FileLoader:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.fs = None
        self.dt = None
        self.fn = None
        self.data_long = None
        self.station = None
        self.component = None
        self.senal = None
        self.titulo = None
        self.starttime = None
        self.st = None  # Stream de ObsPy

    def load_ascii(self):
        lines_number_head = 5
        with open(self.file_path, "r") as f:
            head = list(islice(f, lines_number_head))
            data = np.genfromtxt(f, dtype=None, encoding=None, usecols=(0))

        # Parse header
        mmddHHSS_texto = head[0].strip()
        station_channel = head[1].strip()
        fecha_hora = list(map(str.strip, head[2].split()))
        fs_line = list(map(str.strip, head[3].split()))

        fecha_texto = fecha_hora[0]
        hora_texto = fecha_hora[1]
        fs_texto = fs_line[0]

        self.component = station_channel[-1]
        self.station = station_channel[:3]

        yyyy = fecha_texto[0:4]
        mm = fecha_texto[5:7]
        dd = fecha_texto[8:10]
        HH = hora_texto[0:2]
        MM = hora_texto[3:5]

        self.senal = f"{yyyy}{mm}{dd}{HH}{MM}"
        self.titulo = f"{self.senal} {self.station}"
        self.fs = float(fs_texto)
        self.dt = 1 / self.fs
        self.fn = self.fs / 2
        self.data_long = len(data)

        stats = {
            'network': 'OP',
            'station': self.station,
            'location': '',
            'channel': '',
            'npts': len(data),
            'sampling_rate': self.fs,
            'mseed': {'dataquality': 'D'},
            'starttime': UTCDateTime()
        }

        self.starttime = stats['starttime']
        self.st = Stream([Trace(data=data, header=stats)])

    def load_mseed(self):
        self.st = read(str(self.file_path), format="mseed")
        trace = self.st[0]

        self.fs = trace.stats.sampling_rate
        self.data_long = trace.stats.npts
        self.component = trace.stats.channel
        self.station = trace.stats.station + self.component[2]
        self.starttime = trace.stats.starttime

        fecha = str(self.starttime)[0:16].replace('-', '').replace('T', '').replace(':', '')
        self.senal = fecha
        self.titulo = f"{fecha} {trace.stats.station}"
        self.dt = 1 / self.fs
        self.fn = self.fs / 2