import numpy as np
from numba import jit, prange

@jit(nopython=True)
def julia(z: complex, c: complex, max_iter: int) -> int:

    """
    Vypočet počtu iterácií pre daný komplexný bod v Juliovej množine.

    args:
        z (complex): Počiatočný komplexný bod.
        c (complex): Konštanta c v Juliovej množine.
        max_iter (int): Maximálny počet iterácií.

    return:
        int: Počet iterácií n alebo maximálny počet iterácii max_iter.
    """

    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter

@jit(nopython=True, parallel=True)
def generate_julia_set(xmin: float, xmax: float, ymin: float, ymax: float, width: int, height: int, c: complex, max_iter: int) -> np.ndarray:

    """
    Generovanie Juliovej množiny pre dané parametre.

    args:
        xmin (float): Minimálna hodnota na x-ovej osi.
        xmax (float): Maximálna hodnota na x-ovej osi.
        ymin (float): Minimálna hodnota na y-ovej osi.
        ymax (float): Maximálna hodnota na y-ovej osi.
        width (int): Šírka obrazu v pixeloch.
        height (int): Výška obrazu v pixeloch.
        c (complex): Konštanta c v Juliovej množine.
        max_iter (int): Maximálny počet iterácií.

    return:
        np.ndarray: 2D pole obsahujúce počet iterácií pre každý bod - vygenerovaná Juliova množina.
    """

    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    julia_set = np.empty((width, height), dtype=np.uint32)

    for i in prange(width):
        for j in prange(height):
            julia_set[i, j] = julia(r1[i] + 1j * r2[j], c, max_iter)
    
    return julia_set

class Julia:
    def __init__(self, xmin, xmax, ymin, ymax, width, height, c, max_iter):

        """
        Inicializácia parametrov pre generovanie Juliovej množiny.

        args:
            xmin (float): Minimálna hodnota na x-ovej osi.
            xmax (float): Maximálna hodnota na x-ovej osi.
            ymin (float): Minimálna hodnota na y-ovej osi.
            ymax (float): Maximálna hodnota na y-ovej osi.
            width (int): Šírka obrazu v pixeloch.
            height (int): Výška obrazu v pixeloch.
            c (complex): Konštanta c v Juliovej množine.
            max_iter (int): Maximálny počet iterácií.
        """

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        self.c = c
        self.max_iter = max_iter
    
    def generate(self) -> np.ndarray:

        """
        Generovanie Juliovej množiny na základe aktuálnych parametrov.

        return:
            np.ndarray: 2D pole obsahujúce počet iterácií pre každý bod - vygenerovaná Juliova množina.
        """

        return generate_julia_set(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.c, self.max_iter)