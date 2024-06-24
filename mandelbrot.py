import numpy as np
from numba import jit, prange

@jit(nopython=True)
def mandelbrot(c: complex, max_iter: int) -> int:

    """
    Vypočet počtu iterácií pre daný komplexný bod v Mandelbrotovej množine.

    args:
        c (complex): Počiatočný komplexný bod.
        max_iter (int): Maximálny počet iterácií.

    return:
        int: Počet iterácií n alebo maximálny počet iterácii max_iter.
    """

    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter

@jit(nopython=True, parallel=True)
def generate_mandelbrot_set(xmin: float, xmax: float, ymin: float, ymax: float, width: int, height: int, max_iter: int) -> np.ndarray:

    """
    Generovanie Mandelbrotovej množiny pre dané parametre.

    args:
        xmin (float): Minimálna hodnota na x-ovej osi.
        xmax (float): Maximálna hodnota na x-ovej osi.
        ymin (float): Minimálna hodnota na y-ovej osi.
        ymax (float): Maximálna hodnota na y-ovej osi.
        width (int): Šírka obrazu v pixeloch.
        height (int): Výška obrazu v pixeloch.
        max_iter (int): Maximálny počet iterácií.

    return:
        np.ndarray: 2D pole obsahujúce počet iterácií pre každý bod - vygenerovaná Mandelbrotova množina.
    """

    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    mandelbrot_set = np.empty((width, height), dtype=np.uint32)

    for i in prange(width):
        for j in prange(height):
            mandelbrot_set[i, j] = mandelbrot(r1[i] + 1j * r2[j], max_iter)
    
    return mandelbrot_set

class Mandelbrot:
    def __init__(self, xmin, xmax, ymin, ymax, width, height, max_iter):

        """
        Inicializácia parametrov pre generovanie Mandelbrotovej množiny.

        args:
            xmin (float): Minimálna hodnota na x-ovej osi.
            xmax (float): Maximálna hodnota na x-ovej osi.
            ymin (float): Minimálna hodnota na y-ovej osi.
            ymax (float): Maximálna hodnota na y-ovej osi.
            width (int): Šírka obrazu v pixeloch.
            height (int): Výška obrazu v pixeloch.
            max_iter (int): Maximálny počet iterácií.
        """

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        self.max_iter = max_iter

    def generate(self) -> np.ndarray:   

        """
        Generovanie Mandelbrotovej množiny na základe aktuálnych parametrov.

        return:
            np.ndarray: 2D pole obsahujúce počet iterácií pre každý bod - vygenerovaná Mandelbrotova množina.
        """

        return generate_mandelbrot_set(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter)
