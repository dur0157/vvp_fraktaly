import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pygame

class FractalVisualizer:

    """
    Trieda na vizualizáciu fraktálov pomocou Matplotlib a Pygame.
    """

    def __init__(self, xmin, xmax, ymin, ymax, width, height, max_iter):

        """
        Inicializácia parametrov pre vizualizáciu fraktálov

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

    def visualize_fractal(self, data: np.ndarray, title: str, color_map: str) -> pygame.Surface:

        """
        Vizualizácia fraktálových dát ako Pygame povrch.

        args:
            data (np.ndarray): 2D pole fraktálových dát.
            title (str): Názov fraktálu.
            color_map (str): Farebná mapa použitá na vizualizáciu.

        return:
            pygame.Surface: Pygame povrch s vizualizovaným fraktálom.
        """

        norm = Normalize(vmin=0, vmax=self.max_iter)
        mapper = ScalarMappable(norm=norm, cmap=color_map)
        colors = mapper.to_rgba(data.T, bytes=True)
        surface = pygame.surfarray.make_surface(colors[:, :, :3])
        return surface

    def show_fractal(self, data: np.ndarray, title: str, color_map: str):

        """
        Zobrazenie fraktálových dát pomocou Matplotlib.

        args:
            data (np.ndarray): 2D pole fraktálových dát.
            title (str): Názov fraktálu.
            color_map (str): Farebná mapa použitá na vizualizáciu.
        """

        plt.figure(figsize=(10, 10))
        plt.imshow(data.T, extent=[self.xmin, self.xmax, self.ymin, self.ymax], cmap=color_map)
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.show()
