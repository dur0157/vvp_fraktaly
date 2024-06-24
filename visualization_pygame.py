import pygame
import numpy as np
from julia import Julia
from mandelbrot import Mandelbrot
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

class FractalVisualizerPygame:
    """
    Trieda na vizualizáciu Mandelbrotovej a Juliovej množiny pomocou Pygame.

    atributes:
        xmin (float): Minimálna hodnota na osi x.
        xmax (float): Maximálna hodnota na osi x.
        ymin (float): Minimálna hodnota na osi y.
        ymax (float): Maximálna hodnota na osi y.
        width (int): Šírka obrázka v pixeloch.
        height (int): Výška obrázka v pixeloch.
        max_iter (int): Maximálny počet iterácií.
        zoom_factor (float): Faktor priblíženia pre zoomovanie.
        color_maps (list): Zoznam farebných máp.
        color_map_index (int): Index aktuálnej farebnej mapy.
        color_map (str): Aktuálna farebná mapa.
        current_set (str): Aktuálna množina (mandelbrot alebo julia).
        step_size (float): Veľkosť kroku pre posúvanie.
        screen (pygame.Surface): Pygame povrch pre zobrazenie fraktálu.
        clock (pygame.time.Clock): Pygame hodiny na určovanie FPS.
        running (bool): Hodnoty True alebo False ktoré indikujúci, či je pygame spustené.
    """

    def __init__(self, xmin, xmax, ymin, ymax, width, height, max_iter):
        """
        Inicializácia triedy FractalVisualizerPygame.

        args:
            xmin (float): Minimálna hodnota na osi x.
            xmax (float): Maximálna hodnota na osi x.
            ymin (float): Minimálna hodnota na osi y.
            ymax (float): Maximálna hodnota na osi y.
            width (int): Šírka obrázka v pixeloch.
            height (int): Výška obrázka v pixeloch.
            max_iter (int): Maximálny počet iterácií.
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.zoom_factor = 1.1
        self.color_maps = ['inferno', 'viridis', 'plasma', 'magma', 'cividis']
        self.color_map_index = 0
        self.color_map = self.color_maps[self.color_map_index]
        self.current_set = "mandelbrot"
        self.step_size = 0.1
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.running = True

    def visualize_fractal(self, data: np.ndarray, title: str) -> pygame.Surface:
        """
        Vizualizácia fraktálových dát ako Pygame povrch.
        """
        norm = Normalize(vmin=0, vmax=self.max_iter)
        mapper = ScalarMappable(norm=norm, cmap=self.color_map)
        colors = mapper.to_rgba(data.T, bytes=True)
        surface = pygame.surfarray.make_surface(colors[:, :, :3])
        return surface

    def run(self, mandelbrot_set, julia_set, julia_params):
        """
        Spúšťanie vizualizačnej slučky fraktálov.

        args:
            mandelbrot_set (np.ndarray): 2D pole dát Mandelbrotovej množiny.
            julia_set (np.ndarray): 2D pole dát Juliovej množiny.
            julia_params (dict): Parametre pre generovanie Juliovej množiny.
        """
        fractal_surface = self.visualize_fractal(mandelbrot_set, 'Mandelbrotova množina')

        while self.running:
            for event in pygame.event.get():
                # Vypnutie pygame okna
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    # Ďalšia farebná mapa
                    if event.key == pygame.K_RIGHT:
                        self.color_map_index = (self.color_map_index + 1) % len(self.color_maps)
                        self.color_map = self.color_maps[self.color_map_index]
                    # Predošlá farebná mapa
                    elif event.key == pygame.K_LEFT:
                        self.color_map_index = (self.color_map_index - 1) % len(self.color_maps)
                        self.color_map = self.color_maps[self.color_map_index]
                    # Zmena aktuálneho fraktálu na Juliovu množinu
                    elif event.key == pygame.K_j:
                        self.current_set = "julia"
                    # Zmena aktuálneho fraktálu na Mandelbrotovu množinu
                    elif event.key == pygame.K_m:
                        self.current_set = "mandelbrot"
                    # Zväčšovanie komplexného čísla v Juliovej množine
                    elif event.key == pygame.K_d:
                        if self.current_set == "julia":
                            julia_params["c"] *= 1.05
                    # Zmenšovanie komplexného čísla v Juliovej množine
                    elif event.key == pygame.K_a:
                        if self.current_set == "julia":
                            julia_params["c"] /= 1.05
                    # Zväšovanie počtu iterácii 
                    elif event.key == pygame.K_w:
                        self.max_iter += 50
                    # Zmenšovanie počtu iterácii
                    elif event.key == pygame.K_s:
                        self.max_iter -= 50
                        if self.max_iter < 1:
                            self.max_iter = 1
                    # Pohyb po obrazovke nahor
                    elif event.key == pygame.K_t:
                        self.xmin -= self.step_size / self.zoom_factor
                        self.xmax -= self.step_size / self.zoom_factor
                    # Pohyb po obrazovke nadol
                    elif event.key == pygame.K_g:
                        self.xmin += self.step_size / self.zoom_factor
                        self.xmax += self.step_size / self.zoom_factor
                    # Pohyb po obrazovke vľavo
                    elif event.key == pygame.K_f:
                        self.ymin -= self.step_size / self.zoom_factor
                        self.ymax -= self.step_size / self.zoom_factor
                    # Pohyb po obrazovke vpravo
                    elif event.key == pygame.K_h:
                        self.ymin += self.step_size / self.zoom_factor
                        self.ymax += self.step_size / self.zoom_factor
                    # Oddialenie obrazovky
                    elif event.key == pygame.K_q:
                        self.xmin = (self.xmin + self.xmax) / 2 + (self.xmin - (self.xmin + self.xmax) / 2) * self.zoom_factor
                        self.xmax = (self.xmin + self.xmax) / 2 + (self.xmax - (self.xmin + self.xmax) / 2) * self.zoom_factor
                        self.ymin = (self.ymin + self.ymax) / 2 + (self.ymin - (self.ymin + self.ymax) / 2) * self.zoom_factor
                        self.ymax = (self.ymin + self.ymax) / 2 + (self.ymax - (self.ymin + self.ymax) / 2) * self.zoom_factor
                    # Priblíženie obrazovky
                    elif event.key == pygame.K_e:
                        self.xmin = (self.xmin + self.xmax) / 2 + (self.xmin - (self.xmin + self.xmax) / 2) / self.zoom_factor
                        self.xmax = (self.xmin + self.xmax) / 2 + (self.xmax - (self.xmin + self.xmax) / 2) / self.zoom_factor
                        self.ymin = (self.ymin + self.ymax) / 2 + (self.ymin - (self.ymin + self.ymax) / 2) / self.zoom_factor
                        self.ymax = (self.ymin + self.ymax) / 2 + (self.ymax - (self.ymin + self.ymax) / 2) / self.zoom_factor
            # Vykreslenie Mandelbrotovej množiny na obrazovke
            if self.current_set == "mandelbrot":
                mandelbrot = Mandelbrot(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter)
                mandelbrot_set = mandelbrot.generate()
                fractal_surface = self.visualize_fractal(mandelbrot_set, 'Mandelbrotova množina')
            # Vykreslenie Juliovej množiny na obrazovke
            elif self.current_set == "julia":
                julia_params['xmin'] = self.xmin
                julia_params['xmax'] = self.xmax
                julia_params['ymin'] = self.ymin
                julia_params['ymax'] = self.ymax
                julia_params['width'] = self.width
                julia_params['height'] = self.height
                julia_params['max_iter'] = self.max_iter
                julia = Julia(**julia_params)
                julia_set = julia.generate()
                fractal_surface = self.visualize_fractal(julia_set, f'Juliova množina pre c = {julia_params["c"]}')

            self.screen.fill((0, 0, 0))
            self.screen.blit(fractal_surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
