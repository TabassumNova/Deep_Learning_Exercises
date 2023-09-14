import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.empty([self.resolution, self.resolution])

    def draw(self):
        r = self.resolution
        t = self.tile_size
        white = np.ones((r, r), dtype=np.uint8)
        black = np.zeros((t, t), dtype=np.uint8)

        white[0:t, 0:t] = black
        white[t:2 * t, t:2 * t] = black
        square = white[0:2 * t, 0:2 * t]
        s = round(r / (2 * t))
        self.output = np.tile(square, (s, s))
        copy = np.copy(self.output)
        plt.imshow(copy, cmap='gray')
        plt.show()
        return copy

class Circle:
    def __init__(self, resolution, radius, center):
        self.resolution = resolution
        self.radius = radius
        self.x, self.y = center
        self.output = np.empty([self.resolution, self.resolution])

    def draw(self):
        imageSizeX = np.arange(0,self.resolution,1)
        imageSizeY = np.arange(0,self.resolution,1)
        [columnsInImage,rowsInImage] = np.meshgrid(imageSizeX, imageSizeY)
        centerX = self.x
        centerY = self.y
        radius = self.radius
        circlePixels = np.sqrt((columnsInImage - centerX)**2 + (rowsInImage - centerY)**2)
        outside = circlePixels > radius
        self.output = np.ones(columnsInImage.shape)
        self.output[outside] = 0
        copy = np.copy(self.output)
        plt.imshow(copy, cmap='gray')
        plt.show()
        return copy

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.empty([self.resolution, self.resolution, 3])

    def draw(self):
        imageSizeX = np.linspace(0, 255, self.resolution)
        imageSizeY = np.linspace(0, 255, self.resolution)
        [columnsInImage, rowsInImage] = np.meshgrid(imageSizeX, imageSizeY) #[Red, Green]

        ## For Blue
        blue = np.flip(columnsInImage, axis=1)  # coloumn

        self.output[:, :, 0] = columnsInImage/255  # Red
        self.output[:, :, 1] = rowsInImage/255  # Green
        self.output[:, :, 2] = blue/255  # Blue

        copy = np.copy(self.output)
        plt.imshow(copy)
        plt.show()
        return copy