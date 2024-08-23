from fenics import Function, FunctionSpace, Mesh, interpolate, Cell, UserExpression
import numpy as np
from scipy.ndimage import convolve, gaussian_filter


def func2img(shape: tuple, resolution: tuple, func: Function):
    a, b = shape
    Nx, Ny, *_ = resolution
    # Create a regular grid that matches the output image size
    x = np.linspace(0, a, Nx)
    y = np.linspace(0, b, Ny)
    X, Y = np.meshgrid(x, y)
    # Interpolate the FEniCS function onto the regular grid
    values = np.array([func(x, y) for x, y in zip(X.flatten(), Y.flatten())])
    # Reshape the values to a 2D array
    image = values.reshape([Ny, Nx])
    # Normalize the image to the range [0, 255]
    image = ((image - np.min(image)) / (np.max(image) - np.min(image))) * 255
    return image.astype(np.uint8)
    
def img2func(image: np.ndarray, mesh: Mesh, fs: FunctionSpace):
    class ImageToFunction(UserExpression):
        def __init__(self, image, mesh, **kwargs):
            super().__init__(**kwargs)
            self.mesh = mesh
            self.image = image
            
        def eval_cell(self, value, x, ufc_cell):
            p = Cell(self.mesh, ufc_cell.index).midpoint()
            Nx, Ny = self.image.shape[1], self.image.shape[0]
            i, j = int(p[0] * Nx), int(p[1] * Ny)
            value[:] = self.image[-(j+1), i]
            
        def value_shape(self):
            return ()
    
    # Normalize the image to the range [0, 1]
    normalized_image = image / np.max(image)
    # Ensure image is binarized
    if normalized_image.ndim == 3:
        normalized_image= normalized_image[:,:,0] > 0.5
    # Create the custom UserExpression
    image_to_function = ImageToFunction(normalized_image, mesh, degree=2)
    # Interpolate the UserExpression onto the Function
    func = interpolate(image_to_function, fs)
    return func

def conic_filter(image, kernel_size=5):
    def create_conic_kernel(size, peak=1):
        r = np.linspace(-peak, peak, size)
        x, y = np.meshgrid(r, r)
        d = np.sqrt(x**2 + y**2)
        kernel = np.maximum(0, peak - d)  # Linear decrease
        kernel /= np.sum(kernel)  # Normalize the kernel
        return kernel
    
    kernel = create_conic_kernel(kernel_size)
    blurred_image = convolve(image, kernel)
    
    return blurred_image
def bitmapify(r: Function, shape: tuple, img_resolution: tuple[int, int], threshold: int = 128) -> np.ndarray:
    # This blur is there just to smooth out some of the sharp 
    # corners that the fenics mesh can make if the image resolution >> mesh resolution
    r_img = func2img(shape, img_resolution, r)
    r_img = gaussian_filter(r_img, sigma=1, mode='wrap')
    return np.where(r_img > threshold, 255, 0)

def projection(x, beta=1., eta=0.5):
    tanh_beta_eta = np.tanh(beta * eta)
    tanh_beta_x_minus_eta = np.tanh(beta * (x - eta))
    tanh_beta_one_minus_eta = np.tanh(beta * (1 - eta))

    numerator = tanh_beta_eta + tanh_beta_x_minus_eta
    denominator = tanh_beta_eta + tanh_beta_one_minus_eta

    return np.array(numerator / denominator)