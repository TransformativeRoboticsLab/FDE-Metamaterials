import fenics as fe
import numpy as np
from scipy.ndimage import convolve, gaussian_filter


def func2img(shape: tuple, resolution: tuple, func: fe.Function):
    a, b = shape
    Nx = int(resolution[0] * a)
    Ny = int(resolution[1] * b)
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
    
def img2func(image: np.ndarray, mesh: fe.Mesh, fs: fe.FunctionSpace):
    class ImageToFunction(fe.UserExpression):
        def __init__(self, image, mesh, **kwargs):
            super().__init__(**kwargs)
            self.mesh = mesh
            self.image = image
            
        def eval_cell(self, value, x, ufc_cell):
            p = fe.Cell(self.mesh, ufc_cell.index).midpoint()
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
    # Interpolate the UserExpression onto the fe.Function
    func = fe.interpolate(image_to_function, fs)
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
def bitmapify(r: fe.Function, shape: tuple, img_resolution: tuple[int, int], threshold: int = 128, invert=False) -> np.ndarray:
    if 'tri' in r.function_space().ufl_cell().cellname():
        r_img = func2img(shape, img_resolution, r)
    elif 'quad' in r.function_space().ufl_cell().cellname():
        print('Arbitrary quadrilateral function sampling not supported. Defaulting to using the base mesh resolution as the image resolution')

        r_img = r.vector()[:] * 255
        s = int(np.sqrt(r_img.size))
        r_img = r_img.reshape((s,s))
        r_img = np.flip(r_img.astype(np.uint8), axis=0)
    # This blur is there just to smooth out some of the sharp 
    # corners that the fenics mesh can make if the image resolution >> mesh resolution
    r_img = gaussian_filter(r_img, sigma=img_resolution[0]//100, mode='wrap')
    out = 255 - np.flip(np.where(r_img > threshold, 255, 0), axis=0)
    return 255 - out if invert else out

def projection(x, beta=1., eta=0.5):
    tanh_beta_eta = np.tanh(beta * eta)
    tanh_beta_x_minus_eta = np.tanh(beta * (x - eta))
    tanh_beta_one_minus_eta = np.tanh(beta * (1 - eta))

    numerator = tanh_beta_eta + tanh_beta_x_minus_eta
    denominator = tanh_beta_eta + tanh_beta_one_minus_eta

    return np.array(numerator / denominator)