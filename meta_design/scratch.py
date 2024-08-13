import numpy as np
from mechanics import calculate_elastic_constants, anisotropy_index
nu = 0.3
S = np.array([[1., -nu, 0],
              [-nu, 1., 0],
              [0, 0, 2*(1+nu)]])

print('S',S)
C = np.linalg.inv(S)
print('C',C)

# C = np.array([[1/(1-nu**2), nu/(1-nu**2), 0],
#               [nu/(1-nu**2), 1/(1-nu**2), 0],
#               [0, 0, 1/(2*(1+nu))]])
m = np.diag([1., 1., np.sqrt(2)])
C = m @ C @ m
w, v = np.linalg.eigh(C)
print('w\n',w)
print('v\n',v)
    
    
    
F = np.array([[ 0.07313379, -0.01411028,  0.01486252],
 [-0.01411028,  0.0609288,  -0.00069366],
 [ 0.01486252, -0.00069366,  0.0377074 ]])

print(calculate_elastic_constants(F, input_style='standard'))
print('ASU:', anisotropy_index(F, input_style='standard')[-1])
 
 