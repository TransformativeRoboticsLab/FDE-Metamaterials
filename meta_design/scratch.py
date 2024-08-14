import numpy as np

# np.random.seed(0)
ws = []
ws_normed = []
ws_inf = []
for _ in range(10000):
    C = np.random.uniform(0., 100., (3,3))
    C = (C @ C.T)/2.

    C_normed = C / np.linalg.norm(C, ord='fro')
    
    C_infnormed = C / np.linalg.norm(C, ord=np.inf)

    w = np.linalg.eigvals(C)
    w_normed = np.linalg.eigvals(C_normed)
    ws.append(w)
    ws_normed.append(w_normed)
    ws_inf.append(np.linalg.eigvals(C_infnormed))
    
print(np.mean(np.max(ws, axis=1)))
print(np.mean(np.max(ws_normed, axis=1)))
print(np.mean(np.max(ws_inf, axis=1)))