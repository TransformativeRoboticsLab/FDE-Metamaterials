import numpy as np

M =3  # Number of iterations
N = 10000  # Number of elements minus one filled in each iteration
output_length = M * (N+1)
vector = np.zeros(output_length)  # Initialize the vector

for i in range(M):
    start = i * (N) + 1 * i
    end = (i+1) * (N-1) + i * (M)
    print(f"start: {start}, end: {end}")
    # vector[start:end] = np.arange(start, end)  # Example fill operation with a range of values

# print(vector)
