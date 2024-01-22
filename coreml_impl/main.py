#!python3

import coremltools as ct
from coremltools.converters.mil import Builder as mb
import numpy as np
import time


N = 3000
num_tests = 100

# Input to MIL program is a list of tensors. Here we have one input with
# shape (1, 100, 100, 3) and implicit dtype == fp32
@mb.program(input_specs=[mb.TensorSpec(shape=(N, N)), mb.TensorSpec(shape=(N, 1))])
def prog(A, B):
    # MIL operation takes named inputs (instead of positional inputs).
    # Here `name` argument is optional.
    # A = mb.relu(x=A, name='relu')
    C = mb.matmul(x=A, y=B, name='matmul')
    D = mb.softmax(x=C, name='softmax')
    return D

model = ct.convert(prog)

times = []

for _ in range(num_tests):
    A = np.random.rand(N, N)
    B = np.random.rand(N, 1)
    start = time.time()
    model.predict({
        "A": A,
        "B": B
    })
    end = time.time()
    times.append(end - start)

print(sum(times) / len(times))