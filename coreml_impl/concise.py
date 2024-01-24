import coremltools as ct
from coremltools.models import MLModel
import numpy as np
from coremltools.proto import Model_pb2 as ml

N = 3000

out = ml.Model()

with open("model_proto.txt", "r") as f:
    text_format.Parse(f.read(), out)

model = MLModel(
    out,
    compute_units=ct.ComputeUnit.ALL,
    weights_dir="tmp"                   # does not matter, is an empty directory anyways
)

A = np.random.rand(N, N)
B = np.random.rand(N, N)
out = model.predict({"A": A, "B": B})

print(out)
