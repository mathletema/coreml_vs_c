import coremltools as ct
from typing import Callable, List, Optional
from coremltools.converters.mil import Builder, Program, Placeholder, Function
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
import coremltools.converters.mil.converter as _converter
from coremltools.converters.mil.converter import mil_convert
from coremltools.models import MLModel
from coremltools.converters.mil.backend.mil.load import load as backend_load
from coremltools.converters.mil.mil.passes.pass_pipeline import (
    PassPipeline,
    PassPipelineManager,
)
from coremltools.proto import MIL_pb2 as pm
import tempfile as _tempfile
import numpy as np
from google.protobuf import text_format
import coremltools.converters.mil.backend.mil.load as _load
from coremltools.proto import Model_pb2 as ml

N = 3000

##############################
# define computational graph #
##############################

with Function(
    {"A": Builder.TensorSpec(shape=(N, N)), "B": Builder.TensorSpec(shape=(N, N))}, None
) as func:
    A = func.inputs["A"]
    B = func.inputs["B"]
    C = Builder.matmul(x=A, y=B, name="matmul")
    D = Builder.softmax(x=C, name="softmax")
    func.set_outputs([D])


func.opset_version = func.get_max_opset_version_and_op()[0]
print("OPSET VERSION", func.opset_version)  # coremltools.target.iOS13

######################
# create the program #
######################

prog = Program()
prog.add_function("main", func)

print("Prog:", prog)
"""
> Prog: main[CoreML3](%A: (3000, 3000, fp32)(Tensor),
>               %B: (3000, 3000, fp32)(Tensor)) {
>   block0() {
>     %matmul: (3000, 3000, fp32)(Tensor) = matmul(x=%A, y=%B, transpose_x=False, transpose_y=False, name="matmul")
>     %softmax: (3000, 3000, fp32)(Tensor) = softmax(x=%matmul, axis=-1, name="softmax")
>   } -> (%softmax)
> }
"""

#######################
# compile to CreateML #
#######################

out = backend_load(prog, weights_dir="tmp")

proto1 = _load._pymil_to_milproto(prog, "tmp", ct._SPECIFICATION_VERSION_IOS_15)
proto2 = pm.Program()
out2 = ml.Model()

# with open("model_proto.txt", "w") as f:
#     f.write(text_format.MessageToString(out))

with open("model_proto.txt", "r") as f:
    text_format.Parse(f.read(), out2)

model = MLModel(
    _converter._create_mlpackage(out2, "tmp"),
    compute_units=ct.ComputeUnit.ALL,
)

print("Out", text_format.MessageToString(out))

# model = ml.Model(description=desc, specificationVersion=specification_version)
# model.mlProgram.CopyFrom(proto2)



A = np.random.rand(N, N)
B = np.random.rand(N, N)
out = model.predict({"A": A, "B": B})


print(out)
