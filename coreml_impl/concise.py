import coremltools as ct
from typing import Callable, List, Optional
from coremltools.converters.mil import Builder, Program, Placeholder, Function
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
import numpy as np

N = 3000

##############################
# define computational graph #
##############################

with Function({
    "A": Builder.TensorSpec(shape=(N, N)),
    "B": Builder.TensorSpec(shape=(N, 1))
}, None) as func:
    A = func.inputs["A"]
    B = func.inputs["B"]
    C = Builder.matmul(x=A, y=B, name='matmul')
    D = Builder.softmax(x=C, name='softmax')
    func.set_outputs([D])


func.opset_version = func.get_max_opset_version_and_op()[0]
print("OPSET VERSION", func.opset_version)                      # coremltools.target.iOS13

######################
# create the program #
######################

prog = Program()
prog.add_function("main", func)

print("Prog:", prog)

model = ct.convert(prog)

A = np.random.rand(N, N)
B = np.random.rand(N, 1)
out = model.predict({
    "A": A,
    "B": B
})

print(out)
