
import os
import shutil
import numpy as np

from google.protobuf import text_format

import coremltools as ct
from coremltools.converters.mil.mil import Builder, Function, Program
from coremltools.proto import Model_pb2 as ml
from coremltools.converters.mil.backend.mil.load import load

from coremltools.models.utils import _ModelPackage
from coremltools.libcoremlpython import _MLModelProxy

N = 3000

PACKAGE_PATH = "model.mlpackage"
SPEC_FILE_TXT = "model_proto.txt"
SPEC_FILE_BIN = "model_proto.bin"
TMP_DIR = "tmp"

_FLAG_PROGRAM_TO_PROTO = True
_FLAG_PROTO_TO_PACKAGE = False
_FLAG_PACKAGE_CREATED = False

####################
# Program to proto #
####################

# Coremltools uses the protobuf format to save models as mlpackages. It uses
# MIL (Model Intermediate Language) to apply optimizations to, and finally
# saves it as a protobuf object

if _FLAG_PROGRAM_TO_PROTO:
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

    prog = Program()
    prog.add_function("main", func)

    # fill out proto object with program
    model = load(prog, weights_dir=TMP_DIR)
    shutil.rmtree(TMP_DIR)
    
    with open(SPEC_FILE_TXT, "w") as f:
        f.write(model.MessageToString())

####################
# Proto to package #
####################

# Model exists in as proto object in plain text. Here we serialize to binary
# and create an ml package out of it.

if _FLAG_PROTO_TO_PACKAGE:
    # clear directory
    if os.path.exists(PACKAGE_PATH):
        shutil.rmtree(PACKAGE_PATH)

    # convert txt to binary
    # ml.Model is the protobuf object class for models
    proto_model = ml.Model()
    with open("model_proto.txt", "r") as f:
        text_format.Parse(f.read(), proto_model)
    with open(SPEC_FILE_BIN, "wb") as f:
        f.write(proto_model.SerializeToString())

    # create package
    package = _ModelPackage(PACKAGE_PATH)
    package.setRootModel(SPEC_FILE_BIN, "model.mlmodel", "com.ishank.Test",
                            "CoreML Model Specification")

###############
# Run package #
###############

# Load saved model and run it using CoreML engine. This is done entirely in
# objective C. Note that _MLModelProxy is a wrapper around a c++ class defined
# in coremltoolspython

if _FLAG_PACKAGE_CREATED:
    # load model
    proxy = _MLModelProxy(PACKAGE_PATH, ct.ComputeUnit.ALL.name)

    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    C = proxy.predict({"A": A, "B": B})
    print(C)