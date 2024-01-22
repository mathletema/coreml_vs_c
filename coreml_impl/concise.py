import coremltools as ct
from typing import Callable, List, Optional
from coremltools.converters.mil import Builder, Program, Placeholder, Function
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
import coremltools.converters.mil.converter as _converter
from coremltools.converters.mil.converter import mil_convert
from coremltools.models import MLModel
from coremltools.converters.mil.mil.passes.pass_pipeline import (
    PassPipeline,
    PassPipelineManager,
)
import tempfile as _tempfile
import numpy as np

main_passes = [
    "common::lower_complex_dialect_ops",
    "common::update_output_dtypes",
    "common::cast_optimization",
    "common::noop_elimination",
    "common::int_op_canonicalization",
    "common::nullify_redundant_quantization_zero_point",
    "common::dequantize_quantize_pair_elimination",
    "common::distributive_quantized_binary_op_scale_normalization",
    "common::dequantize_to_constexpr",
    "common::const_elimination",
    "common::sanitize_input_output_names",
    "common::divide_to_multiply",
    "common::select_optimization",
    "common::add_conv_transpose_output_shape",
    "common::const_elimination",
    "common::const_deduplication",
    "common::loop_invariant_elimination",
    "common::remove_symbolic_reshape",
    "common::noop_elimination",
    "common::fuse_matmul_weight_bias",
    "common::fuse_linear_bias",
    "common::fuse_gelu_tanh_approximation",
    "common::fuse_gelu_exact",
    "common::fuse_leaky_relu",
    "common::rank0_expand_dims_swap",
    "common::fuse_squeeze_expand_dims",
    "common::compose_conv1d",
    "common::use_reflection_padding",
    "common::merge_consecutive_paddings",
    "common::fuse_pad_conv",
    "common::image_input_preprocess",
    "common::replace_stack_reshape",
    "common::reduce_transposes",
    "common::fuse_conv_scale",
    "common::fuse_conv_bias",
    "common::fuse_onehot_matmul_to_gather",
    "common::fuse_layernorm_or_instancenorm",
    "common::fuse_elementwise_to_batchnorm",
    "common::fuse_reduce_mean",
    "common::fuse_conv_batchnorm",
    "common::fuse_conv_scale",
    "common::fuse_conv_bias",
    "common::fuse_conv_batchnorm",
    "common::detect_concat_interleave",
    "common::concat_to_pixel_shuffle",
    "common::fuse_prelu",
    "common::prelu_to_lrelu",
    "common::merge_consecutive_relus",
    "common::merge_consecutive_reshapes",
    "common::merge_consecutive_transposes",
    "common::expand_high_rank_reshape_and_transpose",
    "common::reduce_transposes",
    "common::remove_redundant_ops",
    "common::dead_code_elimination",
    "common::dead_code_elimination",
    "common::const_elimination",
    "common::cast_optimization",
    "common::dead_code_elimination",
    "common::const_elimination",
    "common::const_deduplication",
    "common::dead_code_elimination",
    "common::merge_tensorwise_affine_dequantize_with_consecutive_ops",
    "common::loop_invariant_elimination",
    "common::noop_elimination",
    "common::dedup_op_and_var_names",
    "common::reduce_transposes",
    "common::remove_redundant_ops",
    "common::topological_reorder",
    "common::dead_code_elimination",
]

backend_passes = [
    "common::const_elimination",
    "mil_backend::adjust_io_to_supported_types",
    "mil_backend::insert_image_preprocessing_ops",
    "mil_backend::fuse_activation_silu",
    "mil_backend::fuse_pow2_sqrt",
    "common::const_elimination",
    "common::const_deduplication",
    "common::cast_optimization",
    "common::dead_code_elimination",
    "mil_backend::sanitize_name_strings",
    "common::dedup_op_and_var_names",
    "nn_backend::handle_unused_inputs",
]

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

prog_ = Program()
prog_.add_function("main", func)

print("Prog:", prog_)
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

main_pipeline = PassPipeline()
main_pipeline.remove_passes({"common::add_fp16_cast", "common::add_int16_cast"})
frontend_pipeline, backend_pipeline = _converter._construct_other_pipelines(
    main_pipeline, "milinternal", "mlprogram"
)

print(f"{main_pipeline.passes=}")
print(f"{frontend_pipeline.passes=}")
print(f"{backend_pipeline.passes=}")


prog = prog_
PassPipelineManager.apply_pipeline(prog, frontend_pipeline)
PassPipelineManager.apply_pipeline(prog, main_pipeline)
PassPipelineManager.apply_pipeline(prog, backend_pipeline)
out = _converter.ConverterRegistry.backends.get("mlprogram")()(prog, weights_dir="tmp")

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

model = MLModel(
    _converter._create_mlpackage(out, "tmp"),
    mil_program=prog,
    compute_units=ct.ComputeUnit.ALL,
)

A = np.random.rand(N, N)
B = np.random.rand(N, N)
out = model.predict({"A": A, "B": B})


print(out)
