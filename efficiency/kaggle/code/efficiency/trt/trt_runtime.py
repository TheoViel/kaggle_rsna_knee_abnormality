"""TensorRT engines for the efficiency pipeline: build from the fp16 ONNX exports, load, run on torch tensors.

Engines are specific to the GPU model and the TensorRT version: build them on the target (Kaggle T4) with the same
tensorrt package the submission uses. Works with TensorRT 10 (tensorrt-cu12, Kaggle) and 11 (tensorrt-cu13).
"""

import os

import torch


def _trt():
    import tensorrt as trt

    return trt


def build_engine(onnx_path, engine_path, workspace_gb=4, opt_level=3, log=print):
    """fp16 ONNX graph -> serialized engine (strongly typed: the graph's own fp16 dtypes, as in PyTorch)."""
    trt = _trt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError(f"{onnx_path}: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.builder_optimization_level = opt_level
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"{onnx_path}: engine build failed")
    with open(engine_path, "wb") as f:
        f.write(plan)
    return engine_path


_DTYPE = None


def _torch_dtype(trt_dtype):
    global _DTYPE
    trt = _trt()
    if _DTYPE is None:
        _DTYPE = {trt.float16: torch.float16, trt.float32: torch.float32, trt.int32: torch.int32,
                  trt.int64: torch.int64, trt.bool: torch.bool}
    return _DTYPE[trt_dtype]


class TRTModule:
    """One engine with one input "x" and one output "y" (fixed shapes): __call__(x) -> y, on the current stream."""

    def __init__(self, engine_path, device="cuda"):
        trt = _trt()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"{engine_path}: cannot deserialize (other GPU / TensorRT version?)")
        self.context = self.engine.create_execution_context()
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        io = {n: self.engine.get_tensor_mode(n) for n in names}
        self.inp = next(n for n, m in io.items() if m == trt.TensorIOMode.INPUT)
        self.out = next(n for n, m in io.items() if m == trt.TensorIOMode.OUTPUT)
        self.in_shape = tuple(self.engine.get_tensor_shape(self.inp))
        self.in_dtype = _torch_dtype(self.engine.get_tensor_dtype(self.inp))
        self.y = torch.empty(tuple(self.engine.get_tensor_shape(self.out)),
                             dtype=_torch_dtype(self.engine.get_tensor_dtype(self.out)), device=self.device)
        self.context.set_tensor_address(self.out, self.y.data_ptr())

    def __call__(self, x):
        """Runs on a private stream (TensorRT syncs extra on the default one), ordered after / before the caller's."""
        assert tuple(x.shape) == self.in_shape, (x.shape, self.in_shape)
        x = x.to(self.in_dtype).contiguous()
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)
        self.context.set_tensor_address(self.inp, x.data_ptr())
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        caller.wait_stream(self.stream)
        x.record_stream(self.stream)
        return self.y.clone()            # the output buffer is reused by the next call


def engine_name(onnx_name):
    return os.path.splitext(onnx_name)[0] + ".engine"
