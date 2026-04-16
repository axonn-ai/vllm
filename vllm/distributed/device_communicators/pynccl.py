# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# ===================== import region =====================
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp
import os

import vllm.envs as envs
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
    ncclUniqueId,
)
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)

_NCCL_SYMM_OPS_REGISTERED = False


def register_nccl_symmetric_ops(pynccl_comm):
    from vllm.distributed.device_communicators.pynccl_allocator import (
        nccl_symm_mem_context,
    )
    from vllm.utils.torch_utils import direct_register_custom_op

    global _NCCL_SYMM_OPS_REGISTERED
    if _NCCL_SYMM_OPS_REGISTERED:
        return
    _NCCL_SYMM_OPS_REGISTERED = True

    def all_reduce_symmetric_with_copy_impl(input_tensor: torch.Tensor) -> torch.Tensor:
        with nccl_symm_mem_context(pynccl_comm):
            symm_input = torch.empty_like(input_tensor)
            symm_output = torch.empty_like(input_tensor)
        symm_input.copy_(input_tensor)
        symm_output = pynccl_comm.all_reduce(symm_input, symm_output)
        return symm_output

    def all_reduce_symmetric_with_copy_fake(input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(input_tensor)

    direct_register_custom_op(
        op_name="all_reduce_symmetric_with_copy",
        op_func=all_reduce_symmetric_with_copy_impl,
        fake_impl=all_reduce_symmetric_with_copy_fake,
    )


class PyNcclCommunicator:
    def __init__(
        self,
        group: ProcessGroup | StatelessProcessGroup,
        device: int | str | torch.device,
        library_path: str | None = None,
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the PyNcclCommunicator to. If None,
                it will be bound to f"cuda:{local_rank}".
            library_path: the path to the NCCL library. If None, it will
                use the default library path.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """
        if not isinstance(group, StatelessProcessGroup):
            assert dist.is_initialized()
            assert dist.get_backend(group) != dist.Backend.NCCL, (
                "PyNcclCommunicator should be attached to a non-NCCL group."
            )
            # note: this rank is the rank in the group
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
        else:
            self.rank = group.rank
            self.world_size = group.world_size

        self.group = group

        print(f"[DEBUG] PyNcclCommunicator.__init__ called, world_size={self.world_size}, rank={self.rank}")

        # if world_size == 1, no need to create communicator
        if self.world_size == 1 or envs.VLLM_DISABLE_PYNCCL:
            self.available = False
            self.disabled = True
            return
        try:
            self.nccl = NCCLLibrary(library_path)
        except Exception as e:
            # disable because of missing NCCL library
            # e.g. in a non-GPU environment
            print(f"[DEBUG] NCCLLibrary failed: {e}")
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False

        self.use_nvrar = os.environ.get("USE_NVRAR", "1").lower() in ("1", "true", "yes")
        self.nvrar_comm = None

        self.nccl_version = self.nccl.ncclGetRawVersion()
        if self.rank == 0:
            # get the unique id from NCCL
            self.unique_id = self.nccl.ncclGetUniqueId()
            logger.info_once(
                "vLLM is using nccl==%s", self.nccl.ncclGetVersion(), scope="local"
            )
        else:
            # construct an empty unique id
            self.unique_id = ncclUniqueId()

        if not isinstance(group, StatelessProcessGroup):
            tensor = torch.ByteTensor(list(self.unique_id.internal))
            ranks = dist.get_process_group_ranks(group)
            # arg `src` in `broadcast` is the global rank
            dist.broadcast(tensor, src=ranks[0], group=group)
            byte_list = tensor.tolist()
            for i, byte in enumerate(byte_list):
                self.unique_id.internal[i] = byte
        else:
            self.unique_id = group.broadcast_obj(self.unique_id, src=0)
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        # nccl communicator and stream will use this device
        # `torch.cuda.device` is a context manager that changes the
        # current cuda device to the specified one
        with torch.cuda.device(device):
            self.comm: ncclComm_t = self.nccl.ncclCommInitRank(
                self.world_size, self.unique_id, self.rank
            )

            stream = current_stream()
            # A small all_reduce for warmup.
            data = torch.zeros(1, device=device)
            self.all_reduce(data)
            stream.synchronize()
            del data

        if self.use_nvrar:
            from nvrar import nvshmem_comm_cuda as nvshmem_comm_cuda

            unique_id = nvshmem_comm_cuda.NVSHMEMCommWrapper.get_unique_id_bytes()
            ranks = dist.get_process_group_ranks(self.group)
            dist.broadcast(unique_id, src=ranks[0], group=self.group)
            dist.barrier(group=self.group)

            self.nvrar_comm = nvshmem_comm_cuda.NVSHMEMCommWrapper(
                self.rank, self.world_size, self.device.index, unique_id
            )
            logger.info(
                "NVSHMEMCommunicator created for process group %s "
                "with rank %d and nranks %d",
                self.group, self.rank, self.world_size,
            )

            NVRAR_MIN_BYTES = 128 * 1024       # 128KB
            NVRAR_MAX_BYTES = 8 * 1024 * 1024   # 8MB
            NVRAR_DTYPE = torch.bfloat16
            NVRAR_ELEM_SIZE = 2  # bytes per bf16

            self.nvrar_buffers = {}        # {num_elements: (tensor, tensor_id)}

            size_bytes = NVRAR_MIN_BYTES
            while size_bytes <= NVRAR_MAX_BYTES:
                num_elements = size_bytes // NVRAR_ELEM_SIZE
                tensor, tensor_id = self.nvrar_comm.allocate_tensor(
                    num_elements, NVRAR_DTYPE, self.device,
                    nvshmem_comm_cuda.Protocol.LL8)
                self.nvrar_buffers[num_elements] = (tensor, tensor_id)
                config = self.get_launch_config(
                    self.world_size, num_elements, NVRAR_DTYPE)
                self.nvrar_comm.set_kernel_params_for_tensor(
                    tensor_id,
                    config["num_blocks"],
                    config["threads_per_block"],
                    config["chunk_bytes"],
                )
                size_bytes *= 2

            logger.info(
                "NVRAR buffer pool initialized with %d sizes: %s",
                len(self.nvrar_buffers),
                sorted(self.nvrar_buffers.keys()),
            )

    def get_launch_config(self, num_gpus: int, message_bytes: int,
                          dtype: torch.dtype):
        from nvrar import resolve_params
        dtype_str = str(dtype).split(".")[-1]
        return resolve_params(num_gpus, dtype_str).for_message_bytes(
            message_bytes)

    def _is_nvrar_eligible(self, tensor: torch.Tensor) -> bool:
        """Check if tensor is eligible for NVRAR.

        Eligible tensors must be bf16, power-of-2 byte size, 128KB-8MB.
        """
        if tensor.dtype != torch.bfloat16:
            return False
        byte_size = tensor.numel() * tensor.element_size()
        if byte_size < 128 * 1024 or byte_size > 8 * 1024 * 1024:
            return False
        return (byte_size & (byte_size - 1)) == 0

    def all_reduce(
        self,
        in_tensor: torch.Tensor,
        out_tensor: torch.Tensor = None,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ) -> torch.Tensor:
        if self.disabled:
            return None
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert in_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}"
        )

        if self.use_nvrar and self.nvrar_comm is not None \
                and self._is_nvrar_eligible(in_tensor):
            assert out_tensor is None

            num_elements = in_tensor.numel()
            if not hasattr(self, "_nvrar_logged_shapes"):
                self._nvrar_logged_shapes = set()
            key = (tuple(in_tensor.shape), str(in_tensor.dtype))
            if key not in self._nvrar_logged_shapes:
                self._nvrar_logged_shapes.add(key)
                logger.info(
                    "[NVRAR] all_reduce via NVRAR: shape=%s dtype=%s bytes=%d",
                    tuple(in_tensor.shape), in_tensor.dtype,
                    in_tensor.numel() * in_tensor.element_size())
            buf_tensor, buf_id = self.nvrar_buffers[num_elements]

            buf_tensor.copy_(in_tensor.reshape(-1))
            if stream is None:
                stream = current_stream()
            self.nvrar_comm.allreduce_preallocated(
                buf_tensor, buf_id, stream.cuda_stream, "recursive")

            return buf_tensor.clone().reshape(in_tensor.shape)

        else:
            if not hasattr(self, "_nccl_logged_shapes"):
                self._nccl_logged_shapes = set()
            key = (tuple(in_tensor.shape), str(in_tensor.dtype))
            if key not in self._nccl_logged_shapes:
                self._nccl_logged_shapes.add(key)
                reason = "disabled" if not self.use_nvrar else "ineligible"
                logger.info(
                    "[NVRAR] all_reduce via NCCL (%s): shape=%s dtype=%s bytes=%d",
                    reason, tuple(in_tensor.shape), in_tensor.dtype,
                    in_tensor.numel() * in_tensor.element_size())
            if out_tensor is None:
                out_tensor = torch.empty_like(in_tensor)

            if stream is None:
                stream = current_stream()
            self.nccl.ncclAllReduce(
                buffer_type(in_tensor.data_ptr()),
                buffer_type(out_tensor.data_ptr()),
                in_tensor.numel(),
                ncclDataTypeEnum.from_torch(in_tensor.dtype),
                ncclRedOpTypeEnum.from_torch(op),
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )
            return out_tensor

    def all_gather(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor, stream=None
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        self.nccl.ncclAllGather(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            input_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def all_gatherv(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: list[int],
        stream=None,
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        assert output_tensor.shape[0] == sum(sizes)
        split_offset = 0
        self.nccl.ncclGroupStart()
        for root, split_size in enumerate(sizes):
            dst_slice = output_tensor[split_offset : split_offset + split_size]
            self.nccl.ncclBroadcast(
                buffer_type(input_tensor.data_ptr()),
                buffer_type(dst_slice.data_ptr()),
                dst_slice.numel(),
                ncclDataTypeEnum.from_torch(input_tensor.dtype),
                root,
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )
            split_offset += split_size
        self.nccl.ncclGroupEnd()

    def reduce_scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        self.nccl.ncclReduceScatter(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            output_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def reduce_scatterv(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: list[int],
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = current_stream()

        split_offset = 0
        self.nccl.ncclGroupStart()
        for root, split_size in enumerate(sizes):
            chunk = input_tensor[split_offset : split_offset + split_size, ...]
            self.nccl.ncclReduce(
                buffer_type(chunk.data_ptr()),
                buffer_type(output_tensor.data_ptr()),
                chunk.numel(),
                ncclDataTypeEnum.from_torch(input_tensor.dtype),
                ncclRedOpTypeEnum.from_torch(op),
                root,
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )
            split_offset += split_size
        self.nccl.ncclGroupEnd()

    def send(self, tensor: torch.Tensor, dst: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        if tensor.dtype in [
            torch.float8_e5m2,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        ]:
            nccl_dtype = ncclDataTypeEnum.from_torch(torch.uint8)
        else:
            nccl_dtype = ncclDataTypeEnum.from_torch(tensor.dtype)
        self.nccl.ncclSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            nccl_dtype,
            dst,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def recv(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        if tensor.dtype in [
            torch.float8_e5m2,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        ]:
            nccl_dtype = ncclDataTypeEnum.from_torch(torch.uint8)
        else:
            nccl_dtype = ncclDataTypeEnum.from_torch(tensor.dtype)
        self.nccl.ncclRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            nccl_dtype,
            src,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        if src == self.rank:
            sendbuff = buffer_type(tensor.data_ptr())
            # NCCL requires the sender also to have a receive buffer
            recvbuff = buffer_type(tensor.data_ptr())
        else:
            sendbuff = buffer_type()
            recvbuff = buffer_type(tensor.data_ptr())
        self.nccl.ncclBroadcast(
            sendbuff,
            recvbuff,
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def group_start(self):
        self.nccl.ncclGroupStart()

    def group_end(self):
        self.nccl.ncclGroupEnd()

    def register_comm_window(self, tensor: torch.Tensor):
        return self.nccl.ncclCommWindowRegister(
            self.comm,
            buffer_type(tensor.data_ptr()),
            tensor.numel() * tensor.element_size(),
            1,
        )

    def register_comm_window_raw(self, ptr: int, size: int):
        return self.nccl.ncclCommWindowRegister(self.comm, buffer_type(ptr), size, 1)

    def deregister_comm_window(self, window):
        return self.nccl.ncclCommWindowDeregister(self.comm, window)

    def batch_isend_irecv(self, p2p_ops: list, stream=None):
        if self.disabled:
            return
        if stream is None:
            stream = current_stream()
        self.group_start()
        for op in p2p_ops:
            if op.op is torch.distributed.isend:
                self.send(op.tensor, op.group_peer, stream)
            elif op.op is torch.distributed.irecv:
                self.recv(op.tensor, op.group_peer, stream)

        self.group_end()
