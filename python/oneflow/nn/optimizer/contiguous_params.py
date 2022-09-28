import oneflow
from oneflow import nn

class ContiguousParams:
    # TODO: the Tensor's address should be checked all the time

    def __init__(self, parameters):
        self._parameters = list(parameters)
        self._param_buffer = None
        self._grad_buffer = None
        self._init_buffers()
        self.make_params_contiguous()

    def _init_buffers(self):
        dtype = self._parameters[0].dtype
        device = self._parameters[0].device
        if not all(p.dtype == dtype for p in self._parameters):
            raise ValueError("All parameters must be of the same dtype.")
        if not all(p.device == device for p in self._parameters):
            raise ValueError("All parameters must be on the same device.")
        size = sum(p.numel() for p in self._parameters)
        self._param_buffer = oneflow.zeros(size, dtype=dtype, device=device)
        self._grad_buffer = oneflow.zeros(size, dtype=dtype, device=device)

    def make_params_contiguous(self):
        """Create a buffer to hold all params and update the params to be views of the buffer.

        Args:
            parameters: An iterable of parameters.
        """
        index = 0
        for p in self._parameters:
            size = p.numel()
            self._param_buffer[index:index + size] = p.data.view(-1)
            p.data = self._param_buffer[index:index + size].view(p.data.shape)
            p.grad = self._grad_buffer[index:index + size].view(p.data.shape)
            p._is_grad_acc_inplace = True
            index += size

        self._param_buffer.grad = self._grad_buffer
        self._param_buffer = [self._param_buffer]

    def contiguous(self):
        """Return all parameters as one contiguous buffer."""
        return self._param_buffer

    def original(self):
        """Return the non-flattened parameters."""
        return self._parameters
