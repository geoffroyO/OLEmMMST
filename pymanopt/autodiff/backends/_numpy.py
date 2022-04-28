from ._backend import Backend


class NumPyBackend(Backend):
    def __init__(self):
        super().__init__("NumPy")

    @staticmethod
    def is_available():
        return True

    @Backend._assert_backend_available
    def prepare_function(self, function):
        return function

    def _raise_not_implemented_error(self, *args, **kwargs):
        raise NotImplementedError(
            f"No autodiff support available for the canonical '{self}' "
            "backend"
        )

    compute_gradient = _raise_not_implemented_error
    compute_hessian_vector_product = _raise_not_implemented_error
