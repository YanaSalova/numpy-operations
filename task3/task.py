import numpy as np


class SolutionCallable:
    def __init__(
        self, is_single, fd, solution_vector=None, free_vars=None, particular=None
    ):
        self.is_single = is_single
        self.fd = fd
        if is_single:
            self.solution = solution_vector
        else:
            self.particular = particular
            self.free_vars = free_vars

    def __call__(self, params):
        if self.is_single:
            if self.fd != 0:
                raise ValueError("Degrees of freedom must be zero.")
            if params.size != 0:
                raise ValueError("Expected input with last dimension of size 0.")

            if params.shape == (0,):
                return self.solution
            desired_shape = params.shape[:-1] + (1,)
            return np.full(desired_shape, self.solution)
        if len(params.shape) == 0:
            raise ValueError(
                f"Expected last dimension {self.fd}, but parameters have an empty shape."
            )
        if params.shape[-1] != self.fd:
            raise ValueError(
                f"Expected last dimension {self.fd}, but got {params.shape[-1]}."
            )
        return self.particular + np.tensordot(
            params, self.free_vars, axes=([params.ndim - 1], [0])
        )


class Solution:
    def __init__(self, is_single, fd, solution_vector=None, free_vars=None):
        self._is_single = is_single
        self._fd = fd
        if is_single:
            self._solution = solution_vector
            self._solutions = SolutionCallable(
                is_single=True, fd=fd, solution_vector=solution_vector
            )
        else:
            self._particular = solution_vector
            self._free_vars = free_vars
            self._solutions = SolutionCallable(
                is_single=False, fd=fd, particular=solution_vector, free_vars=free_vars
            )
        self.__dict__["_locked"] = True

    @property
    def is_single(self):
        return self._is_single

    @property
    def fd(self):
        return self._fd

    @property
    def solutions(self):
        return self._solutions

    def __setattr__(self, key, value):
        if "_locked" in self.__dict__:
            raise AttributeError("Attributes of Solution object cannot be modified.")
        super().__setattr__(key, value)

    def __delattr__(self, item):
        raise AttributeError("Attributes of Solution object cannot be deleted.")


def gauss(equations):
    A = equations.copy().astype(float)
    m, n = A.shape
    num_vars = n - 1
    pivot_cols = []
    row = 0

    for col in range(num_vars):
        pivot = None
        for r in range(row, m):
            if not np.isclose(A[r, col], 0):
                pivot = r
                break
        if pivot is None:
            continue
        A[[row, pivot]] = A[[pivot, row]]
        pivot_cols.append(col)
        A[row] = A[row] / A[row, col]
        for r in range(m):
            if r != row and not np.isclose(A[r, col], 0):
                A[r] = A[r] - A[r, col] * A[row]
        row += 1

    rank = len(pivot_cols)

    for r in range(rank, m):
        if not np.isclose(A[r, -1], 0):
            return None

    if rank == num_vars:
        if num_vars == 1:
            solution = np.array(A[0, -1] / A[0, 0])
        else:
            solution = np.zeros(num_vars)
            for r in range(rank):
                solution[pivot_cols[r]] = A[r, -1]
        return Solution(is_single=True, fd=0, solution_vector=solution)

    fd = num_vars - rank
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(num_vars) if c not in pivot_set]
    particular = np.zeros(num_vars)
    for r in range(rank):
        particular[pivot_cols[r]] = A[r, -1]
    directions = []
    for free in free_cols:
        vec = np.zeros(num_vars)
        vec[free] = 1
        for r in range(rank):
            vec[pivot_cols[r]] = -A[r, free]
        directions.append(vec)
    directions = np.array(directions)
    return Solution(
        is_single=False, fd=fd, solution_vector=particular, free_vars=directions
    )
