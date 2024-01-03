# third party
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator


# util function
def class_to_dict(inst, ignore_list=[], attr_prefix=""):
    """Writes state of class instance as a dict
        Includes both attributes and properties (i.e. those methods labeled with @property)
        Note: because this capture properties, it should be viewed as a snapshot of instance state
    :param inst:  instance to represent as dict
    :param ignore_list: list of attr
    :return: dict
    """
    output = vars(inst).copy()  # captures regular variables
    cls = type(inst)  # class of instance
    properties = [p for p in dir(cls) if isinstance(getattr(cls, p), property)]
    for p in properties:
        prop = getattr(cls, p)  # get property object by name
        output[p] = prop.fget(inst)  # call its fget
    for k in list(output.keys()):  # filter out dict keys mentioned in ignore-list
        if k in ignore_list:
            del output[k]
        else:  # prepend attr_prefix
            output[attr_prefix + k] = output.pop(k)

    output[attr_prefix + "class"] = cls.__name__
    return output


# Main Matrix classes
class EkteloMatrix(LinearOperator):
    """
    An EkteloMatrix is a linear transformation that can compute matrix-vector products
    """

    # must implement: _matmat, _transpose, matrix
    # can  implement: gram, sensitivity, sum, dense_matrix, sparse_matrix, __abs__

    def __init__(self, matrix):
        """Instantiate an EkteloMatrix from an explicitly represented backing matrix

        :param matrix: a 2d numpy array or a scipy sparse matrix
        """
        self.matrix = matrix
        self.dtype = matrix.dtype
        self.shape = matrix.shape

    def asDict(self):
        d = class_to_dict(self, ignore_list=[])
        return d

    def _transpose(self):
        return EkteloMatrix(self.matrix.T)

    def _matmat(self, V):
        """
        Matrix multiplication of a m x n matrix Q

        :param V: a n x p numpy array
        :return Q*V: a m x p numpy aray
        """
        return self.matrix @ V

    def gram(self):
        """
        Compute the Gram matrix of the given matrix.
        For a matrix Q, the gram matrix is defined as Q^T Q
        """
        return self.T @ self  # works for subclasses too

    def sensitivity(self):
        # note: this works because np.abs calls self.__abs__
        return np.max(np.abs(self).sum(axis=0))

    def sum(self, axis=None):
        # this implementation works for all subclasses too
        # (as long as they define _matmat and _transpose)
        if axis == 0:
            return self.T.dot(np.ones(self.shape[0]))
        ans = self.dot(np.ones(self.shape[1]))
        return ans if axis == 1 else np.sum(ans)

    def inv(self):
        return EkteloMatrix(np.linalg.inv(self.dense_matrix()))

    def pinv(self):
        return EkteloMatrix(np.linalg.pinv(self.dense_matrix()))

    def trace(self):
        return self.diag().sum()

    def diag(self):
        return np.diag(self.dense_matrix())

    def _adjoint(self):
        return self._transpose()

    def __mul__(self, other):
        if np.isscalar(other):
            return Weighted(self, other)  # :noqa F821
        if type(other) == np.ndarray:
            return self.dot(other)
        if isinstance(other, EkteloMatrix):
            return Product(self, other)
            # note: this expects both matrix types to be compatible (e.g., sparse and sparse)
            # todo: make it work for different backing representations
        else:
            raise TypeError(
                "incompatible type %s for multiplication with EkteloMatrix"
                % type(other)
            )

    def __add__(self, other):
        if np.isscalar(other):
            other = Weighted(Ones(self.shape), other)  # :noqa F821
        return Sum([self, other])

    def __sub__(self, other):
        return self + -1 * other

    def __rmul__(self, other):
        if np.isscalar(other):
            return Weighted(self, other)  # :noqa F821
        return NotImplemented

    def __getitem__(self, key):
        """
        return a given row from the matrix

        :param key: the index of the row to return
        :return: a 1xN EkteloMatrix
        """
        # row indexing, subclasses may provide more efficient implementation
        m = self.shape[0]
        v = np.zeros(m)
        v[key] = 1.0
        return EkteloMatrix(self.T.dot(v).reshape(1, self.shape[1]))

    def dense_matrix(self):
        """
        return the dense representation of this matrix, as a 2D numpy array
        """
        if sparse.issparse(self.matrix):
            return self.matrix.toarray()
        return self.matrix

    def sparse_matrix(self):
        """
        return the sparse representation of this matrix, as a scipy matrix
        """
        if sparse.issparse(self.matrix):
            return self.matrix
        return sparse.csr_matrix(self.matrix)

    @property
    def ndim(self):
        # todo: deprecate if possible
        return 2

    def __abs__(self):
        return EkteloMatrix(self.matrix.__abs__())

    def __sqr__(self):
        if sparse.issparse(self.matrix):
            return EkteloMatrix(self.matrix.power(2))
        return EkteloMatrix(self.matrix**2)


class Ones(EkteloMatrix):
    """A m x n matrix of all ones"""

    def __init__(self, m, n, dtype=np.float64):
        self.m = m
        self.n = n
        self.shape = (m, n)
        self.dtype = dtype

    def _matmat(self, V):
        ans = V.sum(axis=0, keepdims=True)
        return np.repeat(ans, self.m, axis=0)

    def _transpose(self):
        return Ones(self.n, self.m, self.dtype)

    def gram(self):
        return self.m * Ones(self.n, self.n, self.dtype)

    def pinv(self):
        c = 1.0 / (self.m * self.n)
        return c * Ones(self.n, self.m, self.dtype)

    def trace(self):
        if self.n != self.m:
            raise ValueError("matrix is not square")
        return self.n

    @property
    def matrix(self):
        return np.ones(self.shape, dtype=self.dtype)

    def __abs__(self):
        return self

    def __sqr__(self):
        return self


class Sum(EkteloMatrix):
    """Class for the Sum of matrices"""

    def __init__(self, matrices):
        # all must have same shape
        self.matrices = matrices
        self.shape = matrices[0].shape
        self.dtype = np.result_type(*[Q.dtype for Q in matrices])

    def _matmat(self, V):
        return sum(Q.dot(V) for Q in self.matrices)

    def _transpose(self):
        return Sum([Q.T for Q in self.matrices])

    def __mul__(self, other):
        if isinstance(other, EkteloMatrix):
            return Sum(
                [Q @ other for Q in self.matrices]
            )  # should use others rmul though
        return EkteloMatrix.__mul__(self, other)

    def diag(self):
        return sum(Q.diag() for Q in self.matrices)

    @property
    def matrix(self):
        def _any_sparse(matrices):
            return any(sparse.issparse(Q.matrix) for Q in matrices)

        if _any_sparse(self.matrices):
            return sum(Q.sparse_matrix() for Q in self.matrices)
        return sum(Q.dense_matrix() for Q in self.matrices)


class Weighted(EkteloMatrix):
    """Class for multiplication by a constant"""

    def __init__(self, base, weight):
        if isinstance(base, Weighted):
            weight *= base.weight
            base = base.base
        self.base = base
        self.weight = weight
        self.shape = base.shape
        self.dtype = base.dtype

    def _matmat(self, V):
        return self.weight * self.base.dot(V)

    def _transpose(self):
        return Weighted(self.base.T, self.weight)

    def gram(self):
        return Weighted(self.base.gram(), self.weight**2)

    def pinv(self):
        return Weighted(self.base.pinv(), 1.0 / self.weight)

    def inv(self):
        return Weighted(self.base.inv(), 1.0 / self.weight)

    def trace(self):
        return self.weight * self.base.trace()

    def __abs__(self):
        return Weighted(self.base.__abs__(), np.abs(self.weight))

    def __sqr__(self):
        return Weighted(self.base.__sqr__(), self.weight**2)

    @property
    def matrix(self):
        return self.weight * self.base.matrix


class Product(EkteloMatrix):
    def __init__(self, A, B):
        if A.shape[1] != B.shape[0]:
            raise ValueError("inner dimensions do not match")
        self._A = A
        self._B = B
        self.shape = (A.shape[0], B.shape[1])
        self.dtype = np.result_type(A.dtype, B.dtype)

    def _matmat(self, X):
        return self._A.dot(self._B.dot(X))

    def _transpose(self):
        return Product(self._B.T, self._A.T)

    @property
    def matrix(self):
        return self._A.matrix @ self._B.matrix

    def gram(self):
        return Product(self.T, self)

    def inv(self):
        return Product(self._B.inv(), self._A.inv())


class Identity(EkteloMatrix):
    def __init__(self, n, dtype=np.float64):
        self.n = n
        self.shape = (n, n)
        self.dtype = dtype

    def _matmat(self, V):
        return V

    def _transpose(self):
        return self

    @property
    def matrix(self):
        return sparse.eye(self.n, dtype=self.dtype)

    def __mul__(self, other):
        if other.shape[0] != self.n:
            raise ValueError("dimension mismatch")
        return other

    def inv(self):
        return self

    def pinv(self):
        return self

    def trace(self):
        return self.n

    def __abs__(self):
        return self

    def __sqr__(self):
        return self
