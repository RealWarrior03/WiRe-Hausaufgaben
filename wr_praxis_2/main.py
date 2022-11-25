import numpy as np
import tomograph


####################################################################################################
# Exercise 1: Gaussian elimination

def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting.

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """
    # Create copies of input matrix and vector to leave them unmodified
    A = A.copy()
    b = b.copy()

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    if A.shape[0] != b.shape[0]:
        raise ValueError("Matrix and vector aren't compatible!")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix is not square")

    # TODO: Perform gaussian elimination
    for i in range(A.shape[0] - 1):
        if use_pivoting:
            gl = A[i][i]
            gl_i = i
            for s in range(i, A.shape[0]):
                cur = np.absolute(A[s][i])
                if cur > gl:
                    gl = cur
                    gl_i = s
            for t in range(A.shape[0]):
                A[gl_i][t], A[i][t] = A[i][t], A[gl_i][t]
            b[gl_i], b[i] = b[i], b[gl_i]
        for j in range(i + 1, A.shape[0]):
            if np.isclose(A[i][i], 0):
                raise ValueError("Dividing by zero in A!")
            else:
                temp = (A[j][i] / A[i][i])
                for t in range(i, A.shape[0]):
                    A[j][t] -= A[i][t] * temp
                b[j] -= b[i] * temp

    return A, b


def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    if A.shape[0] != b.shape[0]:
        raise ValueError("Matrix and vector aren't compatible!")

    # TODO: Initialize solution vector with proper size
    x = np.zeros(b.shape[0])

    # TODO: Run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist
    A_copy, b_copy = gaussian_elimination(A, b, True)
    print(A_copy)
    print(np.isclose(A_copy[2][0], 0))
    print(b_copy)
    for i in range(x.shape[0] - 1, -1, -1):
        temp = 0
        for j in range(i + 1, x.shape[0]):
            temp += A_copy[i][j] * x[j]
        if np.isclose(A_copy[i][i], 0):
            raise ValueError("no/infinite solution(s) exist..")
        else:
            x[i] = 1 / A_copy[i][i] * (b_copy[i] - temp)

    return x


####################################################################################################
# Exercise 2: Cholesky decomposition

def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L :  Cholesky factor of M

    Forbidden:
    - numpy.linalg.*
    """

    # TODO check for symmetry and raise an exception of type ValueError
    (n, m) = M.shape

    if n != m:
        raise ValueError("Matrix isn't square!")
    for i in range(n):
        for j in range(m):
            if not np.isclose(M[i][j], M[j][i]):
                raise ValueError("Matrix isn't symmetric!")

    # TODO build the factorization and raise a ValueError in case of a non-positive definite input matrix
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1):
            temp = 0
            if i == j:
                for k in range(i):
                    temp += L[i][k] * L[i][k]
                temp = M[i][i] - temp
                if temp < 0:  # negative square root
                    raise ValueError("Matrix is not psd! (negative square root)")
                temp = np.sqrt(temp)
                if np.isclose(temp, 0):  # square root is close to 0
                    raise ValueError("Matrix is not psd! (square root is zero)")
                L[i][i] = temp
            else:
                for k in range(j):
                    temp = temp + L[i][k] * L[j][k]
                temp = M[i][j] - temp
                if np.isclose(L[j][j], 0):  # dividing by zero
                    raise ValueError("Matrix is not psd! (dividing by zero)")
                temp = 1 / L[j][j] * temp
                L[i][j] = temp

    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky factor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """

    # TODO Check the input for validity, raising a ValueError if this is not the case
    (n, m) = L.shape
    if n != m:
        raise ValueError("L is not square")
    if n != b.shape[0]:
        raise ValueError("Matrix and vector aren't compatible")
    for r in range(n):
        for i in range(r + 1, m):
            if not np.isclose(L[r][i], 0):
                raise ValueError("Matrix is not a lower triangular matrix")

    # TODO Solve the system by forward- and backsubstitution
    x = np.zeros(m)
    L_T = np.transpose(L)
    A = np.matmul(L, L_T)

    x = back_substitution(A, b)

    return x


####################################################################################################
# Exercise 3: Tomography

def setup_system_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots : number of different shot directions
    n_rays  : number of parallel rays per direction
    n_grid  : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    L : system matrix
    g : measured intensities

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    -
    """

    # TODO: Initialize system matrix with proper size
    L = np.zeros((1, 1))
    # TODO: Initialize intensity vector
    g = np.zeros(1)

    # TODO: Iterate over equispaced angles, take measurements, and update system matrix and sinogram
    theta = 0
    # Take a measurement with the tomograph from direction r_theta.
    # intensities: measured intensities for all <n_rays> rays of the measurement. intensities[n] contains the intensity for the n-th ray
    # ray_indices: indices of rays that intersect a cell
    # isect_indices: indices of intersected cells
    # lengths: lengths of segments in intersected cells
    # The tuple (ray_indices[n], isect_indices[n], lengths[n]) stores which ray has intersected which cell with which length. n runs from 0 to the amount of ray/cell intersections (-1) of this measurement.
    intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)

    return [L, g]


def compute_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots : number of different shot directions
    n_rays  : number of parallel rays per direction
    n_grid  : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    tim : tomographic image

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    """

    # Setup the system describing the image reconstruction
    [L, g] = setup_system_tomograph(n_shots, n_rays, n_grid)

    # TODO: Solve for tomographic image using your Cholesky solver
    # (alternatively use Numpy's Cholesky implementation)

    # TODO: Convert solution of linear system to 2D image
    tim = np.zeros((n_grid, n_grid))

    return tim


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
