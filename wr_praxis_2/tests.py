
import numpy as np
import matplotlib.pyplot as plt
import datetime

import unittest
import tomograph
from main import compute_tomograph, gaussian_elimination, back_substitution, compute_cholesky, solve_cholesky


class Tests(unittest.TestCase):
    def test_gaussian_elimination(self):
        A = np.random.randn(4, 4)
        A1 = np.array([[4, 7, 2, 3], [-8, -13, -10, -5], [4, 9, -5, 6], [12, 20, 12, 6]])
        A2 = np.array([[11, 44, 1], [0.1, 0.4, 3], [0, 1, -1]])
        x = np.random.rand(4)
        b = np.dot(A, x)
        b1 = np.array([9, -11, 19, 18])
        b2 = np.array([1, 1, 1])
        print(A1)
        print(b1)
        A_elim, b_elim = gaussian_elimination(A1, b1)
        print("#"*30)
        print(A_elim)
        print(b_elim)
        self.assertTrue(np.allclose(np.linalg.solve(A_elim, b_elim), x))  # Check if system is still solvable
        self.assertTrue(np.allclose(A_elim, np.triu(A_elim)))  # Check if matrix is upper triangular

    def test_back_substitution(self):
        #pass
        # TODO
        A = np.array([[11, 44, 1], [0.1, 0.4, 3], [0, 1, -1]])
        print(A)
        b = np.array([1, 1, 1])
        print(b)

        x = back_substitution(A, b)
        print(x)

        self.assertTrue(np.allclose(x, np.array([-1732/329, 438/329, 109/329])))

    def test_cholesky_decomposition(self):
        #pass
        # TODO
        M = np.array([[4, 2, 2, 8], [2, 10, 10, 7], [2, 10, 11, 9], [8, 7, 9, 22]])
        L = compute_cholesky(M)

        print(M)
        print(L)

        self.assertTrue(np.allclose(M, np.array([[2, 0, 0, 0], [1, 3, 0, 0], [1, 3, 1, 0], [4, 1, 2, 1]])))

    def test_solve_cholesky(self):
        #pass
        # TODO
        L = np.array([[2, 0, 0, 0], [1, 3, 0, 0], [1, 3, 1, 0], [4, 1, 2, 1]])
        b = np.array([1, 1, 1, 1])
        x = solve_cholesky(L, b)

    def test_compute_tomograph(self):
        t = datetime.datetime.now()
        print("Start time: " + str(t.hour) + ":" + str(t.minute) + ":" + str(t.second))

        # Compute tomographic image
        n_shots = 64  # 128
        n_rays = 64  # 128
        n_grid = 32  # 64
        tim = compute_tomograph(n_shots, n_rays, n_grid)

        t = datetime.datetime.now()
        print("End time: " + str(t.hour) + ":" + str(t.minute) + ":" + str(t.second))

        # Visualize image
        plt.imshow(tim, cmap='gist_yarg', extent=[-1.0, 1.0, -1.0, 1.0],
                   origin='lower', interpolation='nearest')
        plt.gca().set_xticks([-1, 0, 1])
        plt.gca().set_yticks([-1, 0, 1])
        plt.gca().set_title('%dx%d' % (n_grid, n_grid))

        plt.show()


if __name__ == '__main__':
    unittest.main()

