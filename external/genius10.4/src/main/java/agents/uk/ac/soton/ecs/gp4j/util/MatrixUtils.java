package agents.uk.ac.soton.ecs.gp4j.util;

import agents.Jama.CholeskyDecomposition;
import agents.Jama.Matrix;

public class MatrixUtils {

	public static Matrix sum(Matrix matrix) {
		double[][] result = new double[1][matrix.getColumnDimension()];
		double[][] array = matrix.getArray();

		for (int i = 0; i < matrix.getRowDimension(); i++)
			for (int j = 0; j < matrix.getColumnDimension(); j++)
				result[0][j] += array[i][j];

		return new Matrix(result);
	}

	public static Matrix createColumnVector(double[] values) {
		return new Matrix(values, 1).transpose();
	}

	public static Matrix kronecker(Matrix a, Matrix b) {
		int ma = a.getRowDimension();
		int mb = b.getRowDimension();
		int na = a.getColumnDimension();
		int nb = b.getColumnDimension();

		double[][] A = a.getArray();
		double[][] B = b.getArray();

		Matrix X = new Matrix(ma * mb, na * nb);
		double[][] C = X.getArray();
		for (int k = 0; k < ma; k++) {
			for (int l = 0; l < na; l++) {
				for (int i = 0; i < mb; i++) {
					for (int j = 0; j < nb; j++) {
						C[k * mb + i][l * nb + j] = A[k][l] * B[i][j];
					}
				}
			}
		}
		return X;
	}

	public static double logDetChol(CholeskyDecomposition cholesky) {
		return logDetChol(cholesky.getL());
	}

	public static double logDetChol(Matrix cholesky) {
		int n = Math.min(cholesky.getColumnDimension(), cholesky
				.getRowDimension());
		double logDet = 0.0;
		for (int i = 0; i < n; i++)
			logDet += 2 * Math.log(cholesky.get(i, i));

		return logDet;
	}

	public static double logDet(Matrix matrix) {
		return logDetChol(matrix.chol());
	}

	public static Matrix append(Matrix matrix, double[] d) {
		return append(matrix, createColumnVector(d));
	}

	public static Matrix append(Matrix matrix, Matrix d) {
		int n = matrix.getRowDimension();
		int nd = d.getRowDimension();
		int m = matrix.getColumnDimension();
		int md = d.getColumnDimension();

		if (n == 0 || m == 0)
			return d;

		if (md == 0 || nd == 0)
			return matrix;

		if (m != md) {
			throw new IllegalArgumentException(
					"Matrices have different number of columns: " + m + " "
							+ md);
		}

		Matrix updatedMatrix = new Matrix(n + nd, m);
		updatedMatrix.setMatrix(0, n - 1, 0, m - 1, matrix);
		updatedMatrix.setMatrix(n, n + nd - 1, 0, m - 1, d);

		return updatedMatrix;
	}

	public static Matrix solveChol(Matrix chol, Matrix B) {
		// return chol.transpose().solve(chol.solve(y));

		int n = chol.getRowDimension();
		double[][] L = chol.getArray();

		if (B.getRowDimension() != n) {
			throw new IllegalArgumentException(
					"Matrix row dimensions must agree.");
		}

		// Copy right hand side.
		double[][] X = B.getArrayCopy();
		int nx = B.getColumnDimension();
		// Solve L*Y = B;
		for (int k = 0; k < n; k++) {
			for (int j = 0; j < nx; j++) {
				for (int i = 0; i < k; i++) {
					X[k][j] -= X[i][j] * L[k][i];
				}
				X[k][j] /= L[k][k];
			}
		}
		// Solve L'*X = Y;
		for (int k = n - 1; k >= 0; k--) {
			for (int j = 0; j < nx; j++) {
				for (int i = k + 1; i < n; i++) {
					X[k][j] -= X[i][j] * L[i][k];
				}
				X[k][j] /= L[k][k];
			}
		}

		return new Matrix(X, n, nx);
	}

	/**
	 * See Matlab's cholupdate
	 * 
	 * @param L
	 * @param W
	 * @return
	 */
	public static Matrix choleskyUpdate(Matrix L, Matrix W) {
		double beta = 1;
		int n = L.getColumnDimension();
		double[][] l = L.getArray();
		double[][] w = W.getArray();

		for (int j = 0; j < n; j++) {
			double alpha = w[j][0] / l[j][j];
			double beta2 = Math.sqrt(beta * beta + alpha * alpha);
			double gamma = alpha / (beta2 * beta);
			double delta = beta / beta2;
			l[j][j] = delta * l[j][j] + gamma * w[j][0];
			w[j][0] = alpha;
			beta = beta2;

			Matrix W1 = W.getMatrix(j + 1, n - 1, 0, 0);

			W.setMatrix(j + 1, n - 1, 0, 0, W.getMatrix(j + 1, n - 1, 0, 0)
					.minus(L.getMatrix(j + 1, n - 1, j, j).times(alpha)));

			L.setMatrix(j + 1, n - 1, j, j, L.getMatrix(j + 1, n - 1, j, j)
					.times(delta).plus(W1.times(gamma)));
		}

		return L;
	}

	public static Matrix choleskyDowndate(Matrix R) {
		int n = R.getRowDimension();

		Matrix S = R.getMatrix(1, n - 1, 0, 0);
		Matrix U = R.getMatrix(1, n - 1, 1, n - 1);

		return MatrixUtils.choleskyUpdate(U, S);
	}
}
