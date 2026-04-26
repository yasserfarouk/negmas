package agents.anac.y2016.caduceus.agents.Caduceus;

/**
 * Created by burakatalay on 20/03/16.
 */
public class UtilFunctions {
	public static boolean equals(double[] A, double[] B, double precision) {
		assert A.length == B.length;
		double[] distance = UtilFunctions.subtract(A, B);

		for (int i = 0; i < A.length; i++) {
			if (distance[i] > precision)
				return false;
		}

		return true;
	}

	public static double getEuclideanDistance(double[] A, double[] B) {
		assert A.length == B.length;
		double distance = 0;

		for (int i = 0; i < A.length; i++) {
			double d = A[i] - B[i];
			distance += d * d;
		}
		return Math.sqrt(distance);
	}

	public static double[] calculateUnitVector(double[] A, double[] B) {
		assert A.length == B.length;
		// (B − A) / |B−A|
		// A to B direction

		double[] unitVector = new double[A.length];

		unitVector = UtilFunctions.subtract(B, A);
		double norm = UtilFunctions.norm(unitVector);
		unitVector = UtilFunctions.divide(unitVector, norm);

		return unitVector;
	}

	public static double[] add(double[] A, double[] B) {
		assert A.length == B.length;
		double[] result = new double[A.length];

		for (int i = 0; i < A.length; i++) {
			result[i] = A[i] + B[i];
		}

		return result;
	}

	public static double[] multiply(double[] A, double number) {
		double[] result = new double[A.length];

		for (int i = 0; i < A.length; i++) {
			result[i] = A[i] * number;
		}

		return result;
	}

	public static double[] divide(double[] A, double number) {
		double[] result = new double[A.length];

		for (int i = 0; i < A.length; i++) {
			result[i] = A[i] / number;
		}

		return result;
	}

	public static double[] normalize(double[] A) {
		double[] result = new double[A.length];

		double sum = 0;

		for (int i = 0; i < A.length; i++) {
			result[i] = A[i];
			sum += A[i];
		}

		for (int i = 0; i < A.length; i++) {
			result[i] = result[i] / sum;
		}
		return result;
	}

	public static double norm(double[] A) {
		double norm = 0;

		for (int i = 0; i < A.length; i++) {
			norm = A[i] * A[i];
		}
		norm = Math.sqrt(norm);
		return norm;
	}

	public static double[] subtract(double[] A, double[] B) {
		assert A.length == B.length;

		double[] result = new double[A.length];

		for (int i = 0; i < A.length; i++) {
			result[i] = A[i] - B[i];
		}

		return result;
	}

	public static String toString(double[] array, String delimiter) {
		StringBuilder buffer = new StringBuilder();
		for (int i = 0; i < array.length; i++) {
			buffer.append(array[i]);
			buffer.append(" ");
			if (i != array.length - 1) {
				buffer.append(delimiter);
			}
		}
		return buffer.toString();
	}

	public static void print(double[] array) {
		System.out.println("[ " + toString(array, ", ") + " ]");
	}
}
