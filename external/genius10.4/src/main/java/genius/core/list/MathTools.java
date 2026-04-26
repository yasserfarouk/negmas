package genius.core.list;

import java.math.BigInteger;

/**
 * Basic computations
 *
 */
public class MathTools {
	/**
	 * Computes factorial. Also works for big numbers like n &gt; 15
	 * 
	 * @param n
	 *            input value
	 * @return n!
	 */
	public static BigInteger factorial(int n) {
		if (n <= 1)
			return BigInteger.ONE;
		return factorial(n - 1).multiply(BigInteger.valueOf(n));
	}

	/**
	 * @param n
	 *            top coefficient. Positive number
	 * @param k
	 *            bottom coeffiecient, in range [1, n].
	 * @return binomial coefficient n over k which equals to n! / (k! (n-k)!).
	 */
	public static BigInteger over(int n, int k) {
		return factorial(n).divide(factorial(k).multiply(factorial(n - k)));
	}

}
