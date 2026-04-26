package genius.core.list;

/**
 * A generic interface for function of 2 parameters.
 *
 * @param <T1>
 *            type of create parameter 1
 * @param <T2>
 *            type of create parameter 2
 * @param <Out>
 *            the produced type
 */
public interface Function2<T1, T2, Out> {

	/**
	 * @param t1
	 *            the first argument for the function application
	 * @param t2
	 *            the second argument for the function application
	 * @return a new object taking parameters T1 and T2.
	 */
	public Out apply(T1 t1, T2 t2);
}
