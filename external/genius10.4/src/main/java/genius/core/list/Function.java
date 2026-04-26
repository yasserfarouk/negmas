package genius.core.list;

/**
 * A generic interface for function with one argument.
 *
 * @param <T1>
 *            type of the input parameter of the function
 * @param <Out>
 *            the output type of the function
 */
public interface Function<T1, Out> {

	/**
	 * @param t1
	 *            the argument for the function application
	 * @return a function aplied parameters T1.
	 */
	public Out apply(T1 t1);

}
