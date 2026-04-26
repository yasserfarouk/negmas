package genius.core.listener;

/**
 * Simple reusable listenable
 *
 * @param <TYPE>
 *            the type of the data being passed around.
 */
public interface Listenable<TYPE> {
	/**
	 * Add listener for this observable
	 * 
	 * @param l
	 *            the listener to add
	 */
	public void addListener(Listener<TYPE> l);

	/**
	 * Remove listener for this observable
	 * 
	 * @param l
	 *            the listener to remove
	 */

	public void removeListener(Listener<TYPE> l);

}
