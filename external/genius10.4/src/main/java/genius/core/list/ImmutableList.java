package genius.core.list;

import java.math.BigInteger;

/**
 * Immutable (read-only) list. Implementations are possibly procedural, and can
 * therefore "hold" infinite number of items.
 * 
 * @param <E>
 *            type of the contained elements
 */
public interface ImmutableList<E> extends Iterable<E> {
	/**
	 * Returns the element at the specified position in this list.
	 *
	 * @param index
	 *            index of the element to return
	 * @return the element at the specified position in this list
	 * @throws IndexOutOfBoundsException
	 *             if the index is out of range
	 *             (<tt>index &lt; 0 || index &gt;= size()</tt>)
	 */
	E get(BigInteger index);

	/**
	 * @return the number of elements in this list.
	 */
	public BigInteger size();

	/**
	 * Returns the element at the specified position in this list.
	 *
	 * @param index
	 *            index of the element to return. Allows access of elements at
	 *            indices fitting in a long.
	 * @return the element at the specified position in this list
	 * @throws IndexOutOfBoundsException
	 *             if the index is out of range
	 *             (<tt>index &lt; 0 || index &gt;= size()</tt>)
	 */
	E get(long index);

}
