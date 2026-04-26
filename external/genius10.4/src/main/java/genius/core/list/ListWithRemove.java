package genius.core.list;

import java.math.BigInteger;
import java.util.Iterator;
import java.util.TreeSet;

/**
 * Turns immutable list into a list from which elements can be marked as
 * "removed". Immutable
 *
 * @param <E>
 *            type of the elements
 */
public class ListWithRemove<E> extends AbstractImmutableList<E> {
	private ImmutableList<E> list;
	// removed list index numbers
	TreeSet<BigInteger> removedIndices = new TreeSet<BigInteger>();

	public ListWithRemove(ImmutableList<E> list) {
		this.list = list;
	}

	private ListWithRemove(ImmutableList<E> list, TreeSet<BigInteger> removed) {
		this.list = list;
		this.removedIndices = removed;
	}

	@Override
	public E get(BigInteger index) {
		return list.get(realIndex(index));
	}

	/**
	 * 
	 * @param index
	 * @return the real index of an item in the original list. This can be
	 *         larger than given index because {@link #remvedIndices} are
	 *         invisible. This operation can become expensive if many items have
	 *         been removed
	 * 
	 */
	private BigInteger realIndex(BigInteger index) {
		BigInteger realIndex = index;

		/**
		 * invariant: realIndex has correct value if only removedIndices up to
		 * current iteration were in the list.
		 */
		Iterator<BigInteger> removed = removedIndices.iterator();
		while (removed.hasNext() && removed.next().compareTo(realIndex) <= 0) {
			realIndex = realIndex.add(BigInteger.ONE);
		}

		return realIndex;
	}

	@Override
	public BigInteger size() {
		return list.size().subtract(BigInteger.valueOf(removedIndices.size()));
	}

	/**
	 * Remove item at index n
	 * 
	 * @param index
	 *            index of the element to return
	 * @return the element at the specified position in this list before it was
	 *         removed.
	 */
	public ListWithRemove<E> remove(BigInteger index) {
		TreeSet<BigInteger> removed = new TreeSet<BigInteger>(removedIndices);
		removed.add(realIndex(index));
		return new ListWithRemove<E>(list, removed);
	}

}
