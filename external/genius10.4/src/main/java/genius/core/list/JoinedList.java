package genius.core.list;

import java.math.BigInteger;

/**
 * Creates conjunction of two lists.
 * 
 * @param <E>
 *            type of the elements
 */
public class JoinedList<E> extends AbstractImmutableList<E> {
	private ImmutableList<E> list1;
	private ImmutableList<E> list2;

	public JoinedList(ImmutableList<E> list1, ImmutableList<E> list2) {
		this.list1 = list1;
		this.list2 = list2;

	}

	@Override
	public E get(BigInteger index) {

		if (index.compareTo(list1.size()) < 0) {
			return list1.get(index);
		}
		return list2.get(index.subtract(list1.size()));
	}

	@Override
	public BigInteger size() {
		return list1.size().add(list2.size());
	}

}
