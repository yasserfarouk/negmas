package genius.core.list;

import java.math.BigInteger;

/**
 * Assumes that inner lists all contain just 1 element, and turns a nested list
 * into plain list.
 *
 * @param <E>
 *            type of the elements
 */
public class FlatList<E> extends AbstractImmutableList<E> {

	private ImmutableList<ImmutableList<E>> list;

	/**
	 * gives a list of the form [a1,a2,...,an]
	 * 
	 * @param list
	 *            a list of form [[a1],[a2],[a2]...[an]]
	 */
	public FlatList(ImmutableList<ImmutableList<E>> list) {
		this.list = list;
	}

	@Override
	public E get(BigInteger index) {
		return list.get(index).get(BigInteger.ZERO);
	}

	@Override
	public BigInteger size() {
		return list.size();
	}

}
