package genius.core.list;

import java.io.Serializable;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

/**
 * ArrayList implementation of {@link ImmutableList}. This is the key to
 * creating the initial ImmutableList.
 * 
 * @param <E>
 *            type of the elements
 */
@SuppressWarnings("serial")
public class ImArrayList<E> implements ImmutableList<E>, Serializable {

	private ArrayList<E> list;

	/**
	 * Copies elements of given list into an immutable list. SO the resulting
	 * list is really immutable, although the components may still be mutable.
	 * 
	 * @param list
	 *            the source list.
	 */
	public ImArrayList(Collection<E> list) {
		this.list = new ArrayList<E>(list);
	}

	public ImArrayList() {
		list = new ArrayList<E>();
	}

	@Override
	public Iterator<E> iterator() {
		return list.iterator();
	}

	@Override
	public E get(BigInteger index) {
		return list.get(index.intValue());
	}

	@Override
	public BigInteger size() {
		return BigInteger.valueOf(list.size());
	}

	@Override
	public String toString() {
		return list.toString();
	}

	@Override
	public E get(long index) {
		return list.get((int) index);
	}

}
