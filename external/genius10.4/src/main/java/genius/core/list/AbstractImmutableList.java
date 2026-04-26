package genius.core.list;

import java.math.BigInteger;
import java.util.Iterator;

/**
 * 
 * @param <E>
 *            type of the elements
 */
public abstract class AbstractImmutableList<E> implements ImmutableList<E> {

	private static final int PRINT_LIMIT = 20;

	@Override
	public Iterator<E> iterator() {
		return new Iterator<E>() {
			BigInteger i = BigInteger.ZERO;

			@Override
			public boolean hasNext() {
				return i.compareTo(size()) < 0;
			}

			@Override
			public E next() {
				E next = get(i);
				i = i.add(BigInteger.ONE);
				return next;
			}
		};
	}

	@Override
	public String toString() {
		int end;
		if (size().compareTo(BigInteger.valueOf(PRINT_LIMIT)) > 0) {
			end = PRINT_LIMIT;
		} else {
			end = size().intValue();
		}

		String string = "[";
		for (int n = 0; n < end; n++) {
			string += (n != 0 ? "," : "") + get(BigInteger.valueOf(n));
		}
		BigInteger remain = size().subtract(BigInteger.valueOf(end));
		if (remain.signum() > 0) {
			string += ",..." + remain + " more...";
		}
		string += "]";
		return string;
	}

	@Override
	public E get(long index) {
		return get(BigInteger.valueOf(index));
	}

}
