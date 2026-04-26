package genius.core.list;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

/**
 * creates all possible combinations of one element from each of the provided
 * lists
 * 
 * @param T
 *            the element type of all lists that we receive.
 */
public class Outer<T> extends AbstractImmutableList<ImmutableList<T>> {

	private BigInteger size = BigInteger.ZERO;
	private List<ImmutableList<T>> sourceLists = new ArrayList<>();

	public Outer(ImmutableList<T>... lists) {
		if (lists.length == 0) {
			return;
		}
		size = BigInteger.ONE;
		for (ImmutableList<T> l : lists) {
			sourceLists.add(l);
			size = size.multiply(l.size());
		}
	}

	@Override
	public ImmutableList<T> get(BigInteger index) {
		List<T> element = new ArrayList<T>();
		BigInteger i = index;
		for (int item = 0; item < sourceLists.size(); item++) {
			ImmutableList<T> l = sourceLists.get(item);
			BigInteger n = l.size();
			element.add(l.get(i.mod(n)));
			i = i.divide(n);
		}
		return new ImArrayList<T>(element);
	}

	@Override
	public BigInteger size() {
		return size;
	}

}
