package genius.core.list;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;

/**
 * makes a shuffled version of provided list.
 * 
 *
 * @param <E>
 *            type of the elements
 */
public class ShuffledList<E> extends AbstractImmutableList<E> {

	private ImmutableList<E> list;
	private ArrayList<BigInteger> newIndex = null;

	/**
	 * 
	 * @param l
	 *            list to shuffle. size must be &le; 2^31.
	 */
	public ShuffledList(ImmutableList<E> l) {
		l.size().intValueExact(); // just to make sure we can convert it to int.
		this.list = l;
	}

	/**
	 * This call can be expensive. The actual shuffling is done the first time
	 * you call this.
	 * 
	 */
	@Override
	public E get(BigInteger index) {
		if (newIndex == null)
			initIndices();
		return list.get(newIndex.get(index.intValueExact()));
	}

	/**
	 * generates the newIndex list. This is expensive.
	 */
	private void initIndices() {
		int size = list.size().intValue();
		newIndex = new ArrayList<>();
		for (int n = 0; n < size; n++) {
			newIndex.add(BigInteger.valueOf(n));
		}
		Collections.shuffle(newIndex);
	}

	@Override
	public BigInteger size() {
		return list.size();
	}

}
