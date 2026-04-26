package genius.core.list;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

/**
 * Creates all permutation without return (WOR).
 * 
 * @param <E>
 *            type of the elements
 */
public class PermutationsWithoutReturn<E> extends AbstractPermutations<E> {

	public PermutationsWithoutReturn(ImmutableList<E> list, int n) {
		super(list, n);
		if (BigInteger.valueOf(n).compareTo(list.size()) > 0)
			throw new IllegalArgumentException("n bigger than list size");
	}

	@Override
	public ImmutableList<E> get(final BigInteger index) {
		/**
		 * Method: we need to draw #drawsize items. The first has #drawlistsize
		 * choices, the second #drawlistsize-1 etc. So we convert #index into
		 * #drawsize indices, the first running up to #drawlistsize, the second
		 * up to #drawlistsize-1, etc.
		 */
		ListWithRemove<E> sublist = new ListWithRemove<>(drawlist);
		List<E> element = new ArrayList<E>();
		BigInteger i = index;
		for (int item = 0; item < drawsize; item++) {
			BigInteger[] divrem = i.divideAndRemainder(BigInteger.valueOf(drawlistsize - item));
			BigInteger im = divrem[1];
			element.add(sublist.get(im));
			sublist = sublist.remove(im);
			i = divrem[0];
		}
		return new ImArrayList<E>(element);
	}

	@Override
	public BigInteger size() {
		return MathTools.factorial(drawlistsize).divide(MathTools.factorial(drawlistsize - drawsize));
	}

}
