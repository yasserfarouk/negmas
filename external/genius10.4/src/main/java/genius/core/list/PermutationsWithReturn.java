package genius.core.list;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

/**
 * Creates all permutations with return (WR).
 * 
 * @param <E>
 *            type of the elements
 */
public class PermutationsWithReturn<E> extends AbstractPermutations<E> {
	private BigInteger size;

	public PermutationsWithReturn(ImmutableList<E> list, int n) {
		super(list, n);
		this.size = list.size().pow(drawsize);

	}

	@Override
	public ImmutableList<E> get(final BigInteger index) {
		/**
		 * Ok this is easy, we just convert the number to base <drawsize>. We do
		 * this simply using the modulo operator.
		 */
		List<E> element = new ArrayList<E>();
		BigInteger i = index;
		for (int item = 0; item < drawsize; item++) {
			element.add(drawlist.get(i.mod(BigInteger.valueOf(drawlistsize))));
			i = i.divide(BigInteger.valueOf(drawlistsize));
		}
		return new ImArrayList<E>(element);
	}

	@Override
	public BigInteger size() {
		return size;
	}

}
