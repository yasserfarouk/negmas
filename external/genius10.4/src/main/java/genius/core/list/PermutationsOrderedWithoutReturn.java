package genius.core.list;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

/**
 * Creates all ordered permutations without return.
 * 
 * @param <E>
 *            type of the elements
 */
public class PermutationsOrderedWithoutReturn<E> extends AbstractPermutations<E> {
	private BigInteger size;

	public PermutationsOrderedWithoutReturn(ImmutableList<E> list, int n) {
		super(list, n);
		if (BigInteger.valueOf(n).compareTo(list.size()) > 0)
			throw new IllegalArgumentException("n bigger than list size");

		size = MathTools.over(drawlistsize, drawsize);
	}

	@Override
	public ImmutableList<E> get(final BigInteger index) {
		/**
		 * IMPLEMENTATION NOTE.
		 * 
		 * Let N be the remaining number of elements in our drawlist (after
		 * having drawn out a few already) and k the number of elements that
		 * still need to be picked to make up the total drawsize.
		 * <p>
		 * The number of ways to draw k items from a set of size N is N over k.
		 * So there are (N-1) over k possible ways to fill up the rest of the
		 * draws without using the current one.
		 * 
		 * <p>
		 * The crucial notion is that the index will have increased by (N-1)
		 * over k by the time the current item is to be selected. Thus, we can
		 * devide and conquer the problem and reach o(log n) complexity.
		 * <p>
		 * Concrete example. Draw size 3, drawlistsize=6. index=13. We notice
		 * that N-1 over k = 5 over 3 =10 so the first index that has element 1
		 * selected is 10. Since index=13 > 10, we have first item selected and
		 * we need to be 3 after this point. Repeating the procedure, drawsize k
		 * is now 2, drawlistsize=5. N-1 over k is thus 4 over 2 = 6 and this is
		 * bigger than 3. So the 2nd item is not selected. Doing the procedure
		 * again, drawsize still 2, drawlistsize now 4, we have 3 over 2=3. This
		 * is exactly equal to the remaining 3, so we have item 3 also selected
		 * with 0 remaining. 0 remeaining means the last item is also selected.
		 * So index 13 means selection (1,3, 6).
		 */
		List<E> element = new ArrayList<E>();
		int k = drawsize;
		BigInteger remainingIndex = index;

		/**
		 * Invariant: n=remaining number of elements. k is remaining items
		 * needed to reach drawsize.
		 */
		for (int n = drawlistsize - 1; n >= 0; n--) {
			BigInteger nOverK = MathTools.over(n, k);
			if (n + 1 == k || remainingIndex.compareTo(nOverK) >= 0) {
				// found the next element.
				element.add(drawlist.get(BigInteger.valueOf(drawlistsize - n - 1)));
				remainingIndex = remainingIndex.subtract(nOverK);
				k = k - 1;
				if (k == 0)
					break;

			}
		}

		return new ImArrayList<E>(element);
	}

	@Override
	public BigInteger size() {
		return size;
	}

}
