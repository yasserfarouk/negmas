package genius.core.list;

/**
 * Abstract base class for Permutations, common providing standard functions.
 * 
 * @param <E>
 *            type of the elements
 */
public abstract class AbstractPermutations<E> extends AbstractImmutableList<ImmutableList<E>>
		implements Permutations<E> {

	final protected ImmutableList<E> drawlist;
	final protected int drawlistsize;
	final protected int drawsize;

	/**
	 * all permutations of a given list, drawing n items from the list
	 * 
	 * @param list
	 *            list to be permuted
	 * @param n
	 *            the number of items to draw from the list. Must be between 0
	 *            and size of list.
	 */
	public AbstractPermutations(ImmutableList<E> list, int n) {
		if (n < 0)
			throw new IllegalArgumentException("n<0");
		if (list.size().bitLength() > 31) {
			throw new IllegalArgumentException("excessive list size detected for starting a permutation" + list.size());
		}
		this.drawlist = list;
		this.drawlistsize = list.size().intValue();
		this.drawsize = n;
	}

}
