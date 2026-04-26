package genius.core.list;

import java.math.BigInteger;

/**
 * Generate list of Tuple[T1, T2] with all combinations of 1 element from list1
 * and one from list2. Sometimes called "cartesian product". You can use this
 * recursively to generate bigger products; but you can also use {@link Outer}.
 * 
 * @param <T1>
 *            type of the first element of the tuple
 * @param <T2>
 *            type of the second element of the tuple
 */
public class Tuples<T1, T2> extends AbstractImmutableList<Tuple<T1, T2>> {

	private ImmutableList<T1> list1;
	private ImmutableList<T2> list2;
	private BigInteger size;

	/**
	 * contains all possible tuples with first element from list1 and second
	 * from list2
	 * 
	 * @param list1
	 *            first element list
	 * @param list2
	 *            second element list
	 */
	public Tuples(ImmutableList<T1> list1, ImmutableList<T2> list2) {
		this.list1 = list1;
		this.list2 = list2;
		this.size = list1.size().multiply(list2.size());
	}

	@Override
	public Tuple<T1, T2> get(BigInteger index) {
		BigInteger[] indices = index.divideAndRemainder(list1.size());
		return new Tuple<T1, T2>(list1.get(indices[1]), list2.get(indices[0]));
	}

	@Override
	public BigInteger size() {
		return size;
	}

}
