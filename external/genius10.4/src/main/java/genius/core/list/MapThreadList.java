package genius.core.list;

import java.math.BigInteger;

/**
 *
 * @param <OUT>
 *            output type of function and element type of resulting list
 * @param <IN1>
 *            fype of first input parameter of function and of elements in first
 *            list
 * @param <IN2>
 *            type of second input parameter of function and of elements in
 *            second list
 */
public class MapThreadList<OUT, IN1, IN2> extends AbstractImmutableList<OUT> {

	private ImmutableList<IN1> list1;
	private ImmutableList<IN2> list2;
	private Function2<IN1, IN2, OUT> f;

	/**
	 * creates a list [f(a1,b1), f(a2, b2) ,. ..., f(an, bb)].
	 * 
	 * @param f
	 *            function
	 * @param list1
	 *            a list of items [a1,a2,..., an]
	 * @param list2
	 *            a list of items [b1, b2,...,bn] of same length as list1
	 */
	public MapThreadList(Function2<IN1, IN2, OUT> f, ImmutableList<IN1> list1, ImmutableList<IN2> list2) {
		if (f == null || list1 == null || list2 == null)
			throw new NullPointerException("null argument");
		if (!list1.size().equals(list2.size()))
			throw new IllegalArgumentException("lists are unequal size");
		this.list1 = list1;
		this.list2 = list2;
		this.f = f;
	}

	@Override
	public OUT get(BigInteger index) {
		return f.apply(list1.get(index), list2.get(index));
	}

	@Override
	public BigInteger size() {
		return list1.size();
	}

}
