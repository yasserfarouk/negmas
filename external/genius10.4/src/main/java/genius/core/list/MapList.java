package genius.core.list;

import java.math.BigInteger;

/**
 *
 * @param <IN1>
 *            the type of the elements inside the argument list
 * @param <OUT>
 *            the output type of the function
 */
public class MapList<IN1, OUT> extends AbstractImmutableList<OUT> implements ImmutableList<OUT> {

	private ImmutableList<IN1> list1;
	private Function<IN1, OUT> f;

	/**
	 * creates a list [f(a1), f(a2) ,. ..., f(an)].
	 * 
	 * @param f
	 *            function
	 * @param list1
	 *            a list of items [a1,a2,..., an]
	 */
	public MapList(Function<IN1, OUT> f, ImmutableList<IN1> list1) {
		if (f == null || list1 == null)
			throw new NullPointerException("null argument");
		this.list1 = list1;
		this.f = f;
	}

	@Override
	public OUT get(BigInteger index) {
		return f.apply(list1.get(index));
	}

	@Override
	public BigInteger size() {
		return list1.size();
	}

}
