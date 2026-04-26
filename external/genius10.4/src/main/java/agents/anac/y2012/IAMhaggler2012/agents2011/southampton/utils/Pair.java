package agents.anac.y2012.IAMhaggler2012.agents2011.southampton.utils;

import java.io.Serializable;

public class Pair<A, B> implements Serializable {

	private static final long serialVersionUID = -3160841691791187933L;
	public final A fst;
	public final B snd;

	/**
	 * @param fst
	 *            First value.
	 * @param snd
	 *            Second value.
	 */
	public Pair(A fst, B snd) {
		this.fst = fst;
		this.snd = snd;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		return "Pair[" + fst + "," + snd + "]";
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#equals(java.lang.Object)
	 */
	public boolean equals(Object other) {
		return other instanceof Pair<?, ?> && equals(fst, ((Pair<?, ?>) other).fst) && equals(snd, ((Pair<?, ?>) other).snd);
	}

	private static boolean equals(Object x, Object y) {
		return (x == null && y == null) || (x != null && x.equals(y));
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#hashCode()
	 */
	public int hashCode() {
		if (fst == null)
			return (snd == null) ? 0 : snd.hashCode() + 1;
		else if (snd == null)
			return fst.hashCode() + 2;
		else
			return fst.hashCode() * 17 + snd.hashCode();
	}

	public static <A, B> Pair<A, B> of(A a, B b) {
		return new Pair<A, B>(a, b);
	}
}