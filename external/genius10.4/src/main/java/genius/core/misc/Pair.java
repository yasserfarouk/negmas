package genius.core.misc;

import java.io.Serializable;

/**
 * A simple tuple class.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx, Julian de Ruiter
 * @param <A> class of the first object.
 * @param <B> class of the second object.
 */
public class Pair<A, B> implements Serializable {

	private static final long serialVersionUID = -3269964369920187563L;
	/** Reference to the first object of the pair. */
	private A fst;
	/** Reference to the second object of the pair. */
	private B snd;

	/**
	 * Create a pair from the given two objects.
	 * @param fst first object of the pair.
	 * @param snd second object of the pair.
	 */
	public Pair(A fst, B snd) {
		this.fst = fst;
		this.snd = snd;
	}

	/**
	 * Return the first object of the pair.
	 * @return first object of the pair.
	 */
	public A getFirst() { return fst; }
	
	/**
	 * Return the second object of the pair.
	 * @return second object of the pair.
	 */
	public B getSecond() { return snd; }

	/**
	 * Set the first object of the pair.
	 * @param v set first object to v.
	 */
	public void setFirst(A v) { fst = v; }
	
	/**
	 * Set the second object of the pair.
	 * @param v set second object to v.
	 */
	public void setSecond(B v) { snd = v; }

	/**
	 * @return string representation of string.
	 */
	public String toString() {
		return "Pair[" + fst + "," + snd + "]";
	}

	private static boolean equals(Object x, Object y) {
		return (x == null && y == null) || (x != null && x.equals(y));
	}

	/**
	 * @return true if this and other object are equal.
	 */
	public boolean equals(Object other) {
		return
		other instanceof Pair &&
		equals(fst, ((Pair<?, ?>)other).fst) &&
		equals(snd, ((Pair<?, ?>)other).snd);
	}

	/**
	 * @return hashcode of this object.
	 */
	public int hashCode() {
		if (fst == null) return (snd == null) ? 0 : snd.hashCode() + 1;
		else if (snd == null) return fst.hashCode() + 2;
		else return fst.hashCode() * 17 + snd.hashCode();
	}
}