package genius.core.misc;

import java.io.Serializable;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Simple class which can be used to keep track of the score of a set of
 * objects. An example of using this class is to count how many times a set of
 * values has been offered by the opponent.
 * 
 * @author Tim Baarslag, Mark Hendrikx
 *
 * @param <A>
 *            key of the hashmap, for example a Value-object
 */
public class ScoreKeeper<A> implements Comparator<A>, Serializable {

	private static final long serialVersionUID = -8974661138458056269L;
	/** Map of objects and their score **/
	protected Map<A, Integer> m;
	/** The highest score in the map **/
	protected int max;
	/** The sum of all scores in the map **/
	protected int total;

	/**
	 * Creates a ScoreKeeper object by initializing the hashmap.
	 */
	public ScoreKeeper() {
		m = new HashMap<A, Integer>();
		total = 0;
		max = 0;
	}

	/**
	 * Clones the given scorekeeper-object.
	 * 
	 * @param sk
	 *            object keeper which is cloned.
	 */
	public ScoreKeeper(ScoreKeeper<A> sk) {
		this.m = sk.m;
		this.max = sk.max;
		this.total = sk.total;
	}

	/**
	 * Adds one to the score of the given object.
	 * 
	 * @param a
	 *            object to which one must be added to its score.
	 */
	public void score(A a) {
		Integer freq = m.get(a);
		if (freq == null)
			freq = 0;
		freq++;
		if (freq > max) {
			max = freq;
		}
		total++;
		m.put(a, freq);
	}

	/**
	 * Method used to add a given score to a given object.
	 * 
	 * @param a
	 *            object to which the given score must be added.
	 * @param score
	 *            to be added to the object.
	 */
	public void score(A a, int score) {
		Integer freq = m.get(a);
		if (freq == null)
			freq = 0;
		int newValue = freq + score;
		if (newValue > max) {
			max = newValue;
		}
		total += score;
		m.put(a, newValue);
	}

	/**
	 * Returns the score of the given object.
	 * 
	 * @param a
	 *            object from which the score must be returned.
	 * @return score of the object.
	 */
	public int getScore(A a) {
		Integer freq = m.get(a);
		if (freq == null)
			freq = 0;
		return freq;
	}

	/**
	 * Returns the normalized score of the given object. The normalized score is
	 * defined as the score divided by the highest score in the map.
	 * 
	 * @param a
	 *            the object from which the score is requested.
	 * @return score of the object divided by the highest score.
	 */
	public double getNormalizedScore(A a) {
		Integer score = m.get(a);
		if (score == null) {
			score = 0;
		}
		return ((double) score / (double) max);
	}

	/**
	 * Returns the relative score of a given object. The relative score is the
	 * score of the object divided by the sum of all scores in the map.
	 * 
	 * @param a
	 *            object from which the score must be returned
	 * @return score of the object divided by the sum of all scores
	 */
	public double getRelativeScore(A a) {
		Integer score = m.get(a);
		if (score == null) {
			score = 0;
		}
		return ((double) score / (double) total);
	}

	/**
	 * Comparator to compare the score of two objects.
	 * 
	 * @return -1 iff score o1 &gt; o2, 1 if vice versa, else 0.
	 */
	public int compare(final A o1, final A o2) {
		if (o1 == null || o2 == null)
			throw new NullPointerException();
		if (o1.equals(o2))
			return 0;
		if (getScore(o1) > getScore(o2))
			return -1;
		else if (getScore(o1) < getScore(o2))
			return 1;
		else
			return ((Integer) o1.hashCode()).compareTo(o2.hashCode());
	}

	/**
	 * @return string representation of the ScoreKeeper.
	 */
	public String toString() {
		TreeMap<A, Integer> sorted = getSortedCopy();
		return getElements().size() + " entries, " + getTotal() + " total: " + sorted.toString() + "\n";
	}

	/**
	 * @return sorted version of the ScoreKeeper based on the score of the
	 *         elements.
	 */
	public TreeMap<A, Integer> getSortedCopy() {
		TreeMap<A, Integer> sorted = new TreeMap<A, Integer>(this);
		sorted.putAll(m);
		return sorted;
	}

	/**
	 * Returns the highest score in the map.
	 * 
	 * @return score of the object with the highest score.
	 */
	public int getMaxValue() {
		return max;
	}

	/**
	 * Returns the sum of all scores.
	 * 
	 * @return sum of all scores
	 */
	public int getTotal() {
		int total = 0;
		for (A a : m.keySet())
			total += m.get(a);
		return total;
	}

	/**
	 * Returns the objects from which the score is registered.
	 * 
	 * @return objects in the scoring map
	 */
	public Set<A> getElements() {
		return m.keySet();
	}

	/**
	 * Returns a Mathematica list plot of the map.
	 * 
	 * @return Mathematica list plot of the map.
	 */
	public String toMathematicaListPlot() {
		StringBuilder s = new StringBuilder("data={");
		boolean first = true;
		TreeSet<A> sortedKeys = new TreeSet<A>(m.keySet());
		for (A entry : sortedKeys) {
			if (first)
				first = false;
			else
				s.append(",");

			if (entry instanceof Number) {
				s.append("{" + entry + "," + m.get(entry) + "}");
			}

			if (entry instanceof String) {
				s.append("{\"" + entry + "\"," + m.get(entry) + "}");
			}
		}
		s.append("};\n");
		s.append("ListPlot[data, PlotRange -> All]");
		return s.toString();
	}
}