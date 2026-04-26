package agents.anac.y2015.pnegotiator;

/**
 * Created by chad on 2/26/15.
 */
public class Tuple<K, V extends Comparable<V>> implements
		Comparable<Tuple<K, V>> {
	K key;
	V value;

	public Tuple(K key, V value) {
		this.key = key;
		this.value = value;
	}

	@Override
	public int compareTo(Tuple<K, V> t2) {
		return value.compareTo(t2.value);
	}
}
