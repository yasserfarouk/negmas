package agents.anac.y2016.farma.etc;

import java.util.HashMap;
import java.util.Map;

class Pair<F, S> {
	public final F first;
	public final S second;

	Pair(F first, S second) {
		this.first = first;
		this.second = second;
	}

	@Override
	public boolean equals(Object obj) {
		if (!(obj instanceof Pair))
			return false;
		Pair pair = (Pair) obj;
		return (first.equals(pair.first) && second.equals(pair.second));
	}

	@Override
	public int hashCode() {
		return first.hashCode() ^ second.hashCode();
	}

	public static void main(String[] args) {
		Map<Pair<Integer, Integer>, String> map = new HashMap<Pair<Integer, Integer>, String>();

		Pair pair = new Pair(1, 2);
		Pair pair2 = new Pair(1, 2);
		map.put(pair, "a");

		if (map.containsKey(pair2)) {
			System.out.println("equal");
		} else {
			System.out.println("not equal");
		}
	}
}