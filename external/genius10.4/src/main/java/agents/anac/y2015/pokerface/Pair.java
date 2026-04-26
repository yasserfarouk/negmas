package agents.anac.y2015.pokerface;

@SuppressWarnings("hiding")
public class Pair<Integer,Value> {
	
	private final Integer i;
	private final Value val;

	public Pair(Integer i, Value val) {
		this.i = i;
		this.val = val;
	}

	public Integer getInteger() { return i; }
	public Value getValue() { return val; }

	@Override
	public int hashCode() { return i.hashCode() ^ val.hashCode(); }

	@Override
	public boolean equals(Object o) {
		if (o == null) return false;
		if (!(o instanceof Pair)) return false;
		@SuppressWarnings("unchecked")
		Pair<Integer, Value> pairo = (Pair<Integer, Value>) o;
		return this.i.equals(pairo.getInteger()) &&
				this.val.equals(pairo.getValue());
	}

}