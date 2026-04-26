package genius.core.tournament.VariablesAndValues;

public class ProtocolVariable extends TournamentVariable {

	@Override
	public void addValue(TournamentValue value) throws Exception {
		if (!(value instanceof ProtocolValue))
			throw new IllegalArgumentException("Expected ProtocolValue but received "+value);
		values.add(value);
	}

	@Override
	public String varToString() {
		return "Protocol";
	}

}
