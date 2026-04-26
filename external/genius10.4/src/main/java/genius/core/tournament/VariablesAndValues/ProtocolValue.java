package genius.core.tournament.VariablesAndValues;

import genius.core.repository.ProtocolRepItem;

public class ProtocolValue extends TournamentValue {
	private static final long serialVersionUID = -2565640538778156974L;
	ProtocolRepItem value;	
	
	public ProtocolValue(ProtocolRepItem val) { value=val; }
	public String toString() { return value.getName(); }
	public ProtocolRepItem getValue() { return value; }
}
