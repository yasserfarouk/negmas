package genius.core.tournament.VariablesAndValues;

public class TotalSessionNumberValue extends TournamentValue {

	private static final long serialVersionUID = 1577274902162703852L;
	private int value = 1;
	public TotalSessionNumberValue() {
	
	}
	public TotalSessionNumberValue(int value) {
		this.value = value;
	}
	public String toString() {return String.valueOf(value);}
	public int getValue() { return value; }
}
