package genius.core.tournament.VariablesAndValues;

/**
 * @author tim
 * Used for varying values in experiments.
 */
public class ExperimentalValue extends TournamentValue
{
	private static final long serialVersionUID = -7690627644862645404L;
	double value;	
	
	public ExperimentalValue(double val) { value=val; }
	public String toString() { return value+""; }
	public double getValue(){ return value;	}
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		long temp;
		temp = Double.doubleToLongBits(value);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		return result;
	}
	
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		ExperimentalValue other = (ExperimentalValue) obj;
		if (Double.doubleToLongBits(value) != Double
				.doubleToLongBits(other.value))
			return false;
		return true;
	}
	
	
}