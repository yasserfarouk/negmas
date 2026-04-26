package negotiator.boaframework.opponentmodel.nash;

/**
 * 
 * This class represents a range for decimal values. It offers a lower- and upperbound 
 * that indicate the boundaries of the range.
 * 
 * @author Roland van der Linden
 *
 */
public class Range
{
	// ********************************************
	// Fields
	// ********************************************
	
	public double lowerbound, upperbound;
	
	
	// ********************************************
	// Constructor & init
	// ********************************************
	
	/**
	 * This constructs the range with both boundaries set to zero.
	 */
	public Range()
	{
		this(0, 0);
	}
	
	/**
	 * This constructs the range with the specified boundaries.
	 * @param lowerbound The lowerbound of the range.
	 * @param upperbound The upperbound of the range.
	 */
	public Range(double lowerbound, double upperbound)
	{
		this.lowerbound = lowerbound;
		this.upperbound = upperbound;
	}
	
	
	// ********************************************
	// Other methods.
	// ********************************************
	
	/**
	 * This returns the length of the range.
	 * @return
	 */
	public double getLength()
	{
		return this.upperbound - this.lowerbound;
	}
	
	/**
	 * This method specifies whether or not the given value falls within the boundaries
	 * of this range. Note that both boundary values are INCLUSIVE.
	 * @param value The value to test for if it is within the range.
	 * @return True if the value is within the range.
	 */
	public boolean withinBounds(double value)
	{
		return (value >= this.lowerbound && value <= this.upperbound);
	}
	
	/**
	 * A String representation of the range.
	 */
	public String toString()
	{
		return "(" + lowerbound + ", " + ")";
	}
}