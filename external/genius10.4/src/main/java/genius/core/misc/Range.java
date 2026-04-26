package genius.core.misc;


/**
 * This is a tuple class which is used to pass on a double range.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class Range {

	/** Lower bound of the specified window */
	private double lowerbound;
	/** Upper bound of the specified window */
	private double upperbound;
	
	/**
	 * Specifies a continuous range.
	 * 
	 * @param lowerbound of the range.
	 * @param upperbound of the range.
	 */
	public Range(double lowerbound, double upperbound){
		this.lowerbound = lowerbound;
		this.upperbound = upperbound;
	}
	
	/**
	 * Returns the upperbound of the range.
	 * @return upperbound of range.
	 */
	public double getUpperbound(){
		return upperbound;
	}
	
	/**
	 * Returns the lowerbound of the range.
	 * @return lowerbound of range.
	 */
	public double getLowerbound(){
		return lowerbound;
	}
	
	/**
	 * Set the upperbound of the range.
	 * @param upperbound of the range.
	 */
	public void setUpperbound(double upperbound){
		this.upperbound = upperbound;
	}
	
	/**
	 * Set the lowerbound of the range.
	 * @param lowerbound of the range.
	 */
	public void setLowerbound(double lowerbound){
		this.lowerbound = lowerbound;
	}
	
	/**
	 * Increases the upperbound by the given increment.
	 * @param increment amount which should be added to the upperbound.
	 */
	public void increaseUpperbound(double increment) {
		upperbound = upperbound + increment;
	}
}