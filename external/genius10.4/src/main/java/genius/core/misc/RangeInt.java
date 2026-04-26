package genius.core.misc;

import java.io.Serializable;

/**
 * This is a tuple class which is used to pass on an integer range.
 */
public class RangeInt implements Serializable{
	
	/** The lowerbound of the specified range. */
	int lowerbound;
	/** The upperbound of the specified range. */
	int upperbound;
	
	/**
	 * Specifies a discrete range.
	 * 
	 * @param lowerbound of the range.
	 * @param upperbound of the range.
	 */
	public RangeInt(int lowerbound, int upperbound) {
		this.lowerbound = lowerbound;
		this.upperbound = upperbound;
	}
	
	/**
	 * Returns the lowerbound of the range.
	 * @return lowerbound of range.
	 */
	public int getLowerbound() {
		return lowerbound;
	}
	
	/**
	 * Returns the upperbound of the range.
	 * @return upperbound of range.
	 */
	public int getUpperbound() {
		return upperbound;
	}
	
	/**
	 * Set the upperbound of the range.
	 * @param upperbound of the range.
	 */
	public void setUpperbound(int upperbound){
		this.upperbound = upperbound;
	}
	
	/**
	 * Set the upperbound of the range.
	 * @param lowerbound of the range.
	 */
	public void setLowerbound(int lowerbound){
		this.lowerbound = lowerbound;
	}
}