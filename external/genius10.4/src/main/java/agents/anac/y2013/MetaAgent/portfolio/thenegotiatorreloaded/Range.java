package agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded;


/**
 * This is a tuple class which is used to pass on a target utility range.
 * 
 * @author Alex Dirkzwager
 */
public class Range {

	/** Lower bound of the specified window */
	private double lowerbound;
	/** Upper bound of the specified window */
	private double upperbound;
	
	public Range(double lowerbound, double upperbound){
		this.lowerbound = lowerbound;
		this.upperbound = upperbound;
	}
	
	public double getUpperbound(){
		return upperbound;
	}
	
	public double getLowerbound(){
		return lowerbound;
	}
	
	public void setUpperbound(double ubound){
		upperbound = ubound;
	}
	
	public void setLowerbound(double lbound){
		lowerbound = lbound;
	}
	
	/**
	 * Enlarges the upper bound by the given increment.
	 * 
	 * @param increment
	 */
	public void increaseUpperbound(double increment) {
		upperbound = upperbound + increment;
	}
}
