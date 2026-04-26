package genius.core.analysis;

import genius.core.Bid;

/**
 * Specialized version of a BidPoint for the case that there are two agents.
 * In this case, the time of offering the bid can be recorded.
 * 
 * @author Alex Dirkzwager
 */
public class BidPointTime extends BidPoint{
	
	/** Time at which the bid was offered to the opponent. */
	private double time;

	/**
	 * Create a BidPointTime object, which is a tuple of a specific
	 * bid, the utility of this bid for both agents, and the time at
	 * which the bid was offered.
	 * 
	 * @param bid of which the utilities are recorded.
	 * @param utilityA utility of the agent for agent A.
	 * @param utilityB utility of the agent for agent B.
	 * @param time at which the bid was offered.
	 */
	public BidPointTime(Bid bid, Double utilityA, Double utilityB, double time) {
		super(bid, utilityA, utilityB);
		this.time = time;
	}
	
    /**
     * @return string representation of the object..
     */
	@Override
	public String toString(){
		return "BidPointTime ["+getBid()+" utilA["+getUtilityA()+"],utilB["+getUtilityB()+"], Time["+time+"]]";
	}

    /**
     * @return hashcode of this object.
     */
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = super.hashCode();
		long temp;
		temp = Double.doubleToLongBits(time);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		return result;
	}

    /**
     * @param obj object to which this object is compared.
     * @return true if this object is equal to the given object.
     */
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (!super.equals(obj))
			return false;
		if (getClass() != obj.getClass())
			return false;
		BidPointTime other = (BidPointTime) obj;
		if (Double.doubleToLongBits(time) != Double
				.doubleToLongBits(other.time))
			return false;
		return true;
	}

	/**
	 * Returns the time at which the bid was offered.
	 * @return time of offering.
	 */
	public double getTime() {
		return time;
	}

	/**
	 * Sets the time at which the bid is offered.
	 * @param time of offering.
	 */
	public void setTime(double time) {
		this.time = time;
	}
}