package genius.core;

import genius.core.actions.Accept;
import genius.core.actions.Action;

public class NegotiationResult {

	/** The utility received by the agent at the end of the negotiation. */
	private final double myDiscountedUtility;
	/** The last action conducted in the negotiation. */
	private final Action lastAction;
	/** The last bid in the negotiation. */
	private final Bid lastBid;
	
	public NegotiationResult(double myDiscountedUtility, Action lastAction, Bid lastBid) {
		this.myDiscountedUtility = myDiscountedUtility;
		this.lastAction = lastAction;
		this.lastBid = lastBid;
	}
	
	/**
	 * @return true when the match ended in acceptance.
	 */
	public boolean isAgreement() {
		return lastAction instanceof Accept;
	}
	
	/**
	 * @return the utility received at the end of the negotiation.
	 */
	public double getMyDiscountedUtility() {
		return myDiscountedUtility;
	}
	
	/**
	 * @return last action executed in the negotiation.
	 */
	public Action getLastAction() {
		return lastAction;
	}
	
	/**
	 * @return last bid offered in the negotiation.
	 */
	public Bid getLastBid() {
		return lastBid;
	}
	
	public String toString() {
		return myDiscountedUtility + "\n" + lastAction + "\n" + lastBid;
	}
}