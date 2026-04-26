package agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded;

import java.util.HashMap;
import java.util.List;

/**
 * This is the abstract class which determines when the opponent model
 * may be updated, and how it used to select a bid for the opponent.
 * 
 * @author Mark Hendrikx
 */
public abstract class OMStrategy {
	
	/** Reference to the object which holds all information about the negotiation */
	protected NegotiationSession negotiationSession;
	/** Reference to the opponent model */
	protected OpponentModel model;
	/** Increment used to increase the upperbound in case no bid is found in the range */
	private final double RANGE_INCREMENT = 0.05;
	/** Amount of bids used in a variant of getBid. A higher value results in a slower strategy, but
	 * more accuractly following the Pareto line.
	 */
	private final int EXPECTED_BIDS_IN_WINDOW = 25;
	
	public void init(NegotiationSession negotiationSession, OpponentModel model, HashMap<String, Double> parameters) throws Exception {
		this.negotiationSession = negotiationSession;
		this.model = model;
	}
	
	public void init(NegotiationSession negotiationSession, OpponentModel model) {
		this.negotiationSession = negotiationSession;
		this.model = model;
	}
	
	/**
	 * Returns a bid selected using the opponent model from the given
	 * set of similarly preferred bids.
	 * 
	 * @param bidsInRange set of similarly preferred bids
	 * @return bid
	 */
	public abstract BidDetails getBid(List<BidDetails> bidsInRange);
	
	/**
	 * Returns a bid selected using the opponent model with a utility
	 * in the given range.
	 * 
	 * @param space of all possible outcomes
	 * @param range of utility
	 * @return bid
	 */
	public BidDetails getBid(OutcomeSpace space, Range range) {
		List<BidDetails> bids = space.getBidsinRange(range);
		if (bids.size() == 0) {
			if (range.getUpperbound() < 1.1) {
				range.increaseUpperbound(RANGE_INCREMENT);
				return getBid(space, range);
			} else {
				negotiationSession.setOutcomeSpace(space);
				return negotiationSession.getMaxBidinDomain();
			}
		}
		return getBid(bids);
	}

	/**
	 * Use this method in case no range is specified, but only a target utility.
	 * In this case first a small window is used, which is enlarged if there are
	 * too few bids in the window.
	 * 
	 * @param space of all possible outcomes
	 * @param range of utility
	 * @return bid
	 */
	public BidDetails getBid(OutcomeSpace space, double targetUtility) {
		Range range = new Range(targetUtility, targetUtility + 0.02);
		List<BidDetails> bids = space.getBidsinRange(range);
		if (bids.size() < EXPECTED_BIDS_IN_WINDOW) {
			if (range.getUpperbound() < 1.01) {
				range.increaseUpperbound(RANGE_INCREMENT);
				return getBid(space, range);
			} else {
				// futher increasing the window does not help
				return getBid(bids);
			}
		}
		return getBid(bids);
	}
	
	public abstract boolean canUpdateOM();
}