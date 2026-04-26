package negotiator.boaframework.offeringstrategy.other;

import java.util.Map;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.SortedOutcomeSpace;

/**
 * This class implements an offering strategy which creates a list of possible
 * bids and then offers them in descending order. If all bids are offered, then
 * the last bid is repeated.
 * 
 * This strategy has no straight-forward extension of using opponent models.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 * @version 15-12-11
 */
public class ChoosingAllBids extends OfferingStrategy {
	/**
	 * counter used to determine which bid to offer from the sorted list of
	 * possible bids
	 */
	private int counter = 0;

	/**
	 * Empty constructor used for reflexion. Note this constructor assumes that
	 * init is called next.
	 */
	public ChoosingAllBids() {
	}

	/**
	 * Constructor which can be used to createFrom the agent without the GUI.
	 * 
	 * @param negoSession
	 *            reference to the negotiationsession object
	 * @param model
	 *            reference to the opponent model
	 */
	public ChoosingAllBids(NegotiationSession negoSession, OpponentModel model) {
		initializeAgent(negoSession, model);
	}

	@Override
	public void init(NegotiationSession domainKnow, OpponentModel model, OMStrategy omStrat,
			Map<String, Double> parameters) throws Exception {
		initializeAgent(domainKnow, model);
	}

	private void initializeAgent(NegotiationSession negoSession, OpponentModel model) {
		this.negotiationSession = negoSession;

		SortedOutcomeSpace space = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
		negotiationSession.setOutcomeSpace(space);
	}

	/**
	 * Returns the next bid in the sorted array of bids. If there are no more
	 * bids in the list, then the last bid is returned.
	 */
	@Override
	public BidDetails determineNextBid() {
		nextBid = negotiationSession.getOutcomeSpace().getAllOutcomes().get(counter);
		if (counter < negotiationSession.getOutcomeSpace().getAllOutcomes().size() - 1) {
			counter++;
		}

		return nextBid;
	}

	@Override
	public BidDetails determineOpeningBid() {
		return determineNextBid();
	}

	@Override
	public String getName() {
		return "Other - ChoosingAllBids";
	}
}