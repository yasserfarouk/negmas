package negotiator.boaframework.omstrategy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.bidding.BidDetailsSorterUtility;
import genius.core.boaframework.BOAparameter;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OpponentModel;

/**
 * This class uses an opponent model to determine the next bid for the opponent,
 * while taking the opponent's preferences into account. The opponent model is
 * used to select the N best bids. Following, a random bid is selected from this
 * subset. Setting N > 1 is rational, as opponent models cannot be assumed to be
 * perfect.
 * 
 * Due to performance reasons, it is recommended to use BestBid if N = 1.
 * 
 * @author Mark Hendrikx
 */
public class OfferBestN extends OMStrategy {

	private Random rand;
	/** parameter which determines which n best bids should be considered */
	private int bestN;
	/** used to sort the opponent's bid with regard to utility */
	private BidDetailsSorterUtility comp = new BidDetailsSorterUtility();
	/** when to stop updating */
	double updateThreshold = 1.1;

	/**
	 * Initializes the agent by storing the size of the domain, and checking if
	 * the domain is large.
	 * 
	 * @param negotiationSession
	 *            state of the negotiation.
	 * @param model
	 *            opponent model used in conjunction with this opponent modeling
	 *            strategy.
	 * @param parameters
	 *            set of parameters for this opponent model strategy.
	 */
	@Override
	public void init(NegotiationSession negotiationSession, OpponentModel model, Map<String, Double> parameters) {
		super.init(negotiationSession, model, parameters);
		this.negotiationSession = negotiationSession;
		this.model = model;
		if (parameters.get("n") != null) {
			int n = parameters.get("n").intValue();
			if (n == 1) {
				throw new IllegalArgumentException("For \"n\"=1 use BestBid instead.");
			}
			initializeAgent(negotiationSession, model, n);
		} else {
			throw new IllegalArgumentException("Constant \"n\" for amount of best bids was not set.");
		}
		if (parameters.get("t") != null) {
			updateThreshold = parameters.get("t").doubleValue();
		} else {
			System.out.println("OMStrategy assumed t = 1.1");
		}
	}

	private void initializeAgent(NegotiationSession negotiationSession, OpponentModel model, int n) {
		try {
			super.init(negotiationSession, model, new HashMap<String, Double>());
		} catch (Exception e) {
			e.printStackTrace();
		}
		this.rand = new Random();
		this.bestN = n;
	}

	/**
	 * First this method determines the N best bids given the array of similarly
	 * preferred bids. Next, a random bid is offered from this set.
	 * 
	 * @param allBids
	 *            list of similarly preferred bids.
	 * @return random bid from the subset of N best bids in the given set.
	 */
	@Override
	public BidDetails getBid(List<BidDetails> allBids) {
		// 1. Determine the utility for the opponent for each of the bids
		ArrayList<BidDetails> oppBids = new ArrayList<BidDetails>(allBids.size());
		for (BidDetails bidDetail : allBids) {
			Bid bid = bidDetail.getBid();
			BidDetails newBid = new BidDetails(bid, model.getBidEvaluation(bid), negotiationSession.getTime());
			oppBids.add(newBid);
		}

		// 2. Sort the bids on the utility for the opponent
		Collections.sort(oppBids, comp);

		// 3. Select a random bid from the N best bids and offer this bid
		int entry = rand.nextInt(Math.min(bestN, oppBids.size()));
		Bid opponentBestBid = oppBids.get(entry).getBid();
		BidDetails nextBid = null;
		try {
			nextBid = new BidDetails(opponentBestBid, negotiationSession.getUtilitySpace().getUtility(opponentBestBid),
					negotiationSession.getTime());
		} catch (Exception e) {
			e.printStackTrace();
		}
		return nextBid;
	}

	/**
	 * Specifies that the opponent model may be updated when the current time is
	 * smaller than the deadline.
	 * 
	 * @return true if the model may be updated
	 */
	@Override
	public boolean canUpdateOM() {
		return negotiationSession.getTime() < updateThreshold;
	}

	@Override
	public Set<BOAparameter> getParameterSpec() {
		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("n", 3.0, "A random bid is selected from the best n bids"));
		set.add(new BOAparameter("t", 1.1, "Time after which the OM should not be updated"));
		return set;
	}

	@Override
	public String getName() {
		return "Offer Best N";
	}
}