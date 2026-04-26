package negotiator.boaframework.omstrategy;

import java.util.List;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.bidding.BidDetailsSorterUtility;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OpponentModel;

/**
 * Implements the opponent model strategy used by the NiceTitForTat agent in the
 * ANAC2011. The strategy selects a random bid from the best N bids. What is
 * special in comparison to OfferBestN, is that N depends on the domain size.
 * Furthermore, the strategy stops updating the opponent model before the
 * deadline. The time at which the updating stops depends on the domain size.
 * 
 * This component is part of the NiceTitForTat strategy introduced by T.
 * Baarslag in the ANAC 2011.
 * 
 * @author Mark Hendrikx
 */
public class NTFTstrategy extends OMStrategy {

	private boolean domainIsBig;
	private long possibleBids;
	private Random random;
	private BidDetailsSorterUtility comp = new BidDetailsSorterUtility();

	@Override
	public void init(NegotiationSession negotiationSession, OpponentModel model, Map<String, Double> parameters) {
		initializeAgent(negotiationSession, model);
	}

	private void initializeAgent(NegotiationSession negoSession, OpponentModel model) {
		this.negotiationSession = negoSession;
		try {
			super.init(negotiationSession, model, new HashMap<String, Double>());
		} catch (Exception e) {
			e.printStackTrace();
		}
		this.possibleBids = negotiationSession.getUtilitySpace().getDomain().getNumberOfPossibleBids();
		domainIsBig = (possibleBids > 10000);
		random = new Random();
	}

	/**
	 * Selects a random bid from the best N bids, where N depends on the domain
	 * size.
	 * 
	 * @param set
	 *            of similarly preferred bids.
	 * @return nextBid to be offered
	 */
	@Override
	public BidDetails getBid(List<BidDetails> bidsInRange) {
		ArrayList<BidDetails> bidsOM = new ArrayList<BidDetails>();
		for (BidDetails bid : bidsInRange) {
			double utility;
			try {
				utility = model.getBidEvaluation(bid.getBid());
				BidDetails bidDetails = new BidDetails(bid.getBid(), utility);
				bidsOM.add(bidDetails);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		// Pick the top 3 to 20 bids, depending on the domain size
		int n = (int) Math.round(bidsOM.size() / 10.0);
		if (n < 3)
			n = 3;
		if (n > 20)
			n = 20;

		Collections.sort(bidsOM, comp);

		int entry = random.nextInt(Math.min(bidsOM.size(), n));
		Bid opponentBestBid = bidsOM.get(entry).getBid();
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
	 * Method which specifies when the opponent model may be updated. In small
	 * domains the model may be updated up till 0.99 of the time. In large
	 * domains the updating process stops half way.
	 * 
	 * @return true if the opponent model may be updated
	 */
	@Override
	public boolean canUpdateOM() {
		// in the last seconds we don't want to lose any time
		if (negotiationSession.getTime() > 0.99)
			return false;

		// in a big domain, we stop updating half-way
		if (domainIsBig) {
			if (negotiationSession.getTime() > 0.5) {
				return false;
			}
		}
		return true;
	}

	@Override
	public String getName() {
		return "Offer Best Bids";
	}
}