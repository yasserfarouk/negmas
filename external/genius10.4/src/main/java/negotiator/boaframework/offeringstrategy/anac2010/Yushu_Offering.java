package negotiator.boaframework.offeringstrategy.anac2010;

import java.util.Collections;
import java.util.LinkedList;
import java.util.Map;
import java.util.Random;

import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.bidding.BidDetails;
import genius.core.bidding.BidDetailsStrictSorterUtility;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import negotiator.boaframework.opponentmodel.DefaultModel;
import negotiator.boaframework.sharedagentstate.anac2010.YushuSAS;

/**
 * This is the decoupled Offering Strategy for Nozomi (ANAC2010). The code was
 * taken from the ANAC2010 Yushu and adapted to work within the BOA framework.
 * 
 * DEFAULT OM: None
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class Yushu_Offering extends OfferingStrategy {

	BidDetails suggestBid = null; // the bid suggested based on op's bids
	BidDetails maxUtilBid = null;
	private double highPosUtil; // the highest utility that can be achieved
	Random random200;
	private final boolean TEST_EQUIVALENCE = false;

	/**
	 * Empty constructor called by BOA framework.
	 */
	public Yushu_Offering() {
	}

	@Override
	public void init(NegotiationSession domainKnow, OpponentModel model, OMStrategy omStrategy,
			Map<String, Double> parameters) throws Exception {
		if (model instanceof DefaultModel) {
			model = new NoModel();
		}
		super.init(domainKnow, model, omStrategy, parameters);

		maxUtilBid = negotiationSession.getMaxBidinDomain();
		helper = new YushuSAS(negotiationSession);
		highPosUtil = negotiationSession.getMaxBidinDomain().getMyUndiscountedUtil();
		if (TEST_EQUIVALENCE) {
			random200 = new Random(200);
		} else {
			random200 = new Random();
		}
	}

	@Override
	public BidDetails determineOpeningBid() {
		if (negotiationSession.getOpponentBidHistory().size() > 0) {
			((YushuSAS) helper).updateBelief(negotiationSession.getOpponentBidHistory().getLastBidDetails());
		}
		((YushuSAS) helper).setPreviousTime(negotiationSession.getTime());
		nextBid = maxUtilBid;
		return nextBid;
	}

	@Override
	public BidDetails determineNextBid() {
		((YushuSAS) helper).updateBelief(negotiationSession.getOpponentBidHistory().getLastBidDetails());

		BidDetails myLastBid = negotiationSession.getOwnBidHistory().getLastBidDetails();

		double targetUtility;
		if (myLastBid == null)
			targetUtility = 1;
		else {
			targetUtility = ((YushuSAS) helper).calculateTargetUtility();
		}
		suggestBid = ((YushuSAS) helper).getSuggestBid();
		if (this.suggestBid != null) {
			nextBid = this.suggestBid;
		} else if (targetUtility >= highPosUtil) {
			nextBid = maxUtilBid;

		} else {
			try {
				nextBid = getNextBid(targetUtility);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return nextBid;
	}

	private BidDetails getNextBid(double targetuti) throws Exception {
		BidDetails potentialBid = null;
		double maxdiff = Double.MAX_VALUE;
		double tempmaxdiff = Double.MAX_VALUE;
		BidDetails tempnextBid = null;

		LinkedList<BidDetails> candidates = new LinkedList<BidDetails>();

		BidIterator bidsIter = new BidIterator(negotiationSession.getUtilitySpace().getDomain());
		while (bidsIter.hasNext()) {
			Bid bid = bidsIter.next();
			BidDetails tmpBid = new BidDetails(bid, negotiationSession.getUtilitySpace().getUtility(bid),
					negotiationSession.getTime());

			double vlowbound;
			if (((YushuSAS) helper).getRoundLeft() > 30)
				vlowbound = Math.max(((YushuSAS) helper).getBestTenBids().get(0).getMyUndiscountedUtil(), targetuti);
			else
				vlowbound = 0.96 * targetuti;
			if ((tmpBid.getMyUndiscountedUtil() > vlowbound) && (tmpBid.getMyUndiscountedUtil() < 1.08 * targetuti))
				candidates.add(tmpBid);

			double currentdiff = Math.abs(tmpBid.getMyUndiscountedUtil() - targetuti);
			if (currentdiff < tempmaxdiff) {
				tempmaxdiff = currentdiff;
				tempnextBid = tmpBid;
			}

			if ((currentdiff < maxdiff) & (tmpBid.getMyUndiscountedUtil() > targetuti)) {
				maxdiff = currentdiff;
				potentialBid = tmpBid;
			}
		}
		if (negotiationSession.getOwnBidHistory().size() > 10) {
			candidates.add(potentialBid);
			if (TEST_EQUIVALENCE) {
				Collections.sort(candidates, new BidDetailsStrictSorterUtility());
			}

			if (opponentModel instanceof NoModel) {
				int indexc = (int) (random200.nextDouble() * candidates.size());
				potentialBid = candidates.get(indexc);
			} else {
				potentialBid = omStrategy.getBid(candidates);
			}
		}
		if (potentialBid == null) {
			potentialBid = tempnextBid;
		}
		return potentialBid;
	}

	@Override
	public String getName() {
		return "2010 - Yushu";
	}
}