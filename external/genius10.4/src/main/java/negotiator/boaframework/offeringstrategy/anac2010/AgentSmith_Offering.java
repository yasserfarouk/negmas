package negotiator.boaframework.offeringstrategy.anac2010;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.utility.AbstractUtilitySpace;
import negotiator.boaframework.opponentmodel.DefaultModel;
import negotiator.boaframework.opponentmodel.SmithFrequencyModel;
import negotiator.boaframework.opponentmodel.agentsmith.Bounds;

/**
 * This is the decoupled Offering Strategy for AgentSmith (ANAC2010). The code
 * was taken from the ANAC2010 AgentSmith and adapted to work within the BOA
 * framework.
 * 
 * Using another opponent model strategy makes no sense for this agent.
 * 
 * DEFAULT OM: SmithFrequencyModel
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class AgentSmith_Offering extends OfferingStrategy {

	private final static double sTimeMargin = 170.0 / 180.0;
	private final static double sUtilyMargin = 0.7;
	static private double UTILITY_THRESHOLD = 0.7;

	private int fIndex;

	/**
	 * Empty constructor called by BOA framework.
	 */
	public AgentSmith_Offering() {
	}

	public AgentSmith_Offering(NegotiationSession negoSession, OpponentModel om, OMStrategy oms) {
		initializeAgent(negoSession, om, oms);
	}

	@Override
	public void init(NegotiationSession negotiationSession, OpponentModel model, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {
		if (model instanceof DefaultModel) {
			model = new SmithFrequencyModel();
			model.init(negotiationSession, null);
			oms.setOpponentModel(model);
		}
		initializeAgent(negotiationSession, model, oms);
	}

	public void initializeAgent(NegotiationSession negoSession, OpponentModel om, OMStrategy oms) {
		this.negotiationSession = negoSession;
		opponentModel = om;
		omStrategy = oms;
		fIndex = 0;
	}

	@Override
	public BidDetails determineNextBid() {
		// Time in seconds.
		double time = negotiationSession.getTime();
		Bid bid2Offer = null;
		try {
			// Check if the session (2 min) is almost finished
			if (time >= sTimeMargin) {
				// If the session is almost finished check if the utility is
				// "high enough"
				BidDetails lastBid = negotiationSession.getOpponentBidHistory().getLastBidDetails();
				if (lastBid.getMyUndiscountedUtil() < sUtilyMargin) {
					nextBid = negotiationSession.getOpponentBidHistory().getBestBidDetails();
				}
			} else {
				bid2Offer = getMostOptimalBid();
				nextBid = new BidDetails(bid2Offer, negotiationSession.getUtilitySpace().getUtility(bid2Offer),
						negotiationSession.getTime());
			}
		} catch (Exception e) {

		}
		return nextBid;
	}

	@Override
	public BidDetails determineOpeningBid() {
		return negotiationSession.getMaxBidinDomain();
	}

	/**
	 * Calculate the most optimal bid
	 * 
	 * @return the most optimal bid
	 * @throws Exception
	 */
	public Bid getMostOptimalBid() {
		ArrayList<Bid> allBids = getSampledBidList();

		ArrayList<Bid> removeMe = new ArrayList<Bid>();
		for (int i = 0; i < allBids.size(); i++) {
			try {
				if (negotiationSession.getUtilitySpace().getUtility(allBids.get(i)) < UTILITY_THRESHOLD) {
					removeMe.add(allBids.get(i));
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		allBids.removeAll(removeMe);

		if (opponentModel instanceof NoModel) {
			Bid bid = allBids.get(fIndex);
			fIndex++;
			return bid;
		} else {
			// Log.logger.info("Size of bid space: " + lBids.size());
			Comparator<Bid> lComparator = new BidComparator(negotiationSession.getUtilitySpace());

			// sort the bids in order of highest utility
			ArrayList<Bid> sortedAllBids = allBids;
			Collections.sort(sortedAllBids, lComparator);

			Bid lBid = sortedAllBids.get(fIndex);
			if (fIndex < sortedAllBids.size() - 1)
				fIndex++;
			return lBid;
		}
	}

	private ArrayList<Bid> getSampledBidList() {
		ArrayList<Bid> lBids = new ArrayList<Bid>();
		List<Issue> lIssues = negotiationSession.getIssues();
		HashMap<Integer, Bounds> lBounds = Bounds.getIssueBounds(lIssues);

		// first createFrom a new list
		HashMap<Integer, Value> lBidValues = new HashMap<Integer, Value>();
		for (Issue lIssue : lIssues) {
			Bounds b = lBounds.get(lIssue.getNumber());
			Value v = Bounds.getIssueValue(lIssue, b.getLower());
			lBidValues.put(lIssue.getNumber(), v);
		}
		try {
			lBids.add(new Bid(negotiationSession.getUtilitySpace().getDomain(), lBidValues));
		} catch (Exception e) {
		}

		// for each item permutate with issue values, like binary
		// 0 0 0
		// 0 0 1
		// 0 1 0
		// 0 1 1
		// etc.
		for (Issue lIssue : lIssues) {
			ArrayList<Bid> lTempBids = new ArrayList<Bid>();
			Bounds b = lBounds.get(lIssue.getNumber());

			for (Bid lTBid : lBids) {
				for (double i = b.getLower(); i < b.getUpper(); i += b.getStepSize()) {
					HashMap<Integer, Value> lNewBidValues = getBidValues(lTBid);
					lNewBidValues.put(lIssue.getNumber(), Bounds.getIssueValue(lIssue, i));

					try {
						Bid iBid = new Bid(negotiationSession.getUtilitySpace().getDomain(), lNewBidValues);
						lTempBids.add(iBid);

					} catch (Exception e) {

					}
				}
			}
			lBids = lTempBids;
		}

		ArrayList<Bid> lToDestroy = new ArrayList<Bid>();
		for (Bid lBid : lBids) {
			try {
				if (negotiationSession.getUtilitySpace().getUtility(lBid) < UTILITY_THRESHOLD) {
					lToDestroy.add(lBid);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		for (Bid lBid : lToDestroy) {
			lBids.remove(lBid);
		}

		return lBids;
	}

	/**
	 * Get the values of a bid
	 * 
	 * @param pBid
	 * @return
	 */
	private HashMap<Integer, Value> getBidValues(Bid pBid) {
		HashMap<Integer, Value> lNewBidValues = new HashMap<Integer, Value>();
		for (Issue lIssue : negotiationSession.getUtilitySpace().getDomain().getIssues()) {
			try {
				lNewBidValues.put(lIssue.getNumber(), pBid.getValue(lIssue.getNumber()));
			} catch (Exception e) {

			}
		}
		return lNewBidValues;
	}

	public class BidComparator implements Comparator<Bid> {

		private AbstractUtilitySpace space;

		public BidComparator(AbstractUtilitySpace mySpace) {
			this.space = mySpace;
		}

		/*
		 * returns 1 if his own bid is better than the opponents, -1 otherwise
		 */
		public int compare(Bid b1, Bid b2) {
			return getMeasure(b2) > getMeasure(b1) ? -1 : 1;
		}

		/*
		 * returns a double that represents the value of a value of a bid,
		 * taking into account both the agents own and opponents' utility.
		 */
		public double getMeasure(Bid b1) {
			double a = 0;
			try {
				a = (1 - space.getUtility(b1));
			} catch (Exception e) {
				e.printStackTrace();
			}
			double b = (1 - opponentModel.getBidEvaluation(b1));

			double alpha = Math.atan(b / a);

			return a + b + (0.5 * Math.PI / alpha) * 0.5 * Math.PI;
		}

	}

	@Override
	public String getName() {
		return "2010 - AgentSmith";
	}
}