package agents.anac.y2014.TUDelftGroup2.boaframework.acceptanceconditions.other;

import java.util.List;
import java.util.Map;

import agents.anac.y2014.TUDelftGroup2.boaframework.offeringstrategy.other.Group2_BS;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;

/**
 * This {@code Group2_AS} is an extended AC_next strategy. It will accept any
 * bid that is better than the next that we would propose. It will also accept
 * any bid when the utility is above 0.9. With this we can avoid missing out on
 * a great deal. It will also accept any bid when there is too little time left,
 * as a sort of fail-safe, to avoid 0.0 utility outcomes.
 * 
 * @since 2013-12-13
 * @version 1.0
 */
public class Group2_AS extends AcceptanceStrategy {
	private static final int numberofroundstowait = 2000; // 4000 //26;

	/**
	 * This is the scaling factor. If we set this to, for example 2.0 than we
	 * stop when the opponent's bid is twice as good as the next bid we'll
	 * propose If we set this to 1.0, we accept any bid that is at least as good
	 * as ours.
	 */
	private double alpha;

	/**
	 * This is the offset factor. If we set this value > 0 than we accept only
	 * bids that are a constant factor better than the next bid that we would
	 * propose.
	 */
	private double beta;

	/** The reason of accepting the bid */
	private String acReason = "none";

	private int roundcounter;

	/**
	 * Empty constructor is needed for the BOA framework
	 */
	public Group2_AS() {
	}

	/**
	 * Normal constructor receives dependencies form environment/framework
	 * 
	 * @param negotiationSession
	 *            The environment of the negotiation session. Contains deadline,
	 *            domain, etc.
	 * @param offeringStrategy
	 *            This is the used bidding strategy. We use this bidding
	 *            strategy to decide what our next move will be.
	 * @param alpha
	 *            The scaling factor
	 * @param beta
	 *            The constant addition factor
	 */
	public Group2_AS(NegotiationSession negotiationSession, OfferingStrategy offeringStrategy, double alpha,
			double beta) {
		this.negotiationSession = negotiationSession;
		this.offeringStrategy = offeringStrategy;
		this.alpha = alpha;
		this.beta = beta;
	}

	/**
	 * This function gets called by the framework upon initialization. It offers
	 * all commonly used variables (which are also in the constructor). This
	 * function might be redundant. but if left it here in case it actually does
	 * something.
	 * 
	 * @param negotiationSession
	 *            The environment of the negotiation session. Contains deadline,
	 *            domain, etc.
	 * @param offeringStrategy
	 *            This is the used bidding strategy. We use this bidding
	 *            strategy to decide what our next move will be.
	 * @param parameters
	 *            This is a hashmap that contains string key "a" which
	 *            represents {@code alpha}, the scaling factor and string key
	 *            "b" which represents {@code beta}, the addition factor.
	 */
	@Override
	public void init(NegotiationSession negotiationSession, OfferingStrategy offeringStrategy,
			OpponentModel opponentModel, Map<String, Double> parameters) throws Exception {
		// initialize commonly used variables
		this.negotiationSession = negotiationSession;
		this.offeringStrategy = offeringStrategy;
		this.opponentModel = opponentModel;
		roundcounter = 0;

		// if alpha or beta are set, use those values, otherwise use there
		// defaults
		alpha = parameters.get("a") != null ? parameters.get("a") : 1;
		beta = parameters.get("b") != null ? parameters.get("b") : 0;
	}

	/**
	 * Human readable string representation for {@code alpha} and {@code beta}.
	 * This is used for debugging by Hendrikx, do we have any use for it?
	 * 
	 * @return The {@code alpha} and {@code beta} factors.
	 */
	@Override
	public String printParameters() {
		return ""; // "[a: " + alpha + " b: " + beta + "]";
	}

	/**
	 * Calculates the acceptability of the latest bid by the opponent.
	 * 
	 * @return Either {@code Actions.Accept} if we can accept the bid or
	 *         {@code Action.Reject} if we should reject the bid.
	 */
	@Override
	public Actions determineAcceptability() {
		// use a counter, and basically accept after 26 bids.

		if (Group2_BS.disp_exploration) {

			roundcounter++;
			if (roundcounter > numberofroundstowait) {
				return Actions.Accept;
			} else {
				return Actions.Reject;
			}

		} else {
			String AS = "ACNEXT";
			// String AS="ACGOODENOUGH";
			// String AS="ACNEVER";

			if (AS == "ACNEVER") {
				return Actions.Reject;
			}

			if (AS == "ACNEXT") {
				// Use AC Next
				// Calculate ACnext
				// LINEAR CODE
				// used for getting timing and bid util
				Group2_BS metaStrategy = (Group2_BS) offeringStrategy;

				// find out the utilities of yourself and the opponent
				double nextMyBidUtil = metaStrategy.determineNextBid().getMyUndiscountedUtil();
				double lastOpponentBidUtil = negotiationSession.getOpponentBidHistory().getLastBidDetails()
						.getMyUndiscountedUtil();

				boolean accept = alpha * lastOpponentBidUtil + beta >= nextMyBidUtil;
				if (accept)
					acReason = "ACnext (Your next offer would be smaller)";

				return accept ? Actions.Accept : Actions.Reject;
			}

			if (AS == "ACGOODENOUGH") {
				// Use AC Good Enough
				// A more genreal AC Next: if the new bid of the opponent is
				// better then any of the ones that we have bid ourselves
				// accept it.

				// LINEAR CODE
				// used for getting timing and bid util
				Group2_BS metaStrategy = (Group2_BS) offeringStrategy;

				// find out the utilities of yourself and the opponent
				double nextMyBidUtil = metaStrategy.determineNextBid().getMyUndiscountedUtil();
				double lastOpponentBidUtil = negotiationSession.getOpponentBidHistory().getLastBidDetails()
						.getMyUndiscountedUtil();

				// Look through history
				// the opp bid is better than any of them, accept it
				List<BidDetails> history = negotiationSession.getOwnBidHistory().getHistory();
				boolean accept = false;

				for (BidDetails bidDetails : history) {
					if (lastOpponentBidUtil >= bidDetails.getMyUndiscountedUtil())
						accept = true;
				}

				// Try Ac_next if still false
				if (!accept)
					accept = alpha * lastOpponentBidUtil + beta >= nextMyBidUtil;
				if (accept)
					acReason = "AC good enough";

				return accept ? Actions.Accept : Actions.Reject;

			}

			// by default accept any bid
			return Actions.Accept;

		}

		// LINEAR CODE
		// // used for getting timing and bid util
		// BidSpaceExtractor metaStrategy = (BidSpaceExtractor)offeringStrategy;
		//
		// // find out the utilities of yourself and the opponent
		// double nextMyBidUtil =
		// metaStrategy.determineNextBid().getMyUndiscountedUtil();
		// double lastOpponentBidUtil =
		// negotiationSession.getOpponentBidHistory()
		// .getLastBidDetails()
		// .getMyUndiscountedUtil();
		//
		// // MatLab link stuff
		// if (matlabLink.active())
		// {
		// // Save a text version of the opponent bid History
		// matlabLink.eval_commandline("opp_bid_hist_bids=cell("+negotiationSession.getOpponentBidHistory().size()+",1);");
		// for (int j = 0; j <
		// negotiationSession.getOpponentBidHistory().size(); j++)
		// matlabLink.eval_commandline("opp_bid_hist_bids{"+(j+1)+"}='"+negotiationSession.getOpponentBidHistory().getHistory().get(j).getBid().toString()+"';");
		// }
		//
		// // Calculate ACnext
		// boolean accept = alpha * lastOpponentBidUtil + beta >= nextMyBidUtil;
		// if (accept) acReason="ACnext (Your next offer would be smaller)";
		//
		// // Calculate ACtime (only after 10 bids)
		// if (metaStrategy.timing_numbids>10)
		// {
		// // Accept if T > 1 - "average time of last 10 bids" + "standard
		// deviation of timall bids"
		// boolean ACtime =
		// (negotiationSession.getTimeline().getTotalTime()-negotiationSession.getTimeline().getCurrentTime())<metaStrategy.timing_10avg+metaStrategy.timing_sd*1D;
		// accept = accept || ACtime;
		// if (ACtime) acReason="ACtime (Failsafe because time ran out)";
		// }
		//
		// // Calculate ACthreshold
		// boolean ACthreshold = lastOpponentBidUtil>0.9;
		// accept = accept || ACthreshold;
		// if (ACthreshold) acReason="ACthreshold (Util above 0.9)";
		//
		// // return Actions accept if the inequality holds, reject otherwise.
		// // if (accept) System.out.println("Accepted, reason: "+acReason);
		// return accept ? Actions.Accept : Actions.Reject;
	}

	@Override
	public String getName() {
		return "2014 - Group2";
	}
}