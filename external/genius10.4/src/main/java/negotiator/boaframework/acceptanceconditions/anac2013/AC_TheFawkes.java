package negotiator.boaframework.acceptanceconditions.anac2013;

import java.util.List;
import java.util.Map;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import negotiator.boaframework.opponentmodel.TheFawkes_OM;

/**
 * Acceptance Strategy
 *
 * This is ACcombi = ACnext || ACtime(T) & ACconst(MAXw)
 *
 * For this implementation I used the following article: "Acceptance Conditions
 * in Automated Negotion - Tim Baarslag, Koen Hindriks and Catholijn Jonker"
 *
 * We accept when our current bid is worse than the last bid we got from the
 * opponent or when we get close to the time limit and the utility we got from
 * opponent's last bid has a maximum value in a window.
 */
public final class AC_TheFawkes extends AcceptanceStrategy {
	private TheFawkes_OM OM;
	private double minimumAcceptable;

	@Override
	public void init(NegotiationSession nSession, OfferingStrategy biddingStrategy, OpponentModel oppModel,
			Map<String, Double> params) throws Exception {
		super.init(nSession, biddingStrategy, oppModel, params);
		this.OM = (TheFawkes_OM) oppModel;

		List<BidDetails> allBids = nSession.getOutcomeSpace().getAllOutcomes();
		double total = 0;
		for (BidDetails bid : allBids) {
			total += bid.getMyUndiscountedUtil();
		}
		this.minimumAcceptable = total / (double) allBids.size(); // TODO: use
																	// median
																	// instead
																	// of
																	// average
																	// perhaps?!
	}

	@Override
	public Actions determineAcceptability() {
		BidDetails lastOpponentBid = this.negotiationSession.getOpponentBidHistory().getLastBidDetails();
		double lastOpponentBidUtility = lastOpponentBid.getMyUndiscountedUtil();
		BidDetails myNextBid = this.offeringStrategy.determineNextBid();
		double myNextBidUtility = myNextBid.getMyUndiscountedUtil();
		if (lastOpponentBidUtility < this.minimumAcceptable) {
			// Group3_Agent.debug( "REJECT (ACavg) (" + lastOpponentBidUtility +
			// "<" + this.minimumAcceptable + ")" );
			return Actions.Reject;
		} else if (lastOpponentBidUtility >= myNextBidUtility) {
			// Group3_Agent.debug( "ACCEPT (ACnext) (" + lastOpponentBidUtility
			// + ">=" + myNextBidUtility + ")" );
			return Actions.Accept;
		} else if (this.negotiationSession.getTime() >= (1 - this.OM.getMaxOpponentBidTimeDiff())) {
			double time = this.negotiationSession.getTime();
			BidDetails bestOpponentBid = this.negotiationSession.getOpponentBidHistory()
					.filterBetweenTime(time - (this.OM.getMaxOpponentBidTimeDiff() * 10), time).getBestBidDetails();
			double bestOpponentBidUtility = bestOpponentBid.getMyUndiscountedUtil();
			if (lastOpponentBidUtility >= bestOpponentBidUtility) {
				// Group3_Agent.debug( "ACCEPT (ACtime&&ACconst) (" +
				// this.negotiationSession.getTime() + ">=" + ( 1 -
				// this.OM.getMaxOpponentBidTimeDiff() )
				// + "&&" + lastOpponentBidUtility + ">=" +
				// bestOpponentBidUtility + ")" );
				return Actions.Accept;
			} else {
				// Group3_Agent.debug( "REJECT (ACtime&&ACconst) (" +
				// this.negotiationSession.getTime() + ">=" + ( 1 -
				// this.OM.getMaxOpponentBidTimeDiff() )
				// + "&& NOT(" + lastOpponentBidUtility + ">=" +
				// bestOpponentBidUtility + "))" );
				return Actions.Reject;
			}
		} else {
			// Group3_Agent.debug( "REJECT (ACnext) (NOT(" +
			// lastOpponentBidUtility + ">=" + myNextBidUtility + "))" );
			return Actions.Reject;
		}
	}

	@Override
	public String getName() {
		return "2013 - TheFawkes";
	}
}
