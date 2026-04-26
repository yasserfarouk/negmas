package negotiator.boaframework.offeringstrategy.anac2012.TheNegotiatorReloaded;

import genius.core.BidHistory;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;

/**
 * An analyzer which returns the type of bidding strategy employed by the opponent.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class BSPredictor {
	
	// the negotiation environment
	private NegotiationSession negoSession;
	// the amount of windows
	private double numberOfWindows;
	// the time when the window began
	private double beginWindowTime = 0;
	// the maximum utility found in the previous window
	private BidDetails prevBidUtilInWindow;
	/**the expected total concession of the opponent**/
	private final double TOTAL_EXPECTED_CONCESSION = 1.0;
	
	/**
	 * Initializes the amount of windows and sets the negotiation session.
	 * 
	 * @param negoSession the negotiation environment
	 * @param numberOfWindows the total amount of window
	 */
	public BSPredictor(NegotiationSession negoSession, int numberOfWindows) {
		this.negoSession = negoSession;
		this.numberOfWindows = numberOfWindows;
	}
	
	/**
	 * Predict the strategy of the opponent.
	 * 
	 * @return the strategy of the opponent
	 */
	public StrategyTypes calculateOpponentStrategy() {
		StrategyTypes opponentStrategy = null;
		double endWindowTime = negoSession.getOpponentBidHistory().getLastBidDetails().getTime();
		BidHistory bidsInWindow = negoSession.getOpponentBidHistory().filterBetweenTime(beginWindowTime, endWindowTime);
		BidDetails bestBidUtilInWindow = bidsInWindow.getBestBidDetails();
		
		// occurs on large domains such as Energy, where it takes a long time to initialize the agent
		if (bestBidUtilInWindow == null) {
			return StrategyTypes.Hardliner;
		}
		
		double concession;
		if (prevBidUtilInWindow == null) {
			concession = 1.0 - bestBidUtilInWindow.getMyUndiscountedUtil();
		} else {
			concession = bestBidUtilInWindow.getMyUndiscountedUtil() - prevBidUtilInWindow.getMyUndiscountedUtil();
		}

		double concessionThreshold = (TOTAL_EXPECTED_CONCESSION / numberOfWindows / 2.0);

		if (concession <= concessionThreshold) {
			opponentStrategy = StrategyTypes.Hardliner;
		} else {
			opponentStrategy = StrategyTypes.Conceder;
		}
		prevBidUtilInWindow = bestBidUtilInWindow;
		beginWindowTime = endWindowTime;
		
		return opponentStrategy;
	}
}