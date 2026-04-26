package agents.anac.y2014.TUDelftGroup2.boaframework.offeringstrategy.other;

import genius.core.Bid;
import genius.core.analysis.BidSpace;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OpponentModel;

/**
 * Abstract class for our opening/midgame/endgame strategies.
 */
public abstract class AbstractBiddingStrategy {

	NegotiationSession negotiationSession;
	BidSpace bidSpace;
	OpponentModel opponentModel;
	
	/**
	 * Abstract class constructor
	 * We need the current session and oppent model to calculate our actual strategy
	 * @param negotiationSession The {@link NegotiationSession} we are currently running
	 * @param opponentModel The {@link OpponentModel} this BOA agent is now using.
	 */
	AbstractBiddingStrategy(NegotiationSession negotiationSession,OpponentModel opponentModel)
	{
		this.opponentModel = opponentModel;
		this.negotiationSession = negotiationSession;
		refreshBidSpace();
	}

	/**
	 * Updates the {@link BidSpace} by getting a new one from the negotiation session.
	 * The {@link BidSpace} might change and is not updated accordingly if the {@link OpponentModel} changes.
	 */
	void refreshBidSpace()
	{
		// DONT DO THIS IS NONLINEAR
//		try {
//			this.bidSpace = new BidSpace(negotiationSession.getUtilitySpace(), opponentModel.getOpponentUtilitySpace(),false);
//		} catch (Exception e) {
//			e.printStackTrace();
//		}
	}
	
	/**
	 * Calculate the bid that is considered best by this strategy
	 * @return Best bid according to current strategy
	 */
	abstract Bid getBid();
}
