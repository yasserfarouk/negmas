package agents.anac.y2011.TheNegotiator;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.utility.AbstractUtilitySpace;

/**
 * The Acceptor class is used to decide when to accept a bid.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx, Julian de Ruiter
 */
public class Acceptor {

	// utilityspace of the negotiation
	private AbstractUtilitySpace utilitySpace;
	// reference to the bidscollection
	private BidsCollection bidsCollection;
	private AgentID agentID;

	/**
	 * Creates an Acceptor-object which determines which offers should be
	 * accepted during the negotiation.
	 * 
	 * @param utilitySpace
	 * @param bidsCollection
	 *            of all possible bids (for us) and the partner bids
	 */
	public Acceptor(AbstractUtilitySpace utilitySpace,
			BidsCollection bidsCollection, AgentID agent) {
		this.agentID = agent;
		this.utilitySpace = utilitySpace;
		this.bidsCollection = bidsCollection;
	}

	/**
	 * Determine if it is wise to accept for a given phase on a given time.
	 * 
	 * @param phase
	 *            of the negotiation
	 * @param minimum
	 *            threshold
	 * @param time
	 *            in negotiation
	 * @param movesLeft
	 *            is the estimated moves left
	 * @param lastOpponentBid
	 *            the last opponent bid
	 * @return move to (not) accept
	 */
	public Action determineAccept(int phase, double threshold, double time,
			int movesLeft, Bid lastOpponentBid) {
		Action action = null;
		double utility = 0;

		try {
			// effectively this will ensure that the utility is 0 if our agent
			// is first
			if (bidsCollection.getPartnerBids().size() > 0) {
				// get the last opponent bid
				utility = utilitySpace.getUtility(bidsCollection
						.getPartnerBid(0));
			}
		} catch (Exception e) {
		}

		if (phase == 1 || phase == 2) {
			if (utility >= threshold) {
				action = new Accept(agentID, lastOpponentBid);
			}
		} else { // phase 3
			if (movesLeft >= 15) {
				if (utility >= threshold) {
					action = new Accept(agentID, lastOpponentBid);
				}
			} else {
				if (movesLeft < 15) {
					action = new Accept(agentID, lastOpponentBid);
				}
			}
		}
		return action;
	}
}