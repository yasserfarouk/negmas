package agents.anac.y2011.TheNegotiator;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.actions.Action;
import genius.core.actions.Offer;

/**
 * The Decider class is used to decide each turn which action the agent should
 * perform.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx, Julian de Ruiter
 */
public class Decider {

	// storageobject which contains all possible bids, bids already used, and
	// opponent bids.
	private BidsCollection bidsCollection;
	// reference to the timemanager (manages time related function)
	private TimeManager timeManager;
	// reference to the bidsgenerator (generates a (counter)offer)
	private BidGenerator bidGenerator;
	// reference to the acceptor (generates an action to accept)
	private Acceptor acceptor;
	// reference to the negotiation agent
	private Agent agent;
	private Bid lastOppbid;

	/**
	 * Creates a Decider-object which determines which offers should be made
	 * during the negotiation.
	 * 
	 * @param agent
	 *            in negotiation
	 */
	public Decider(Agent agent) {
		this.agent = agent;
		bidsCollection = new BidsCollection();
		bidGenerator = new BidGenerator(agent, bidsCollection);
		acceptor = new Acceptor(agent.utilitySpace, bidsCollection,
				agent.getAgentID());
		timeManager = new TimeManager(agent.timeline,
				agent.utilitySpace.getDiscountFactor(), bidsCollection);
	}

	/**
	 * Stores the bids of the partner in the history with the corresponding
	 * utility.
	 * 
	 * @param action
	 *            action made by partner
	 */
	public void setPartnerMove(Action action) {
		if (action instanceof Offer) {
			lastOppbid = ((Offer) action).getBid();
			try {
				bidsCollection.addPartnerBid(lastOppbid,
						agent.utilitySpace.getUtility(lastOppbid),
						timeManager.getTime());
			} catch (Exception e) {

			}
		}
	}

	/**
	 * Method which returns the action to be performed by the agent. Assumes
	 * setPartnerMove has been called with last opponent move.
	 */
	public Action makeDecision() {

		int phase = timeManager.getPhase(timeManager.getTime());
		int movesLeft = 0;

		if (phase == 3) {
			movesLeft = timeManager.getMovesLeft();
		}
		// if the negotiation is still going on and a bid has already been made
		double threshold = timeManager.getThreshold(timeManager.getTime());
		Action action = acceptor.determineAccept(phase, threshold,
				agent.timeline.getTime(), movesLeft, lastOppbid);

		// if we didn't accept, generate an offer (end of negotiation is never
		// played
		if (action == null) {
			action = bidGenerator.determineOffer(agent.getAgentID(), phase,
					timeManager.getThreshold(timeManager.getTime()));
		}
		return action;
	}
}