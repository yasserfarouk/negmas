package agents.anac.y2010.Southampton;

import java.util.ArrayList;
import java.util.Random;

import agents.anac.y2010.Southampton.analysis.BidSpace;
import agents.anac.y2010.Southampton.utils.ActionCreator;
import agents.anac.y2010.Southampton.utils.OpponentModel;
import genius.core.Agent;
import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * @author Colin Williams
 * 
 */
public abstract class SouthamptonAgent extends Agent {

	private final boolean TEST_EQUIVALENCE = false;

	private static enum ActionType {
		ACCEPT, BREAKOFF, OFFER, START;
	}

	// protected static double CONCESSIONFACTOR = 0.04;

	/**
	 * Our maximum aspiration level.
	 */
	protected static double MAXIMUM_ASPIRATION = 0.9;

	/**
	 * Gets the version number.
	 * 
	 * @return the version number.
	 */
	@Override
	public String getVersion() {
		return "2.0 (Genius 3.1)";
	}

	/**
	 * Our BidSpace.
	 */
	protected BidSpace bidSpace;

	/**
	 * The message received from the opponent.
	 */
	private Action messageOpponent;

	/**
	 * My previous action.
	 */
	protected Action myLastAction = null;

	/**
	 * My previous bid.
	 */
	protected Bid myLastBid = null;

	/**
	 * A list of our previous bids.
	 */
	protected ArrayList<Bid> myPreviousBids;

	/**
	 * The bids made by the opponent.
	 */
	protected ArrayList<Bid> opponentBids;

	/**
	 * Our model of the opponent.
	 */
	protected OpponentModel opponentModel;

	/**
	 * The opponent's previous bid.
	 */
	protected Bid opponentPreviousBid = null;

	protected final double acceptMultiplier = 1.02;

	protected boolean opponentIsHardHead;

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#finalize()
	 */
	@Override
	protected void finalize() throws Throwable {
		// displayFrame.dispose();
		super.finalize();
	}

	/**
	 * Get the action type of a given action.
	 * 
	 * @param action
	 *            The action.
	 * @return The action type of the action.
	 */
	private ActionType getActionType(Action action) {
		ActionType actionType = ActionType.START;
		if (action instanceof Offer)
			actionType = ActionType.OFFER;
		else if (action instanceof Accept)
			actionType = ActionType.ACCEPT;
		else if (action instanceof EndNegotiation)
			actionType = ActionType.BREAKOFF;
		return actionType;
	}

	public final Action chooseAction() {

		Action chosenAction = null;
		Bid opponentBid = null;

		try {
			switch (getActionType(this.messageOpponent)) {
			case OFFER:
				opponentBid = ((Offer) this.messageOpponent).getBid();
				chosenAction = handleOffer(opponentBid);
				break;
			case ACCEPT:
			case BREAKOFF:
				break;
			default:
				if (this.myLastAction == null) {

					chosenAction = new Offer(getAgentID(), proposeInitialBid());

				} else {

					chosenAction = this.myLastAction;
				}
				break;
			}

		} catch (Exception e) {
			e.printStackTrace();
			chosenAction = ActionCreator.createOffer(this, myLastBid);
		}
		myLastAction = chosenAction;
		if (myLastAction instanceof Offer) {
			Bid b = ((Offer) myLastAction).getBid();
			myPreviousBids.add(b);
			myLastBid = b;
		}
		return chosenAction;
	}

	/**
	 * Get the number of the agent.
	 * 
	 * @return the number of the agent.
	 */
	public int getAgentNo() {
		if (this.getName() == "Agent A")
			return 1;
		if (this.getName() == "Agent B")
			return 2;
		return 0;
	}

	/**
	 * Get all of the bids in a utility range.
	 * 
	 * @param lowerBound
	 *            The minimum utility level of the bids.
	 * @param upperBound
	 *            The maximum utility level of the bids.
	 * @return all of the bids in a utility range.
	 * @throws Exception
	 */
	private ArrayList<Bid> getBidsInRange(double lowerBound, double upperBound)
			throws Exception {
		ArrayList<Bid> bidsInRange = new ArrayList<Bid>();
		BidIterator iter = new BidIterator(utilitySpace.getDomain());
		while (iter.hasNext()) {
			Bid tmpBid = iter.next();
			double util = 0;
			try {
				util = utilitySpace.getUtility(tmpBid);
				if (util >= lowerBound && util <= upperBound)
					bidsInRange.add(tmpBid);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		return bidsInRange;
	}

	/**
	 * Get the concession factor.
	 * 
	 * @return the concession factor.
	 */
	/*
	 * private double getConcessionFactor() { // The more the agent is willing
	 * to concede on its aspiration value, the // higher this factor. return
	 * CONCESSIONFACTOR; }
	 */

	/**
	 * Get a random bid in a given utility range.
	 * 
	 * @param lowerBound
	 *            The lower bound on utility.
	 * @param upperBound
	 *            The upper bound on utility.
	 * @return a random bid in a given utility range.
	 * @throws Exception
	 */
	protected Bid getRandomBidInRange(double lowerBound, double upperBound)
			throws Exception {
		ArrayList<Bid> bidsInRange = getBidsInRange(lowerBound, upperBound);

		int index = (new Random()).nextInt(bidsInRange.size() - 1);

		return bidsInRange.get(index);
	}

	/**
	 * Get the target utility.
	 * 
	 * @param myUtility
	 *            This agent's utility.
	 * @param opponentUtility
	 *            The opponent's utility.
	 * @return the target utility.
	 */
	// protected abstract double getTargetUtility(double myUtility, double
	// opponentUtility);
	/*
	 * { return myUtility - getConcessionFactor(); }
	 */

	/**
	 * Handle an opponent's offer.
	 * 
	 * @param opponentBid
	 *            The bid made by the opponent.
	 * @return the action that we should take in response to the opponent's
	 *         offer.
	 * @throws Exception
	 */
	@SuppressWarnings("unused")
	private Action handleOffer(Bid opponentBid) throws Exception {
		Action chosenAction = null;

		if (myLastAction == null) {
			// Special case to handle first action

			Bid b = proposeInitialBid();
			if (opponentBid != null && TEST_EQUIVALENCE) {
				double currentTime = timeline.getTime()
						* timeline.getTotalTime() * 1000;
				double totalTime = timeline.getTotalTime() * 1000;
				opponentModel.updateBeliefs(opponentBid,
						Math.round(currentTime), totalTime);
			}
			myLastBid = b;
			chosenAction = new Offer(this.getAgentID(), b);
		} else if (utilitySpace.getUtility(opponentBid) * acceptMultiplier >= utilitySpace
				.getUtility(myLastBid)) {
			// Accept opponent's bid based on my previous bid.
			chosenAction = ActionCreator.createAccept(this, opponentBid);
			log("Opponent's bid is good enough compared to my last bid, ACCEPTED.");
			opponentBids.add(opponentBid);
			opponentPreviousBid = opponentBid;
		} else if (utilitySpace.getUtility(opponentBid) * acceptMultiplier >= MAXIMUM_ASPIRATION) {
			// Accept opponent's bid based on my previous bid.
			chosenAction = ActionCreator.createAccept(this, opponentBid);
			log("Utility of opponent bid: "
					+ utilitySpace.getUtility(opponentBid));
			log("acceptMultiplier: " + acceptMultiplier);
			log("MAXIMUM_ASPIRATION: " + MAXIMUM_ASPIRATION);
			log("Opponent's bid is good enough compared to my maximum aspiration, ACCEPTED.");
			opponentBids.add(opponentBid);
			opponentPreviousBid = opponentBid;
		} else {
			Bid plannedBid = proposeNextBid(opponentBid);

			chosenAction = ActionCreator.createOffer(this, plannedBid);

			if (plannedBid == null) {
				chosenAction = ActionCreator.createAccept(this, opponentBid);
			} else {

				if (utilitySpace.getUtility(opponentBid) * acceptMultiplier >= utilitySpace
						.getUtility(plannedBid)) {
					// Accept opponent's bid based on my planned bid.
					chosenAction = ActionCreator
							.createAccept(this, opponentBid);
					log("Opponent's bid is good enough compared to my planned bid, ACCEPTED");
				}
				opponentBids.add(opponentBid);
				opponentPreviousBid = opponentBid;
			}
		}
		return chosenAction;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see negotiator.Agent#init()
	 */
	public void init() {
		messageOpponent = null;
		myLastBid = null;
		myLastAction = null;
		myPreviousBids = new ArrayList<Bid>();
		opponentBids = new ArrayList<Bid>();

		try {
			bidSpace = new agents.anac.y2010.Southampton.analysis.BidSpace(
					(AdditiveUtilitySpace) this.utilitySpace);
		} catch (Exception e) {
			e.printStackTrace();
		}

		opponentIsHardHead = true;
	}

	/**
	 * Output a message, but only if debugging is turned on.
	 * 
	 * @param message
	 *            The message to output.
	 */
	public final void log(String message) {
		// sae.log(this, message);
	}

	/**
	 * Propose the initial bid.
	 * 
	 * @return The action to be bid.
	 * @throws Exception
	 */
	protected abstract Bid proposeInitialBid() throws Exception;

	/**
	 * Propose the next bid.
	 * 
	 * @param opponentBid
	 *            The bid that has just been made by the opponent.
	 * @return The action to be bid.
	 * @throws Exception
	 */
	protected abstract Bid proposeNextBid(Bid opponentBid) throws Exception;

	/*
	 * (non-Javadoc)
	 * 
	 * @see negotiator.Agent#ReceiveMessage(negotiator.actions.Action)
	 */
	public final void ReceiveMessage(Action opponentAction) {
		// Log the received opponentAction
		if (opponentAction == null) {
			log("Received (null) from opponent.");
		} else {
			log("--------------------------------------------------------------------------------");
			log("Received " + opponentAction.toString() + " from opponent.");
			if (opponentAction instanceof Offer) {
				try {
					log("It has a utility of "
							+ utilitySpace.getUtility(((Offer) opponentAction)
									.getBid()));

					if (opponentIsHardHead
							&& opponentBids.size() > 0
							&& Math.abs(utilitySpace.getUtility(opponentBids
									.get(0))
									- utilitySpace
											.getUtility(((Offer) opponentAction)
													.getBid())) > 0.02) {
						opponentIsHardHead = false;
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}

		// Store the received opponentAction
		messageOpponent = opponentAction;
	}
}
