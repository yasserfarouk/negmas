package agents.anac.y2013.MetaAgent.portfolio.IAMhaggler2012.agents2011;

import java.util.ArrayList;
import java.util.Random;

import agents.Jama.Matrix;
import agents.anac.y2013.MetaAgent.portfolio.IAMhaggler2012.agents2011.southampton.utils.ActionCreator;
import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;

/**
 * @author Colin Williams
 * 
 */
public abstract class SouthamptonAgent extends VersionIndependentAgent {

	private static enum ActionType {
		ACCEPT, BREAKOFF, OFFER, START;
	}

	/**
	 * Our maximum aspiration level.
	 */
	protected double MAXIMUM_ASPIRATION = 0.9;

	/**
	 * Gets the version number.
	 * 
	 * @return the version number.
	 */
	@Override
	public String getVersion() {
		return "2.0";
	}

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
	 * The opponent's previous bid.
	 */
	protected Bid opponentPreviousBid = null;

	protected double acceptMultiplier = 1.02;

	private boolean opponentIsHardHead;

	private ArrayList<Bid> opponentBids;

	private ArrayList<Bid> myPreviousBids;

	protected boolean debug;

	public final Action chooseAction(long ourTime, long opponentTime) {
		setOurTime(ourTime);
		setOpponentTime(opponentTime);
		return chooseAction(false);
	}

	public final Action chooseAction() {
		return chooseAction(true);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see negotiator.Agent#chooseAction()
	 */
	private final Action chooseAction(boolean recordTimes) {
		Action chosenAction = null;
		Bid opponentBid = null;
		log("Choose action");

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
					Bid b = proposeInitialBid();
					if (b == null)
						chosenAction = new EndNegotiation(getAgentID());
					else
						chosenAction = new Offer(getAgentID(), b);
				} else {
					chosenAction = this.myLastAction;
				}
				break;
			}

		} catch (Exception e) {
			log("Exception in chooseAction:" + e.getMessage());
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
	 * Handle an opponent's offer.
	 * 
	 * @param opponentBid
	 *            The bid made by the opponent.
	 * @return the action that we should take in response to the opponent's
	 *         offer.
	 * @throws Exception
	 */
	private Action handleOffer(Bid opponentBid) throws Exception {
		Action chosenAction = null;

		if (myLastAction == null) {
			// Special case to handle first action
			Bid b = proposeInitialBid();
			if (b == null) {
				chosenAction = ActionCreator.createEndNegotiation(this);
			} else {
				myLastBid = b;
				chosenAction = ActionCreator.createOffer(this, b);
			}
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
			if (plannedBid == null)
				chosenAction = ActionCreator.createEndNegotiation(this);
			else
				chosenAction = ActionCreator.createOffer(this, plannedBid);

			if (opponentBid == null)
				logError("opponentBid is null");
			if (plannedBid == null)
				logError("plannedBid is null");

			if (utilitySpace.getUtility(opponentBid) * acceptMultiplier >= utilitySpace
					.getUtility(plannedBid)) {
				// Accept opponent's bid based on my planned bid.
				chosenAction = ActionCreator.createAccept(this, opponentBid);
				log("Opponent's bid is good enough compared to my planned bid, ACCEPTED");
			}
			opponentBids.add(opponentBid);
			opponentPreviousBid = opponentBid;
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

		log(this.utilitySpace.toString());

		myPreviousBids = new ArrayList<Bid>();
		opponentBids = new ArrayList<Bid>();
		opponentIsHardHead = true;
	}

	/**
	 * Output a message, but only if debugging is turned on.
	 * 
	 * @param message
	 *            The message to output.
	 */
	public final void log(String message) {
		if (debug)
			System.out.println(message);
	}

	/**
	 * Output a message, but only if debugging is turned on.
	 * 
	 * @param message
	 *            The message to output.
	 */
	public final void logError(String message) {
		if (debug)
			System.err.println(message);
	}

	/**
	 * Output a message, but only if debugging is turned on.
	 * 
	 * @param message
	 *            The message to output.
	 */
	public final void flushLog() {
		if (debug)
			System.out.flush();
	}

	/**
	 * Output a matrix, but only if debugging is turned on.
	 * 
	 * @param matrix
	 *            The matrix to output.
	 */
	public final void log(Matrix matrix) {
		if (debug)
			matrix.print(7, 4);
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

	public final void ReceiveMessage(Action opponentAction, long ourTime,
			long opponentTime) {
		setOurTime(ourTime);
		setOpponentTime(opponentTime);
		ReceiveMessage(opponentAction, true);
	}

	public final void ReceiveMessage(Action opponentAction) {
		ReceiveMessage(opponentAction, true);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see negotiator.Agent#ReceiveMessage(negotiator.actions.Action)
	 */
	private final void ReceiveMessage(Action opponentAction, boolean recordTimes) {

		// Log the received opponentAction
		if (opponentAction == null) {
			log("Received (null) from opponent.");
		} else {
			log("--------------------------------------------------------------------------------");
			log("Received " + opponentAction.toString() + " from opponent.");
			if (opponentAction instanceof Offer) {
				OfferReceived((Offer) opponentAction);
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
						log("Opponent of " + getName()
								+ " no longer considered to be hardheaded");
						opponentIsHardHead = false;
					}

				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}

		// Store the received opponentAction
		messageOpponent = opponentAction;
	}

	public void OfferReceived(Offer opponentAction) {
	}
}
