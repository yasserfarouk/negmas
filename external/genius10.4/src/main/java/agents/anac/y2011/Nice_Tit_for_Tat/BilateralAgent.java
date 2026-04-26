package agents.anac.y2011.Nice_Tit_for_Tat;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.Domain;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;

/**
 * @author Tim Baarslag Agent skeleton for a bilateral agent. It contains
 *         service functions to have access to the bidding history.
 */
public abstract class BilateralAgent extends Agent implements BidHistoryKeeper {
	private static final boolean LOGGING = false;
	protected Domain domain;
	private Action opponentAction;
	protected BidHistory myHistory;
	protected BidHistory opponentHistory;

	public void init() {
		domain = utilitySpace.getDomain();
		myHistory = new BidHistory();
		opponentHistory = new BidHistory();
	}

	@Override
	public String getVersion() {
		return "1.0";
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		this.opponentAction = opponentAction;
		if (opponentAction instanceof Offer) {
			Bid bid = ((Offer) opponentAction).getBid();
			double time = timeline.getTime();
			double myUndiscountedUtility = getUndiscountedUtility(bid);
			BidDetails bidDetails = new BidDetails(bid, myUndiscountedUtility,
					time);
			opponentHistory.add(bidDetails);
		}
	}

	protected double getUndiscountedUtility(Bid bid) {
		double myUndiscountedUtility = 0;
		try {
			myUndiscountedUtility = utilitySpace.getUtility(bid);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return myUndiscountedUtility;
	}

	/**
	 * @param remember
	 *            : remember the action or not.
	 */
	@Override
	public Action chooseAction() {
		Bid opponentLastBid = getOpponentLastBid();
		Action myAction = null;

		// We start
		if (opponentLastBid == null) {
			Bid openingBid = chooseOpeningBid();
			myAction = new Offer(getAgentID(), openingBid);
		}

		// We are second
		else if (getMyLastBid() == null) {
			Bid firstCounterBid = chooseFirstCounterBid();
			// Check to see if we want to accept the first offer
			if (isAcceptable(firstCounterBid))
				myAction = makeAcceptAction();
			else
				myAction = new Offer(getAgentID(), firstCounterBid);
		}

		// We make a normal counter-offer
		else if (opponentAction instanceof Offer) {
			Bid counterBid = chooseCounterBid();
			// Check to see if we want to accept
			if (isAcceptable(counterBid))
				myAction = makeAcceptAction();
			else
				myAction = new Offer(getAgentID(), counterBid);

			if (myAction == null) { // need this code to check equivalence
				myAction = new Offer(getAgentID(), counterBid);
			}
		}
		remember(myAction);
		return myAction;
	}

	/**
	 * By default, if an offer is deemed acceptable, we send accept. Overwrite
	 * this method to change this behaviour.
	 */
	protected Action makeAcceptAction() {
		Action myAction;
		myAction = new Accept(getAgentID(),
				((ActionWithBid) opponentAction).getBid());
		return myAction;
	}

	/**
	 * Remember our action, if it was an offer
	 */
	private void remember(Action myAction) {
		if (myAction instanceof Offer) {
			Bid myLastBid = ((Offer) myAction).getBid();
			double time = timeline.getTime();
			double myUndiscountedUtility = getUndiscountedUtility(myLastBid);
			BidDetails bidDetails = new BidDetails(myLastBid,
					myUndiscountedUtility, time);
			myHistory.add(bidDetails);
		}
	}

	/**
	 * At some point, one of the parties has to accept an offer to end the
	 * negotiation. Use this method to decide whether to accept the last offer
	 * by the opponent.
	 * 
	 * @param plannedBid
	 */
	public abstract boolean isAcceptable(Bid plannedBid);

	/**
	 * The opponent has already made a bid. Use this method to make an counter
	 * bid.
	 */
	public abstract Bid chooseCounterBid();

	/**
	 * Use this method to make an opening bid.
	 */
	public abstract Bid chooseOpeningBid();

	/**
	 * Use this method to make the first counter-bid.
	 */
	public abstract Bid chooseFirstCounterBid();

	public Bid getMyLastBid() {
		return myHistory.getLastBid();
	}

	public Bid getMySecondLastBid() {
		return myHistory.getSecondLastBid();
	}

	public BidHistory getOpponentHistory() {
		return opponentHistory;
	}

	public Bid getOpponentLastBid() {
		return opponentHistory.getLastBid();
	}

	/**
	 * Returns the current round number, starting at 0. This is equal to the
	 * total number of placed bids.
	 */
	public int getRound() {
		return myHistory.size() + opponentHistory.size();
	}

	protected static void log(String s) {
		System.out.println(s);
	}

	/**
	 * Rounds to two decimals
	 */
	public static double round2(double x) {
		return Math.round(100 * x) / 100d;
	}
}
