package genius.gui.progress.session;

import java.util.ArrayList;
import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.parties.PartyWithUtility;
import genius.core.utility.UtilitySpace;

/**
 * One outcome in a negotiation
 */
public class Outcome {
	private int round = 1;
	private int turn = 1;
	private Bid bid;
	private List<? extends PartyWithUtility> participants;
	private List<Double> discUtils = new ArrayList<Double>();

	// true if this bid is an agreement
	private boolean isAgreement;
	private AgentID agentID;

	/**
	 * Create new outcome. We calculate discounted utilities right here.
	 * 
	 * @param bid
	 *            the current bid or accepted bid (if party does an accept for a
	 *            bid).
	 * @param round
	 * @param turn
	 * @param parties
	 * @param isAgreement
	 * @param agent
	 *            the agent that placed this bid
	 * @param time
	 *            the time at which this bid was placed.
	 */
	public Outcome(Bid bid, int round, int turn, List<? extends PartyWithUtility> parties, boolean isAgreement,
			AgentID agent, double time) {
		this.round = round;
		this.turn = turn;
		this.bid = bid;
		this.participants = parties;
		this.isAgreement = isAgreement;
		this.agentID = agent;
		for (PartyWithUtility party : participants) {
			UtilitySpace us = party.getUtilitySpace();
			discUtils.add(us.discount(us.getUtility(bid), time));
		}
	}

	public int getRound() {
		return round;
	}

	public int getTurn() {
		return turn;
	}

	public Bid getBid() {
		return bid;
	}

	public List<? extends PartyWithUtility> getParticipants() {
		return participants;
	}

	public List<Double> getDiscountedUtilities() {
		return discUtils;
	}

	public boolean isAgreement() {
		return isAgreement;
	}

	/**
	 * @return The agent that placed this bid.
	 * 
	 */
	public AgentID getAgentID() {
		return agentID;
	}
}
