package genius.core.events;

import java.util.List;

import genius.core.Bid;
import genius.core.actions.Action;
import genius.core.parties.NegotiationPartyInternal;

/**
 * Agent did an action.
 */
public class MultipartyNegoActionEvent implements NegotiationEvent {
	private int round;
	private int turn;
	private double time; // current run time in seconds.
	private List<NegotiationPartyInternal> parties;
	private Bid agreement;
	private Action action;

	/**
	 * 
	 * @param action
	 *            the action done by the agent
	 * @param round
	 *            the current round number
	 * @param turn
	 *            the turn within the round
	 * @param time
	 *            the time, running from t = 0 (start) to t = 1 (deadline). The
	 *            time is normalized, so agents need not be concerned with the
	 *            actual internal clock.
	 * 
	 * @param parties
	 *            the discounted utils of the parties
	 * @param agreed
	 *            the agreement {@link Bid} that the parties agreed on , or null
	 *            if no agreement yet.
	 */
	public MultipartyNegoActionEvent(Action action, int round, int turn, double time,
			List<NegotiationPartyInternal> parties, Bid agreed) {
		this.action = action;
		this.round = round;
		this.turn = turn;
		this.time = time;
		this.parties = parties;
		this.agreement = agreed;
	}

	/**
	 * 
	 * @return the current round number
	 */
	public int getRound() {
		return round;
	}

	/**
	 * 
	 * @return the current turn within this round
	 */
	public int getTurn() {
		return turn;
	}

	/**
	 * 
	 * @return the time, running from t = 0 (start) to t = 1 (deadline). The
	 *         time is normalized, so agents need not be concerned with the
	 *         actual internal clock.
	 * 
	 * 
	 */
	public double getTime() {
		return time;
	}

	public String toString() {
		return "MultipartyNegotiationOfferEvent[" + action + " at " + round + " round]";
	}

	public Action getAction() {
		return action;
	}

	public List<NegotiationPartyInternal> getParties() {
		return parties;
	}

	public Bid getAgreement() {
		return agreement;
	}

}
