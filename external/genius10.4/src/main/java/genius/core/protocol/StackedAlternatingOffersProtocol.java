package genius.core.protocol;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Inform;
import genius.core.actions.Offer;
import genius.core.exceptions.NegotiationPartyTimeoutException;
import genius.core.parties.NegotiationParty;
import genius.core.session.Round;
import genius.core.session.Session;
import genius.core.session.Turn;

/**
 * Implementation of an alternating offer protocol using offer/counter-offer.
 * <p/>
 * Protocol:
 * 
 * <pre>
 * The first agent makes an offer
 * Other agents can accept or make a counter-offer
 * 
 * If no agent makes a counter-offer, the negotiation end with this offer.
 * Otherwise, the process continues until reaching deadline or agreement.
 * </pre>
 *
 * @author David Festen
 * @author Reyhan Aydogan
 * @author Catholijn Jonker
 */
public class StackedAlternatingOffersProtocol extends DefaultMultilateralProtocol {

	static final AgentID protocolID = new AgentID("StackedAlternatingOffersProtocol");

	/**
	 * Get all possible actions for given party in a given session.
	 *
	 * @param parties
	 * @param session
	 *            the current state of this session
	 * @return A list of possible actions
	 */

	/**
	 * Defines the round structure.
	 * 
	 * <pre>
	 * The first agent makes an offer
	 * Other agents can accept or make a counter-offer
	 * </pre>
	 *
	 * @param parties
	 *            The parties currently participating
	 * @param session
	 *            The complete session history
	 * @return A list of possible actions
	 */
	@Override
	public Round getRoundStructure(List<NegotiationParty> parties, Session session) {
		Round round = new Round();
		// we will create the very first action if this is the first round
		boolean isVeryFirstAction = session.getRoundNumber() == 0;

		for (NegotiationParty party : parties) {
			if (isVeryFirstAction) {
				// If this is the first party in the first round, it can not
				// accept.
				round.addTurn(new Turn(party, Offer.class, EndNegotiation.class));
			} else {
				// Each party can either accept the outstanding offer, or
				// propose a counteroffer.
				round.addTurn(new Turn(party, Accept.class, Offer.class, EndNegotiation.class));
			}
			isVeryFirstAction = false;
		}

		// return round structure
		return round;
	}

	/**
	 * Will return the current agreement.
	 *
	 * @param session
	 *            The complete session history up to this point
	 * @return The agreed upon bid or null if no agreement
	 */
	@Override
	public Bid getCurrentAgreement(Session session, List<NegotiationParty> parties) {

		// if not all parties agree, we did not find an agreement
		if (getNumberOfAgreeingParties(session, parties) < parties.size())
			return null;

		// all parties agreed, return most recent offer
		return getMostRecentBid(session);
	}

	@Override
	public int getNumberOfAgreeingParties(Session session, List<NegotiationParty> parties) {
		int nAccepts = 0;
		ArrayList<Action> actions = getMostRecentTwoRounds(session);
		for (int i = actions.size() - 1; i >= 0; i--) {
			if (actions.get(i) instanceof Accept) {
				nAccepts++;
			} else {
				if (actions.get(i) instanceof Offer) {
					// voting party also counts towards agreeing parties
					nAccepts++;
				}
				// we found at least one not accepting party (offering/ending
				// negotiation) so stop counting
				break;
			}
		}
		return nAccepts;
	}

	/**
	 * Get a list of all actions of the most recent two rounds
	 * 
	 * @param session
	 *            Session to extract the most recent two rounds out of
	 * @return A list of actions done in the most recent two rounds.
	 */
	private ArrayList<Action> getMostRecentTwoRounds(Session session) {

		// holds actions
		ArrayList<Action> actions = new ArrayList<Action>();

		// add previous round if exists
		if (session.getRoundNumber() >= 2) {
			Round round = session.getRounds().get(session.getRoundNumber() - 2);
			for (Action action : round.getActions())
				actions.add(action);
		}

		// add current round if exists (does not exists before any offer is
		// made)
		if (session.getRoundNumber() >= 1) {
			Round round = session.getRounds().get(session.getRoundNumber() - 1);
			for (Action action : round.getActions())
				actions.add(action);
		}

		// return aggregated actions
		return actions;
	}

	private Bid getMostRecentBid(Session session) {

		// reverse rounds/actions until offer is found or return null
		for (int roundIndex = session.getRoundNumber() - 1; roundIndex >= 0; roundIndex--) {
			for (int actionIndex = session.getRounds().get(roundIndex).getActions().size()
					- 1; actionIndex >= 0; actionIndex--) {
				Action action = session.getRounds().get(roundIndex).getActions().get(actionIndex);
				if (action instanceof Offer)
					return ((Offer) action).getBid();
			}
		}

		// No offer found, so return null (no most recent bid exists)
		// since this is only possible when first party quits negotiation, it is
		// probably a bug when this happens
		System.err.println("Possible bug: No Offer was placed during negotiation");
		return null;
	}

	/**
	 * If all agents accept the most recent offer, then this negotiation ends.
	 * Also, when any agent ends the negotiation (EndNegotiationAction) the
	 * negotiation ends
	 *
	 * @param session
	 *            the current state of this session
	 * @return true if the protocol is finished
	 */
	@Override
	public boolean isFinished(Session session, List<NegotiationParty> parties) {
		return getCurrentAgreement(session, parties) != null || session.getMostRecentAction() instanceof EndNegotiation;
	}

	/**
	 * Get a map of parties that are listening to each other.
	 *
	 * @return who is listening to who
	 */
	@Override
	public Map<NegotiationParty, List<NegotiationParty>> getActionListeners(List<NegotiationParty> parties) {

		return listenToAll(parties);
	}

	@Override
	public Map<NegotiationParty, List<Action>> beforeSession(Session session, final List<NegotiationParty> parties)
			throws InterruptedException, ExecutionException, NegotiationPartyTimeoutException {
		HashMap<NegotiationParty, List<Action>> map = new HashMap<NegotiationParty, List<Action>>();

		for (NegotiationParty party : parties) {
			ArrayList<Action> actions = new ArrayList<Action>();
			actions.add(new Inform(protocolID, "NumberOfAgents", parties.size()));
			map.put(party, actions);
		}

		return map;
	}
}
