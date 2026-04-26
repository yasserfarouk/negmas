package genius.core.protocol;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.exceptions.NegotiationPartyTimeoutException;
import genius.core.parties.NegotiationParty;
import genius.core.session.ActionException;
import genius.core.session.InvalidActionContentsError;
import genius.core.session.Round;
import genius.core.session.Session;

/**
 * An adapter for the protocol class. This implements default functionality for
 * the methods in the Protocol interface and return default values for them.
 *
 * @author David Festen
 */
public class DefaultMultilateralProtocol implements MultilateralProtocol {

	protected boolean isAborted = false;
	private Bid lastOffer = null;

	@Override
	public Round getRoundStructure(List<NegotiationParty> parties, Session session) {
		return new Round();
	}

	@Override
	public Map<NegotiationParty, List<Action>> beforeSession(Session session, List<NegotiationParty> parties)
			throws NegotiationPartyTimeoutException, ExecutionException, InterruptedException {
		return new HashMap<NegotiationParty, List<Action>>();

	}

	@Override
	public void afterSession(Session session, List<NegotiationParty> parties) {

	}

	@Override
	public void applyAction(Action action, Session session) throws ActionException {
		if (action instanceof Offer) {
			checkOffer((Offer) action);
		}
		if (action instanceof Accept) {
			checkAccept((Accept) action);
		}
	}

	private void checkAccept(Accept action) throws ActionException {
		if (lastOffer == null) {
			throw new InvalidActionContentsError(action.getAgent(),
					"In DefaultMultilateralProtocol, an accept can only be done after an offer has been placed");
		}
		if (!lastOffer.equals(action.getBid())) {
			throw new InvalidActionContentsError(action.getAgent(),
					"In DefaultMultilateralProtocol, only the last placed offer can be accepted.");
		}
	}

	/**
	 * Check incoming offer
	 * 
	 * @param action
	 *            the offer
	 */
	protected void checkOffer(Offer offer) throws ActionException {
		lastOffer = offer.getBid();
	}

	@Override
	public boolean isFinished(Session session, List<NegotiationParty> parties) {
		return isAborted;
	}

	/**
	 * Get a map of parties that are listening to each other's response
	 *
	 * @param parties
	 *            The parties involved in the current negotiation
	 * @return A map where the key is a
	 *         {@link genius.core.parties.NegotiationParty} that is responding to
	 *         a {@link NegotiationParty#chooseAction(java.util.List)} event,
	 *         and the value is a list of {@link NegotiationParty} that are
	 *         listening to that key party's response.
	 */
	@Override
	public Map<NegotiationParty, List<NegotiationParty>> getActionListeners(final List<NegotiationParty> parties) {
		return new HashMap<NegotiationParty, List<NegotiationParty>>(0);
	}

	/**
	 * This method should return the current agreement.
	 * <p/>
	 * Some protocols only have an agreement at the negotiation session, make
	 * sure that this method returns null until the end of the session in that
	 * case, because this method might be queried at intermediary steps.
	 *
	 * @param session
	 *            The complete session history up to this point
	 * @return The agreed upon bid or null if no agreement
	 */
	@Override
	public Bid getCurrentAgreement(Session session, List<NegotiationParty> parties) {
		return null;
	}

	/**
	 * Gets the number of parties that currently agree to the offer.
	 * <p/>
	 * Default implementation returns 0 if no agreement or number of parties if
	 * agreement exists.
	 *
	 * @param session
	 *            the current state of this session
	 * @param parties
	 *            The parties currently participating
	 * @return the number of parties agreeing to the current agreement
	 */
	@Override
	public int getNumberOfAgreeingParties(Session session, List<NegotiationParty> parties) {
		return getCurrentAgreement(session, parties) == null ? 0 : parties.size();
	}

	/**
	 * Filters the list by including only the type of negotiation parties.
	 * Optionally, this behavior can be reversed (i.e. excluding only the given
	 * type of negotiation parties).
	 *
	 * @param negotiationParties
	 *            The original list of parties
	 * @param negotiationPartyClass
	 *            The type of parties to include (or exclude if inclusionFilter
	 *            is set to false)
	 * @param inclusionFilter
	 *            If set to true, we include the given type. Otherwise, exclude
	 *            the given type
	 * @return The filtered list of parties
	 */
	private Collection<NegotiationParty> filter(Collection<NegotiationParty> negotiationParties,
			Class negotiationPartyClass, boolean inclusionFilter) {
		Collection<NegotiationParty> filtered = new ArrayList<NegotiationParty>(negotiationParties.size());

		for (NegotiationParty party : negotiationParties) {
			// if including and class is of the type searching for,
			// or excluding and class is not of the type searching for.
			if ((inclusionFilter && party.getClass().equals(negotiationPartyClass))
					|| (!inclusionFilter && !party.getClass().equals(negotiationPartyClass))) {
				filtered.add(party);
			}
		}

		return filtered;
	}

	/**
	 * Filters the list by including only the type of negotiation parties.
	 * Optionally, this behavior can be reversed (i.e. excluding only the given
	 * type of negotiation parties).
	 *
	 * @param negotiationParties
	 *            The original list of parties
	 * @param negotiationPartyClass
	 *            The type of parties to include
	 *
	 * @return The filtered list of parties
	 */
	public Collection<NegotiationParty> includeOnly(Collection<NegotiationParty> negotiationParties,
			Class negotiationPartyClass) {
		return filter(negotiationParties, negotiationPartyClass, true);
	}

	/**
	 * Filters the list by including only the type of negotiation parties.
	 * Optionally, this behavior can be reversed (i.e. excluding only the given
	 * type of negotiation parties).
	 *
	 * @param negotiationParties
	 *            The original list of parties
	 * @param negotiationPartyClass
	 *            The type of parties to include
	 *
	 * @return The filtered list of parties
	 */
	public Collection<NegotiationParty> exclude(Collection<NegotiationParty> negotiationParties,
			Class negotiationPartyClass) {
		return filter(negotiationParties, negotiationPartyClass, false);
	}

	/**
	 * Overwrites the rest of the protocol and sets the protocol state to finish
	 */
	@Override
	public void endNegotiation() {
		System.out.println("Negotiation aborted");
		isAborted = true;
	}

	/**
	 * Overwrites the rest of the protocol and sets the protocol state to finish
	 *
	 * @param reason
	 *            Optionally give a reason why the protocol is finished.
	 */
	@Override
	public void endNegotiation(String reason) {
		System.out.println("Negotiation aborted: " + reason);
		isAborted = true;
	}

	/**
	 * @param parties
	 *            all the parties in the negotiation
	 * @return map with as keys all {@link NegotiationParty} and as values for
	 *         each key all other {@link NegotiationParty}s.
	 */
	protected static Map<NegotiationParty, List<NegotiationParty>> listenToAll(List<NegotiationParty> parties) {
		// create a new map of parties
		Map<NegotiationParty, List<NegotiationParty>> map = new HashMap<NegotiationParty, List<NegotiationParty>>();

		// for each party add each other party
		for (NegotiationParty listener : parties) {
			ArrayList<NegotiationParty> talkers = new ArrayList<NegotiationParty>();
			for (NegotiationParty talker : parties) {
				if (talker != listener) {
					talkers.add(talker);
				}
			}
			map.put(listener, talkers);
		}

		return map;
	}

	/**
	 * @param parties
	 *            all the parties in the negotiation
	 * @return map with as keys all {@link NegotiationParty} and as values an
	 *         empty list.
	 */
	protected static Map<NegotiationParty, List<NegotiationParty>> listenToNone(List<NegotiationParty> parties) {
		Map<NegotiationParty, List<NegotiationParty>> listenersMap = new HashMap<NegotiationParty, List<NegotiationParty>>();
		for (NegotiationParty party : parties) {
			listenersMap.put(party, parties);
		}
		return listenersMap;
	}

}
