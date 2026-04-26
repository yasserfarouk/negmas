package negotiator.parties;

import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.actions.OfferForVoting;
import genius.core.actions.Reject;
import genius.core.parties.AbstractNegotiationParty;

/**
 * Basic voting implementation: this agent accepts and rejects offers with a 50%
 * chance.
 * <p/>
 * The class was created as part of a series of agents used to understand the
 * api better
 *
 * @author David Festen
 */
public class RandomFiftyFiftyNegotiationParty extends AbstractNegotiationParty {

	private Bid lastBid;

	/**
	 * If placing offers: do random offer if voting: accept/reject with a 50%
	 * chance on both
	 *
	 * @param possibleActions
	 *            List of all actions possible.
	 * @return The chosen action
	 */
	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {

		// if we are the first party, place offer.
		if (possibleActions.contains(OfferForVoting.class)) {
			return new OfferForVoting(getPartyId(), generateRandomBid());
		}

		// else do 50/50 accept or reject
		return rand.nextBoolean() ? new Accept(getPartyId(), lastBid) : new Reject(getPartyId(), lastBid);
	}

	/**
	 * Processes action messages received by a given sender.
	 *
	 * @param sender
	 *            The initiator of the action
	 * @param arguments
	 *            The action performed
	 */
	@Override
	public void receiveMessage(AgentID sender, Action action) {
		if (action instanceof Offer) {
			lastBid = ((Offer) action).getBid();
		}
	}

	@Override
	public String getDescription() {
		return "Random Fifty-Fifty Party";
	}

}
