package negotiator.parties;

import java.util.List;

import genius.core.AgentID;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;

/**
 * Most basic voting agent implementation I could think of: this agent accepts
 * any offer.
 * <p/>
 * The class was created as part of a series of agents used to understand the
 * api better
 *
 * @author David Festen
 */
public class AcceptingNegotiationParty extends AbstractNegotiationParty {

	/**
	 * If offer was proposed: Accept offer, otherwise: Propose random offer
	 *
	 * @param possibleActions
	 *            List of all actions possible.
	 * @return Accept or Offer action
	 */
	@Override
	public Action chooseAction(final List<Class<? extends Action>> possibleActions) {

		System.out.println("getNumberOfParties() = " + getNumberOfParties());

		if (possibleActions.contains(Accept.class)) {
			return new Accept(getPartyId(), ((ActionWithBid) getLastReceivedAction()).getBid());
		} else {
			return new Offer(getPartyId(), generateRandomBid());
		}
	}

	/**
	 * We ignore any messages received.
	 *
	 * @param sender
	 *            The initiator of the action
	 * @param arguments
	 *            The action performed
	 */
	@Override
	public void receiveMessage(final AgentID sender, final Action arguments) {
		super.receiveMessage(sender, arguments);
	}

	@Override
	public String getDescription() {
		return "Always Accepting Party";
	}

}
