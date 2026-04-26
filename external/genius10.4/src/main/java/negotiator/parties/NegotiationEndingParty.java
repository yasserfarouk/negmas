package negotiator.parties;

import java.util.List;

import genius.core.AgentID;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;

/**
 * Agent that always ends the nego as quick as possible, only placing a random
 * bid if it starts.
 * <p/>
 * The class was created as part of a series of agents used to understand the
 * api better
 *
 * @author David Festen
 */
public class NegotiationEndingParty extends AbstractNegotiationParty {

	/**
	 * Initializes a new instance of the {@link NegotiationEndingParty} class.
	 *
	 * @param utilitySpace
	 *            The utility space used by this class
	 * @param deadlines
	 *            The deadlines for this session
	 * @param timeline
	 *            The time line (if time deadline) for this session, can be null
	 * @param randomSeed
	 *            The seed that should be used for all randomization (to be
	 *            reproducible)
	 */
	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
	}

	/**
	 * If offer was proposed: Accept offer, otherwise: Propose random offer
	 *
	 * @param possibleActions
	 *            List of all actions possible.
	 * @return Accept or Offer action
	 */
	@Override
	public Action chooseAction(final List<Class<? extends Action>> possibleActions) {

		if (possibleActions.contains(EndNegotiation.class)) {
			return new EndNegotiation(getPartyId());
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
		return "Stop Negotiation Party";
	}

}
