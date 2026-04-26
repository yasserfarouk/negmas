package negotiator.parties;

import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;

/**
 * Basic voting implementation: this agent accepts and rejects offers with a 50%
 * chance.
 * <p/>
 * The class was created as part of a series of agents used to understand the
 * api better
 *
 * @author David Festen
 */
public class RandomCounterOfferNegotiationParty	extends AbstractNegotiationParty 
{
	private Bid lastOffer;

	/**
	 * Initializes a new instance of the
	 * {@link negotiator.parties.RandomCounterOfferNegotiationParty} class.
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
	 * If placing offers: do random offer if voting: accept/reject with a 50%
	 * chance on both
	 *
	 * @param possibleActions
	 *            List of all actions possible.
	 * @return The chosen action
	 */
	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {

		// if we can not accept, we need to do random voting
		if (!possibleActions.contains(Accept.class)) {
			return new Offer(getPartyId(), generateRandomBid());
		}

		// else do 10/90: random offer, accept
		return timeline.getTime() >= 0.9 ? new Accept(getPartyId(), lastOffer)
				: new Offer(getPartyId(), generateRandomNonZeroBid());
	}

	private Bid generateRandomNonZeroBid() {
		Bid randomBid;
		double util;
		do {
			randomBid = generateRandomBid();
			try {
				util = utilitySpace.getUtility(randomBid);
			} catch (Exception e) {
				util = 0.0;
			}
		} while (util < 0.1);
		return randomBid;
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
			lastOffer = ((Offer) action).getBid();
		}
	}

	@Override
	public String getDescription() {
		return "Place random bids until 90% of time";
	}

}
