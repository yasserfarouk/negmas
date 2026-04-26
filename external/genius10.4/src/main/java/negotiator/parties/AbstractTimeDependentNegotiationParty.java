package negotiator.parties;

import static java.lang.Math.pow;

import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.NoAction;
import genius.core.actions.Offer;
import genius.core.actions.OfferForVoting;
import genius.core.actions.Reject;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.protocol.MultilateralProtocol;
import genius.core.timeline.DiscreteTimeline;

/**
 * Boulware/Conceder tactics, by Tim Baarslag, adapted from [1]. Adapted by Mark
 * Hendrikx to use the SortedOutcomeSpace instead of BidHistory. Adapted by
 * David Festen for multilateral case.
 *
 * [1] S. Shaheen Fatima Michael Wooldridge Nicholas R. Jennings Optimal
 * Negotiation Strategies for Agents with Incomplete Information
 * http://eprints.ecs.soton.ac.uk/6151/1/atal01.pdf
 *
 * @author Tim Baarslag, Mark Hendrikx
 */
public abstract class AbstractTimeDependentNegotiationParty extends AbstractNegotiationParty 
{
	SortedOutcomeSpace outcomeSpace;
	Bid lastReceivedBid = null;

	@Override
	public void init(NegotiationInfo info) 
	{
		super.init(info);
		outcomeSpace = new SortedOutcomeSpace(getUtilitySpace());
	}

	/**
	 * When this class is called, it is expected that the Party chooses one of
	 * the actions from the possible action list and returns an instance of the
	 * chosen action. This class is only called if this
	 * {@link genius.core.parties.NegotiationParty} is in the
	 * {@link MultilateralProtocol#getRoundStructure(java.util.List, negotiator.session.Session)}
	 * .
	 *
	 * @param possibleActions
	 *            List of all actions possible.
	 * @return The chosen action
	 */
	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {
		Bid nextBid = getNextBid();
		double lastUtil = lastReceivedBid != null ? utilitySpace.getUtilityWithDiscount(lastReceivedBid, timeline) : 0;
		double nextUtil = nextBid != null ? utilitySpace.getUtilityWithDiscount(nextBid, timeline) : 0;

		// Accept is for both voting and counter offer protocols
		if (possibleActions.contains(Accept.class) && nextUtil < lastUtil)
			return new Accept(getPartyId(), lastReceivedBid);

		// Counter offer based actions
		else if (possibleActions.contains(Offer.class))
			return new Offer(getPartyId(), nextBid);

		// Voting based actions
		else if (possibleActions.contains(OfferForVoting.class))
			return new OfferForVoting(getPartyId(), nextBid);
		else if (possibleActions.contains(Reject.class))
			return new Reject(getPartyId(), lastReceivedBid); // Accept is higher up the chain
		// default action
		else
			return new NoAction(getPartyId());
	}

	/**
	 * Get the next bid we should do
	 */
	protected Bid getNextBid() {
		return outcomeSpace.getBidNearUtility(getTargetUtility()).getBid();
	}

	/**
	 * This method is called when an observable action is performed. Observable
	 * actions are defined in
	 * {@link MultilateralProtocol#getActionListeners(java.util.List)}
	 *
	 * @param sender
	 *            The initiator of the action
	 * @param arguments
	 *            The action performed
	 */
	@Override
	public void receiveMessage(AgentID sender, Action arguments) {
		if (arguments instanceof Offer)
			lastReceivedBid = ((Offer) arguments).getBid();
	}

	/**
	 * Gets the target utility for the next bid
	 *
	 * @return The target utility for the given time
	 */
	public double getTargetUtility() {

		// timeline runs from 0.0 to 1.0

		// we have a slight offset because discrete timeline is 1-based, this
		// needs to be addressed
		double offset = timeline instanceof DiscreteTimeline ? 1d / ((DiscreteTimeline) timeline).getTotalRounds() : 0d;
		double target = 1d - f(timeline.getTime() - offset);
//		System.out.println("Target util: " + target);
		return target;
	}

	/**
	 * From [1]:
	 *
	 * A wide range of time dependent functions can be defined by varying the
	 * way in which f(t) is computed. However, functions must ensure that 0 <=
	 * f(t) <= 1, f(0) = k, and f(1) = 1.
	 *
	 * That is, the offer will always be between the value range, at the
	 * beginning it will give the initial constant and when the deadline is
	 * reached, it will offer the reservation value.
	 *
	 * For e = 0 (special case), it will behave as a Hardliner.
	 */
	public double f(double t) {
		if (getE() == 0) {
			return 0;
		}
		return pow(t, 1 / getE());
	}

	/**
	 * Depending on the value of e, extreme sets show clearly different patterns
	 * of behaviour [1]:
	 *
	 * 1. Boulware: For this strategy e < 1 and the initial offer is maintained
	 * till time is almost exhausted, when the agent concedes up to its
	 * reservation value.
	 *
	 * 2. Conceder: For this strategy e > 1 and the agent goes to its
	 * reservation value very quickly.
	 *
	 * 3. When e = 1, the price is increased linearly.
	 *
	 * 4. When e = 0, the agent plays hardball.
	 */
	public abstract double getE();
}
