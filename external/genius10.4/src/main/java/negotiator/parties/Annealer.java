package negotiator.parties;

import static java.lang.Math.E;
import static java.lang.Math.max;
import static java.lang.Math.pow;

import java.util.List;

import genius.core.AgentID;
import genius.core.DeadlineType;
import genius.core.Vote;
import genius.core.actions.Action;
import genius.core.actions.InformVotingResult;
import genius.core.actions.OfferForVoting;
import genius.core.actions.VoteForOfferAcceptance;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.parties.NegotiationParty;
import genius.core.protocol.MultilateralProtocol;

/**
 * Implementation of a party that uses simulated annealing strategy to get to an
 * agreement.
 * <p/>
 * This party should be run with {@link genius.core.protocol.MediatorProtocol}
 *
 * @author David Festen
 * @author Reyhan
 */
public class Annealer extends AbstractNegotiationParty {

	/**
	 * Holds the utility value for the most recently accepted offer.
	 */
	private double lastAcceptedBidUtility;

	/**
	 * Holds the utility value for the most recently received offer.
	 */
	private double lastReceivedBidUtility;

	/**
	 * Holds the vote that will be done when asked for voting.
	 */
	private Vote currentVote;

	/**
	 * Holds the current round number (used for cooling down the annealing).
	 */
	private int currentRound;

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		lastAcceptedBidUtility = 0;
		lastReceivedBidUtility = 0;
		currentVote = Vote.REJECT;
		currentRound = 0;
	}

	/**
	 * When this class is called, it is expected that the Party chooses one of
	 * the actions from the possible action list and returns an instance of the
	 * chosen action. This class is only called if this {@link NegotiationParty}
	 * is in the
	 * {@link negotiator.protocol .DefaultProtocol#getRoundStructure(java.util.List, negotiator.session.Session)}
	 * .
	 *
	 * @param possibleActions
	 *            List of all actions possible.
	 * @return The chosen action
	 */
	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {
		return new VoteForOfferAcceptance(getPartyId(), currentVote);
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
		if (arguments instanceof OfferForVoting) {
			lastReceivedBidUtility = getUtility(((OfferForVoting) arguments).getBid());
			if (lastReceivedBidUtility >= lastAcceptedBidUtility) {
				currentVote = Vote.ACCEPT;
			} else {
				double temperature = getNormalizedRoundOrTimeValue();

				double probability = pow(E, ((lastReceivedBidUtility - lastAcceptedBidUtility) / temperature));
				currentVote = probability > rand.nextDouble() ? Vote.ACCEPT : Vote.REJECT;
			}
		} else if (arguments instanceof InformVotingResult) {
			if (((InformVotingResult) arguments).getVotingResult() == Vote.ACCEPT) {
				lastAcceptedBidUtility = lastReceivedBidUtility;
			}
			currentRound++;
		}
	}

	/**
	 * returns the highest of normalized time and round values
	 *
	 * @return a value between 0.0 and 1.0
	 */
	private double getNormalizedRoundOrTimeValue() {
		// return the most urgent value
		return max(getNormalizedRoundValue(), getNormalizedTimeValue());
	}

	/**
	 * Gets the normalized current round between 0.0 and 1.0
	 *
	 * @return A value between 0.0 and 1.0 or 0.0 if no round deadline given
	 */
	private double getNormalizedRoundValue() {
		if (getDeadlines().getType() == DeadlineType.ROUND) {
			return ((double) currentRound / (double) getDeadlines().getValue());
		} else {
			return 0d;
		}
	}

	/**
	 * Gets the normalized current time, which runs from 0.0 to 1.0
	 *
	 * @return A value between 0.0 and 1.0 which defaults to 0.0 if no time
	 *         deadline set
	 */
	private double getNormalizedTimeValue() {
		return timeline == null ? 0d : timeline.getTime();
	}

	@Override
	public String getDescription() {
		return "Annealer Party";
	}

}
