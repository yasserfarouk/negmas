package negotiator.parties;

import java.util.List;

import genius.core.AgentID;
import genius.core.Vote;
import genius.core.actions.Action;
import genius.core.actions.InformVotingResult;
import genius.core.actions.OfferForVoting;
import genius.core.actions.VoteForOfferAcceptance;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.parties.NegotiationParty;
import genius.core.protocol.DefaultMultilateralProtocol;
import genius.core.protocol.MediatorProtocol;
import genius.core.protocol.MultilateralProtocol;

/**
 * Implementation of a party that uses hill climbing strategy to get to an
 * agreement.
 * <p/>
 * This party should be run with {@link genius.core.protocol.MediatorProtocol}
 *
 * @author David Festen
 * @author Reyhan
 */
public class HillClimber extends AbstractNegotiationParty {

	private double lastAcceptedBidUtility;

	private double lastReceivedBidUtility;

	private Vote currentVote;

	/**
	 * Initializes a new instance of the {@link HillClimber} class.
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
		lastAcceptedBidUtility = 0;
		lastReceivedBidUtility = 0;
		currentVote = Vote.REJECT;
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
			double reservationValue = (timeline == null) ? utilitySpace.getReservationValue()
					: utilitySpace.getReservationValueWithDiscount(timeline);

			if (lastReceivedBidUtility < reservationValue) {
				currentVote = Vote.REJECT;
			} else {
				currentVote = lastReceivedBidUtility >= lastAcceptedBidUtility ? Vote.ACCEPT : Vote.REJECT;
			}
		} else if (arguments instanceof InformVotingResult) {
			if (((InformVotingResult) arguments).getVotingResult() == Vote.ACCEPT) {
				lastAcceptedBidUtility = lastReceivedBidUtility;
			}
		}
	}

	@Override
	public Class<? extends DefaultMultilateralProtocol> getProtocol() {
		return MediatorProtocol.class;
	}

	@Override
	public String getDescription() {
		return "Hill Climber Party";
	}
}
