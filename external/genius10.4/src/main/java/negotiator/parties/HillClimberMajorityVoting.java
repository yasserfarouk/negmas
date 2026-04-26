package negotiator.parties;

import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.InformVotingResult;
import genius.core.actions.OfferForVoting;
import genius.core.actions.Reject;
import genius.core.parties.AbstractNegotiationParty;

/**
 * Implementation of a party that uses hill climbing strategy to get to an
 * agreement.
 * <p/>
 * This party should be run with {@link genius.core.protocol.MediatorProtocol}
 *
 * @author David Festen
 * @author Reyhan
 */
public class HillClimberMajorityVoting extends AbstractNegotiationParty {

	private boolean lastOfferIsAcceptable = false;

	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {
		if (possibleActions.contains(OfferForVoting.class)) {
			return new OfferForVoting(getPartyId(), this.generateRandomBid());
		} else {
			if (lastOfferIsAcceptable) {
				return new Accept(getPartyId(), ((ActionWithBid) getLastReceivedAction()).getBid());
			}
			return new Reject(getPartyId(), ((ActionWithBid) getLastReceivedAction()).getBid());
		}
	}

	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		if (action instanceof OfferForVoting) {
			if (isAcceptable(((OfferForVoting) action).getBid())) {
				lastOfferIsAcceptable = true;
			}
			lastOfferIsAcceptable = false;
		} else if (action instanceof InformVotingResult) {
			// WHATEVER.
		}
	}

	protected boolean isAcceptable(Bid bid) {
		double lastReceivedBidUtility = getUtility(bid);
		double reservationValue = (timeline == null) ? utilitySpace.getReservationValue()
				: utilitySpace.getReservationValueWithDiscount(timeline);

		if (lastReceivedBidUtility >= reservationValue) {
			return true;
		}
		return false;
	}

	@Override
	public String getDescription() {
		return "Hill Climber Majority Voting";
	}

}
