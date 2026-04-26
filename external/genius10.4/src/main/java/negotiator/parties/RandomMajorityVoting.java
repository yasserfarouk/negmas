package negotiator.parties;

import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.OfferForVoting;
import genius.core.actions.Reject;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.protocol.AlternatingMajorityConsensusProtocol;
import genius.core.protocol.DefaultMultilateralProtocol;

/**
 * Random agent suited for AlternatingOfferMajorityVotingProtocol
 * <p/>
 * This party should be run with {@link genius.core.protocol.MediatorProtocol}
 *
 * @author W.Pasman
 */
public class RandomMajorityVoting extends AbstractNegotiationParty {

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

	@Override
	public void receiveMessage(AgentID sender, Action arguments) {
		super.receiveMessage(sender, arguments);
		if (arguments instanceof OfferForVoting) {
			if (isAcceptable(((OfferForVoting) arguments).getBid())) {
				lastOfferIsAcceptable = true;
			}
			lastOfferIsAcceptable = false;
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
	public Class<? extends DefaultMultilateralProtocol> getProtocol() {
		return AlternatingMajorityConsensusProtocol.class;
	}

	@Override
	public String getDescription() {
		return "Random Majority Voting Party";
	}

}
