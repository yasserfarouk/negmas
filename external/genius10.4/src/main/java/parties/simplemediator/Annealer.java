package parties.simplemediator;

import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.Vote;
import genius.core.actions.Action;
import genius.core.actions.DefaultAction;
import genius.core.actions.InformVotingResult;
import genius.core.actions.VoteForOfferAcceptance;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.protocol.DefaultMultilateralProtocol;
import genius.core.protocol.SimpleMediatorBasedProtocol;
import genius.core.timeline.Timeline.Type;

public class Annealer extends AbstractNegotiationParty {

	private double lastAcceptedBidUtility;
	private double lastReceivedBidUtility;
	private Vote currentVote;

	public Annealer() {
		super();
		lastAcceptedBidUtility = 0.0;
		lastReceivedBidUtility = 0.0;
	}

	@Override
	public void receiveMessage(AgentID sender, Action opponentAction) {

		if (opponentAction instanceof InformVotingResult) {
			if (((InformVotingResult) opponentAction).getVotingResult() == Vote.ACCEPT)
				/*
				 * update the utility of last accepted bid by all
				 */
				lastAcceptedBidUtility = lastReceivedBidUtility;
			return;
		}

		Bid receivedBid = DefaultAction.getBidFromAction(opponentAction);
		if (receivedBid == null)
			return;

		if (getTimeLine().getType() == Type.Time)
			lastReceivedBidUtility = getUtilityWithDiscount(receivedBid);
		else
			lastReceivedBidUtility = getUtility(receivedBid);

		if (lastAcceptedBidUtility <= lastReceivedBidUtility)
			currentVote = Vote.ACCEPT;
		else {

			double T = getTimeLine().getTime();

			double probability = Math.pow(Math.E, ((double) (lastReceivedBidUtility - lastAcceptedBidUtility) / T));

			if (probability > Math.random())
				currentVote = Vote.ACCEPT;
			else
				currentVote = Vote.REJECT;
		}

	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		return (new VoteForOfferAcceptance(getPartyId(), currentVote));
	}

	@Override
	public Class<? extends DefaultMultilateralProtocol> getProtocol() {
		return SimpleMediatorBasedProtocol.class;
	}

	@Override
	public String getDescription() {
		return "Annealer";
	}

}
