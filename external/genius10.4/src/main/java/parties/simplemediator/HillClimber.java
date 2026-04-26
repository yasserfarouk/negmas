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

public class HillClimber extends AbstractNegotiationParty {

	private double lastAcceptedBidUtility;
	private double lastReceivedBidUtility;
	private Vote currentVote;

	public HillClimber() {
		super();
		lastAcceptedBidUtility = 0.0;
		lastReceivedBidUtility = 0.0;
	}

	@Override
	public void receiveMessage(AgentID sender, Action act) {

		if (act instanceof InformVotingResult) {
			/*
			 * update the utility of last accepted bid by all
			 */
			if (((InformVotingResult) act).getVotingResult() == Vote.ACCEPT)
				lastAcceptedBidUtility = lastReceivedBidUtility;
			return;
		}

		Bid receivedBid = DefaultAction.getBidFromAction(act);
		if (receivedBid == null)
			return;

		if (getTimeLine().getType() == Type.Time)
			lastReceivedBidUtility = getUtilityWithDiscount(receivedBid);
		else
			lastReceivedBidUtility = getUtility(receivedBid);

		if (lastAcceptedBidUtility <= lastReceivedBidUtility)
			currentVote = Vote.ACCEPT;
		else
			currentVote = Vote.REJECT;

	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {
		return (new VoteForOfferAcceptance(getPartyId(), currentVote));
	}

	@Override
	public Class<? extends DefaultMultilateralProtocol> getProtocol() {
		return SimpleMediatorBasedProtocol.class;
	}

	@Override
	public String getDescription() {
		return "HillClimber party";
	}

}
