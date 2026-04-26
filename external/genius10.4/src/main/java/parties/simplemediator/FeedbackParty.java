package parties.simplemediator;

import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.Feedback;
import genius.core.Vote;
import genius.core.actions.Action;
import genius.core.actions.DefaultAction;
import genius.core.actions.GiveFeedback;
import genius.core.actions.InformVotingResult;
import genius.core.actions.OfferForFeedback;
import genius.core.actions.OfferForVoting;
import genius.core.actions.VoteForOfferAcceptance;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.protocol.DefaultMultilateralProtocol;
import genius.core.protocol.SimpleMediatorBasedProtocol;
import genius.core.timeline.Timeline.Type;

public class FeedbackParty extends AbstractNegotiationParty {

	private double lastBidUtility;
	private double lastAcceptedUtility;
	private double currentBidUtility;
	private Feedback currentFeedback;
	private Vote currentVote;
	private boolean voteTime;

	public FeedbackParty() {
		super();
		lastBidUtility = 0.0;
		lastAcceptedUtility = 0.0;
		currentBidUtility = 0.0;
		currentFeedback = Feedback.SAME;
		voteTime = false;
	}

	@Override
	public void receiveMessage(AgentID sender, Action opponentAction) {

		if (opponentAction instanceof InformVotingResult) {

			if (((InformVotingResult) opponentAction).getVotingResult() == Vote.ACCEPT) // update
																						// the
																						// utility
																						// of
																						// last
																						// accepted
																						// bid
																						// by
																						// all
				lastAcceptedUtility = currentBidUtility;
			return;
		}

		Bid receivedBid = DefaultAction.getBidFromAction(opponentAction);
		if (receivedBid == null)
			return;

		if (getTimeLine().getType() == Type.Time)
			currentBidUtility = getUtilityWithDiscount(receivedBid);
		else
			currentBidUtility = getUtility(receivedBid);

		if (opponentAction instanceof OfferForFeedback) {
			currentFeedback = Feedback.madeupFeedback(lastBidUtility, currentBidUtility);
			voteTime = false;
		}
		if (opponentAction instanceof OfferForVoting) {
			voteTime = true;
			if (lastAcceptedUtility <= currentBidUtility)
				currentVote = Vote.ACCEPT;
			else
				currentVote = Vote.REJECT;
		}

		lastBidUtility = currentBidUtility;

	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {

		if (voteTime)
			return (new VoteForOfferAcceptance(getPartyId(), currentVote));
		else
			return (new GiveFeedback(getPartyId(), currentFeedback));

	}

	@Override
	public Class<? extends DefaultMultilateralProtocol> getProtocol() {
		return SimpleMediatorBasedProtocol.class;
	}

	@Override
	public String getDescription() {
		return "Feedback Negotiator";
	}

}
