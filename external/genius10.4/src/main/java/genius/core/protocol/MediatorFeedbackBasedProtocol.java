package genius.core.protocol;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import genius.core.Bid;
import genius.core.Feedback;
import genius.core.Vote;
import genius.core.actions.Action;
import genius.core.actions.GiveFeedback;
import genius.core.actions.OfferForFeedback;
import genius.core.parties.NegotiationParty;
import genius.core.session.Round;
import genius.core.session.Session;
import genius.core.session.Turn;

/**
 * This is the protocol for the mediated feedback. This protocol has this cycle:
 * <ol>
 * <li>mediator proposes bid
 * <li>participants give feedback (better, worse, same)
 * </ol>
 * 
 * The accepted bid is the bid from the last round that contained only "better"
 * or "same" votes.
 */
public class MediatorFeedbackBasedProtocol extends MediatorProtocol {

	@Override
	public Round getRoundStructure(List<NegotiationParty> parties, Session session) {

		// initialize and split parties
		Round round = new Round();
		NegotiationParty mediator = getMediator(parties);
		List<NegotiationParty> otherParties = getNonMediators(parties);

		// mediator places offer
		round.addTurn(new Turn(mediator, OfferForFeedback.class));

		// other parties give feedback
		for (NegotiationParty otherParty : otherParties) {
			round.addTurn(new Turn(otherParty, GiveFeedback.class));
		}

		return round;
	}

	@Override
	public Map<NegotiationParty, List<NegotiationParty>> getActionListeners(List<NegotiationParty> parties) {

		Map<NegotiationParty, List<NegotiationParty>> map = new HashMap<NegotiationParty, List<NegotiationParty>>();

		NegotiationParty mediator = getMediator(parties);

		// all other negotiating parties listen to the mediator
		for (NegotiationParty party : getNonMediators(parties)) {
			map.put(party, Arrays.asList(mediator));
		}

		// the mediator listens to all other negotiating parties.
		map.put(mediator, getNonMediators(parties));

		return map;
	}

	/**
	 * 
	 * Returns the last offer that got accepted.
	 *
	 * @param session
	 *            The complete session history up to this point
	 * @return The current agreement (the bid from the last action from the
	 *         mediator that was {@link Vote#ACCEPT}), or null if no agreement
	 *         yet.
	 */
	@Override
	public Bid getCurrentAgreement(Session session, List<NegotiationParty> parties) {

		// search from last to first bid for an accepted bid.
		List<Round> rounds = new ArrayList<Round>(session.getRounds());
		Collections.reverse(rounds);
		for (Round round : rounds) {
			Bid agreement = getAcceptedBid(round);
			if (agreement != null)
				return agreement;
		}

		return null;
	}

	/********************* support functions ************************/

	/**
	 * A round contains an accepted bid if all votes are "better" or "same".
	 * 
	 * @param round
	 *            the round to check
	 * @return the accepted bid this round if the bid was accepted, or null if
	 *         there was no accepted bid in this round.
	 */
	private Bid getAcceptedBid(Round round) {
		for (Action action : round.getActions()) {
			if (action instanceof GiveFeedback && ((GiveFeedback) action).getFeedback() == Feedback.WORSE) {
				return null;
			}
		}
		// the round is accepted. Return the accepted bid, which should be in
		// the first turn.
		return ((OfferForFeedback) round.getTurns().get(0).getAction()).getBid();

	}

}
