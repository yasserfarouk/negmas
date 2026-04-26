package genius.core.protocol;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import genius.core.Bid;
import genius.core.Vote;
import genius.core.actions.Action;
import genius.core.actions.InformVotingResult;
import genius.core.actions.OfferForVoting;
import genius.core.actions.VoteForOfferAcceptance;
import genius.core.parties.NegotiationParty;
import genius.core.session.Round;
import genius.core.session.Session;
import genius.core.session.Turn;

/**
 * Basic implementation of a mediator based protocol.
 * <p>
 * Protocol:
 * 
 * <ol>
 * <li>Mediator proposes an {@link OfferForVoting}
 * <li>Agents {@link VoteForOfferAcceptance} to accept/reject
 * <li>Mediator sends parties a {@link InformVotingResult}
 * </ol>
 *
 * This protocol takes the last {@link InformVotingResult} that contains a
 * {@link Vote#ACCEPT} as the current agreement. If no such vote exists, it is
 * assumed no agreement has been reached yet.
 * 
 * @author David Festen
 * @author Reyhan
 */
public class SimpleMediatorBasedProtocol extends MediatorProtocol {

	@Override
	public Round getRoundStructure(List<NegotiationParty> parties, Session session) {

		// initialize and split parties
		Round round = new Round();
		NegotiationParty mediator = getMediator(parties);
		List<NegotiationParty> otherParties = getNonMediators(parties);

		// mediator opening turn
		round.addTurn(new Turn(mediator, OfferForVoting.class));

		// other parties' turn
		for (NegotiationParty otherParty : otherParties) {
			round.addTurn(new Turn(otherParty, VoteForOfferAcceptance.class));
		}

		// mediator finishing turn
		round.addTurn(new Turn(mediator, InformVotingResult.class));

		// return new round structure
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
	 * Returns the last offer for voting as the current agreement.
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
			InformVotingResult voteres = getInformVotingResult(round);
			if (voteres != null && voteres.getVotingResult() == Vote.ACCEPT) {
				return voteres.getBid();
			}
		}

		return null;
	}

	/********************* support functions ************************/
	/**
	 * Find the first {@link InformVotingResult} action in this round.
	 * 
	 * @param round
	 *            the round to check
	 * @return first {@link InformVotingResult} of round, or null if no such
	 *         action.
	 */
	private InformVotingResult getInformVotingResult(Round round) {
		for (Action action : round.getActions()) {
			if (action instanceof InformVotingResult) {
				return (InformVotingResult) action;
			}
		}
		return null;
	}

}
