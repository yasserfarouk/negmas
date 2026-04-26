package genius.core.protocol;

import java.util.List;

import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.parties.NegotiationParty;
import genius.core.session.Round;
import genius.core.session.Session;
import genius.core.session.Turn;

/**
 * Implementation of an alternating offer protocol using majority voting. This
 * is essentially equal to the {@link AlternatingMultipleOffersProtocol} but now
 * an offer is accepted if the majority (instead of all) accept.
 * <p/>
 * Protocol:
 * 
 * <pre>
 * Round 1 (offers): Each agent makes an offer
 * Round 2 (voting): Each agent votes for each offer on the table
 * 
 * The offer that is supported by the most parties, will stay on the table.
 * If a new offer has more supporting parties, it overwrites the old offer.
 * This protocol always has some agreement.
 * 
 * When deadline reached, the most recent agreement will be considered the final agreement.
 * </pre>
 *
 * @author David Festen
 * @author Reyhan
 * @author W.Pasman removed all code except acceptedOffer and extend existing
 *         AMOP
 */
public class AlternatingMajorityConsensusProtocol extends AlternatingMultipleOffersProtocol {

	/**
	 * Holds the number of parties that voted for the most recently accepted
	 * offer.
	 */
	private int mostRecentlyAcceptedOfferVoteCount;
	/**
	 * Holds the most recently accepted offer. i.e. The offer with the most
	 * support
	 */
	private Offer mostRecentlyAcceptedOffer;

	@Override
	public boolean isFinished(Session session, List<NegotiationParty> parties) {
		if (isVotingRound(session)) {
			Round votingRound = session.getMostRecentRound();
			Round offerRound = session.getRounds().get(session.getRoundNumber() - 2);
			acceptedOffer(votingRound, offerRound);
		}

		// if everyone accepts a vote, we're done, otherwise continue
		return mostRecentlyAcceptedOfferVoteCount == parties.size();
	}

	/**
	 * returns the first offer with more support than the current one, or null
	 * if no such offer.
	 *
	 * @param votingRound
	 *            the round with the voting (expected number of turns is agent#
	 *            * agent#)
	 * @param offerRound
	 *            the round with the offers (expected number of turns is agent#)
	 * @return The best offer on the table
	 */
	protected Offer acceptedOffer(Round votingRound, Round offerRound) {
		allActionsAreOffers(offerRound);
		int numOffers = offerRound.getTurns().size();
		List<Turn> turns = votingRound.getTurns();
		List<Action> voteActions = offerRound.getActions();

		int availableOfferRounds = Math.min(numOffers, numOffers > 0 ? turns.size() / numOffers : 0);

		for (int offerNumber = 0; offerNumber < availableOfferRounds; offerNumber++) {
			// update the maxNumberOfVotes we got
			int votes = nrOfVotes(numOffers, turns, offerNumber);
			// if enough votes, accept bid
			if (votes > mostRecentlyAcceptedOfferVoteCount) {
				mostRecentlyAcceptedOffer = (Offer) voteActions.get(offerNumber);
				mostRecentlyAcceptedOfferVoteCount = votes;
				System.out.println("    New most recently accepted bid (votes=" + mostRecentlyAcceptedOfferVoteCount
						+ "/" + numOffers + "): " + mostRecentlyAcceptedOffer);
			}
		}

		return mostRecentlyAcceptedOffer;
	}

}
