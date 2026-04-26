package genius.core.protocol;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.actions.OfferForVoting;
import genius.core.actions.Reject;
import genius.core.parties.NegotiationParty;
import genius.core.session.Round;
import genius.core.session.Session;
import genius.core.session.Turn;

/**
 * Implementation of an alternating offer protocol using voting consensus.
 * <p/>
 * Protocol in short:
 * 
 * <pre>
 * Round 1: Each agent makes their own offer.
 * Round 2: Each agent votes (accept/reject) for each offer on the table.
 * 
 * If there is one offer that everyone accepts, the negotiation ends with this offer.
 * Otherwise, the process continues until reaching deadline or agreement.
 * </pre>
 *
 * <h1>Detailed description</h1>
 * <p>
 *
 * 
 * The AMOP protocol is an alternating offers protocol in which all players get
 * the same opportunities. That is, every bid that is made in a round is
 * available to all agents before they vote on these bids. This implemented in
 * the following way: The AMOP protocol has a bidding phase followed by voting
 * phases. In the bidding phase all negotiators put their offer on the table. In
 * the voting phases all participants vote on all of the bids on the negotiation
 * table. If one of the bids on the negotiation table is accepted by all of the
 * parties, then the negotiation ends with this bid. This is an iterative
 * process continuing until reaching an agreement or reaching the deadline. The
 * essential difference with the SAOP protocol is that the players do not
 * override the offers made by others and the agents can take all offers into
 * account before they vote on the proposals.
 * </p>
 * 
 * @author David Festen
 * @author Reyhan Aydogan
 * @author Catholijn Jonker
 * @author W.Pasman modification to improve testability
 */
public class AlternatingMultipleOffersProtocol extends DefaultMultilateralProtocol {
	// keeps track of max number of accepts that an offer got.
	// this value never resets. It should not be here.
	private int maxNumberOfVotes = 0;

	/**
	 * Get the round structure used by this algorithm.
	 * <p/>
	 * Structure:
	 * 
	 * <pre>
	 * Round 1: Each agent makes their own offer.
	 * Round 2: Each agent votes (accept/reject) for each offer on the table.
	 * </pre>
	 *
	 * @param parties
	 *            The parties currently participating
	 * @param session
	 *            The complete session history
	 * @return A list of possible actions
	 */
	@Override
	public Round getRoundStructure(List<NegotiationParty> parties, Session session) {
		Round round = createRound();

		// NOTE: while roundnumber is normally one-based, in this function it's
		// zero based as you are initializing the
		// new round right in this function
		if (session.getRoundNumber() % 2 == 0) {
			// request an offer from each party
			for (NegotiationParty party : parties) {
				round.addTurn(createTurn(party, OfferForVoting.class));
			}
		} else {
			ArrayList<Class<? extends Action>> acceptOrReject = new ArrayList<Class<? extends Action>>(2);
			acceptOrReject.add(Accept.class);
			acceptOrReject.add(Reject.class);

			// request a reaction on each offer from party
			for (NegotiationParty votedOnParty : parties) {
				for (NegotiationParty votingParty : parties) {
					round.addTurn(createTurn(votingParty, acceptOrReject));
				}
			}
		}
		return round;
	}

	@Override
	public boolean isFinished(Session session, List<NegotiationParty> parties) {
		// if we are making new offers, we are never finished
		if (session.getRoundNumber() < 2 || !isVotingRound(session)) {
			return false;
		}

		// find an acceptable offer
		Round thisRound = session.getMostRecentRound();
		Round prevRound = session.getRounds().get(session.getRoundNumber() - 2);
		Offer acceptedOffer = acceptedOffer(thisRound, prevRound);

		// if null, we are not finished, otherwise we are
		return acceptedOffer != null;
	}

	/**
	 * Gets the current agreement if any. This assumes that the session contains
	 * {@link Round}s containing offer, votes, offer, votes, etc in this order.
	 * An agreement consists an {@link Offer} that was {@link Accept}ed by all
	 * the votes. See also {@link #acceptedOffer(Round, Round)}.
	 *
	 * @param session
	 *            The complete session history up to this point
	 * @return The agreement bid or null if none
	 */
	@Override
	public Bid getCurrentAgreement(Session session, List<NegotiationParty> parties) {
		int round = session.getRoundNumber();
		if (round % 2 == 1 || round < 2) {
			return null;
		}
		Round thisRound = session.getMostRecentRound();
		Round prevRound = session.getRounds().get(session.getRoundNumber() - 2);
		Offer acceptedOffer = acceptedOffer(thisRound, prevRound);
		return acceptedOffer == null ? null : acceptedOffer.getBid();
	}

	/**
	 * returns the first offer in the given {@link Round} that everyone
	 * accepted, or null if no such offer.
	 *
	 * @param votingRound
	 *            the round with the voting ({@link Accept} or {@link Reject})
	 *            actions. The turns in votingRound must contain the following:
	 *            (N is the number of turns in the offer round)
	 *            <p>
	 *            vote(party1,offer1), vote(party2, offer1), ..., vote(partyN,
	 *            offer1), vote(party1, offer2), ......, vote(party1, offerN),
	 *            ... , vote(partyN, offerN)
	 *            </p>
	 *            We only consider offers that ALL N parties have voted on.
	 * @param offerRound
	 *            the round with the offers (one for each party is expected).
	 * @return The first accepted offer (all parties accepted the offer) if such
	 *         an offer exists, null otherwise.
	 * @throws IllegalArgumentException
	 *             if the offerRound contains {@link Action}(s) not extending
	 *             {@link Offer}
	 */
	protected Offer acceptedOffer(Round votingRound, Round offerRound) {
		allActionsAreOffers(offerRound);
		int numOffers = offerRound.getActions().size();
		if (numOffers == 0) {
			return null;
		}
		List<Turn> turns = votingRound.getTurns();
		List<Action> voteActions = offerRound.getActions();

		int availableOfferRounds = Math.min(numOffers, turns.size() / numOffers);

		for (int offerNumber = 0; offerNumber < availableOfferRounds; offerNumber++) {
			// update the maxNumberOfVotes we got
			maxNumberOfVotes = Math.max(maxNumberOfVotes, nrOfVotes(numOffers, turns, offerNumber));

			// if enough votes, accept bid
			if (maxNumberOfVotes == numOffers) {
				return (Offer) voteActions.get(offerNumber);
			}

		}

		return null;
	}

	/**
	 * 
	 * @param numOffers
	 *            the number of offers on the table (in each turn)
	 * @param turns
	 *            all the voting turns
	 * @param offerNumber
	 *            the offer number that is being checked.
	 * @return number of {@link Accept}s for given offer number
	 */
	protected int nrOfVotes(int numOffers, List<Turn> turns, int offerNumber) {
		int votes = 0;
		// count number of votes
		for (int voteNr = 0; voteNr < numOffers; voteNr++) {
			// count the vote
			if (turns.get(offerNumber * numOffers + voteNr).getAction() instanceof Accept) {
				votes++;
			}
		}
		return votes;
	}

	/**
	 * Checks if all actions are offers.
	 * 
	 * @throws IllegalArgumentException
	 *             if not.
	 * @param offerRound
	 */
	protected void allActionsAreOffers(Round offerRound) {
		for (Action action : offerRound.getActions()) {
			if (!(action instanceof Offer)) {
				throw new IllegalArgumentException(
						"encountered an action " + action + " in the offer round, which is not an Offer");
			}
		}
	}

	/**
	 * Returns whether this is a voting round. First voting round is even round
	 * and >= 2.
	 *
	 * @param session
	 *            the current state of this session
	 * @return true is this is an even round > 0.
	 */
	protected boolean isVotingRound(Session session) {
		return session.getRoundNumber() > 0 && session.getRoundNumber() % 2 == 0;
	}

	/**
	 * Gets the maximum number of vote this protocol found.
	 *
	 * @param session
	 *            the current state of this session
	 * @param parties
	 *            The parties currently participating
	 * @return the number of parties agreeing to the current agreement
	 */
	@Override
	public int getNumberOfAgreeingParties(Session session, List<NegotiationParty> parties) {
		return maxNumberOfVotes;
	}

	@Override
	public Map<NegotiationParty, List<NegotiationParty>> getActionListeners(List<NegotiationParty> parties) {
		return listenToAll(parties);
	}

	/******************************************/

	/**
	 * factory function. To support testing.
	 * 
	 * @param votingParty
	 * @param allowedActions
	 *            list of allowed action classes
	 * @return a new {@link Turn} with given actions as possible actions.
	 */
	public Turn createTurn(NegotiationParty votingParty, Collection<Class<? extends Action>> allowedActions) {
		return new Turn(votingParty, allowedActions);
	}

	/**
	 * create factory function. To support testing.
	 * 
	 * @param party
	 * @param allowedAction
	 *            the class of action that is possible.
	 * @return a new {@link Turn} with given action as possible actions.
	 */
	public Turn createTurn(NegotiationParty party, Class<? extends Action> allowedAction) {
		return new Turn(party, allowedAction);
	}

	/**
	 * factory function. To support testing.
	 * 
	 * @return round
	 */
	public Round createRound() {
		return new Round();
	}

}
