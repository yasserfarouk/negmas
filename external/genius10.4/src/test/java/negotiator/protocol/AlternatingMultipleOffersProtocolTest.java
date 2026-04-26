package negotiator.protocol;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.Matchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.Domain;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.actions.OfferForVoting;
import genius.core.actions.Reject;
import genius.core.parties.NegotiationParty;
import genius.core.protocol.AlternatingMultipleOffersProtocol;
import genius.core.session.Round;
import genius.core.session.Session;
import genius.core.session.Turn;

public class AlternatingMultipleOffersProtocolTest {

	protected AlternatingMultipleOffersProtocol protocol;
	private static AgentID AGENTID = new AgentID("test");
	private static Bid receivedBid = mock(Bid.class);

	private static int VOTING_ROUND = 0;
	private static int OFFER_ROUND = 1;
	private static int VOTING_ROUND_2 = 2;

	protected static Action ACCEPT = new Accept(AGENTID, receivedBid);
	protected static Action REJECT = new Reject(AGENTID, receivedBid);

	Session session = mock(Session.class);
	protected ArrayList<NegotiationParty> parties;
	private NegotiationParty party1;
	private NegotiationParty party2;
	private NegotiationParty party3;

	@Before
	public void init() {

		protocol = mock(getProtocol(), Mockito.CALLS_REAL_METHODS);

		parties = new ArrayList<NegotiationParty>();
		party1 = mock(NegotiationParty.class);
		party2 = mock(NegotiationParty.class);
		party3 = mock(NegotiationParty.class);

		parties.add(party1);
		parties.add(party2);
		parties.add(party3);

	}

	protected Class<? extends AlternatingMultipleOffersProtocol> getProtocol() {
		return AlternatingMultipleOffersProtocol.class;
	}

	@Test
	public void testGetRoundStructureVotingRound() {
		when(session.getRoundNumber()).thenReturn(VOTING_ROUND_2);

		Round round = protocol.getRoundStructure(parties, session);

		// check that there were created for each party one turn for an offer
		verify(protocol, times(1)).createTurn(eq(party1), eq(OfferForVoting.class));
		verify(protocol, times(1)).createTurn(eq(party2), eq(OfferForVoting.class));
		verify(protocol, times(1)).createTurn(eq(party3), eq(OfferForVoting.class));

		// check that round contains 3 turns and that they are associated to our
		// parties.
		assertEquals(3, round.getTurns().size());
		assertEquals(party1, round.getTurns().get(0).getParty());
		assertEquals(party2, round.getTurns().get(1).getParty());
		assertEquals(party3, round.getTurns().get(2).getParty());

	}

	@Test
	public void testGetRoundStructureOfferRound() {
		when(session.getRoundNumber()).thenReturn(OFFER_ROUND);

		// the specs has poor type, that's why we have it here too.
		Collection<Class<? extends Action>> acceptOrReject = new ArrayList<Class<? extends Action>>(2);
		acceptOrReject.add(Accept.class);
		acceptOrReject.add(Reject.class);

		Round round = protocol.getRoundStructure(parties, session);

		// check that everyone has 3 votes, one for each proposal.
		verify(protocol, times(3)).createTurn(eq(party1), eq(acceptOrReject));
		verify(protocol, times(3)).createTurn(eq(party2), eq(acceptOrReject));
		verify(protocol, times(3)).createTurn(eq(party3), eq(acceptOrReject));

		// check that round contains 9 turns (3 parties * 3 votes) and that they
		// are associated correctly to our parties.
		assertEquals(9, round.getTurns().size());
		assertEquals(party1, round.getTurns().get(0).getParty());
		assertEquals(party2, round.getTurns().get(1).getParty());
		assertEquals(party3, round.getTurns().get(2).getParty());
		assertEquals(party1, round.getTurns().get(3).getParty());
		assertEquals(party2, round.getTurns().get(4).getParty());
		assertEquals(party3, round.getTurns().get(5).getParty());
		assertEquals(party1, round.getTurns().get(6).getParty());
		assertEquals(party2, round.getTurns().get(7).getParty());
		assertEquals(party3, round.getTurns().get(8).getParty());

	}

	/**
	 * Call isFinished when in round 0 (initial situation). The round should not
	 * be finished, nothing happened yet. FAILS because
	 * get(session.getRoundNumber() - 2) in the isFinished code will give
	 * indexOutOfBoundsException.
	 */
	@Test
	public void isFinishedTestVoting() {
		when(session.getRoundNumber()).thenReturn(VOTING_ROUND);
		when(session.getRounds()).thenReturn(new ArrayList<Round>());
		assertFalse(protocol.isFinished(session, parties));
		assertNull(protocol.getCurrentAgreement(session, parties));
	}

	/**
	 * Call isFinished when in round 1
	 */
	@Test
	public void isFinishedTestNonVoting() {
		when(session.getRoundNumber()).thenReturn(OFFER_ROUND);
		assertFalse(protocol.isFinished(session, parties));
		assertNull(protocol.getCurrentAgreement(session, parties));
	}

	/**
	 * Call isFinished when in round OFFER_ROUND+2 which is an offer round too.
	 */
	@Test
	public void isFinishedTestNonVotingRound() {
		when(session.getRoundNumber()).thenReturn(OFFER_ROUND + 2);
		assertFalse(protocol.isFinished(session, parties));
	}

	/**
	 * call isFinished when in round 2. The round should not be finished,
	 * nothing happened yet.
	 */
	@Test
	public void isFinishedTestVoting2() {
		when(session.getRoundNumber()).thenReturn(VOTING_ROUND_2);
		Round round1 = mock(Round.class);
		ArrayList<Turn> turns1 = new ArrayList<Turn>();
		for (int n = 0; n < 3; n++)
			turns1.add(mock(Turn.class));
		when(round1.getTurns()).thenReturn(turns1);

		Round round2 = mock(Round.class);
		ArrayList<Turn> turns = new ArrayList<Turn>();
		when(round2.getTurns()).thenReturn(turns);
		ArrayList<Round> rounds = new ArrayList<Round>();
		rounds.add(round2);
		when(session.getRounds()).thenReturn(rounds);
		when(session.getMostRecentRound()).thenReturn(round2);

		assertFalse(protocol.isFinished(session, parties));
	}

	@Test
	public void testIsFinishedWithIncompleteAcceptVotes() {

		// We need at least two rounds for a finish.
		List<Round> rounds = new ArrayList<Round>();

		// 3 turns is apparently insufficient. Not clear why.
		// but protocol.isFinished should not crash on that.
		Offer firstoffer = offer();
		rounds.add(mockedRound(new Action[] { firstoffer, offer(), offer() }));
		// incomplete: only votes for bid 1.
		Round voteRound = mockedRound(new Action[] { ACCEPT, ACCEPT, ACCEPT });
		rounds.add(voteRound);

		when(session.getMostRecentRound()).thenReturn(voteRound);
		when(session.getRounds()).thenReturn(rounds);

		// we have round 0 and 1 done
		when(session.getRoundNumber()).thenReturn(2);
		when(session.getRounds()).thenReturn(rounds);

		assertTrue(protocol.isFinished(session, parties));
		assertEquals(firstoffer.getBid(), protocol.getCurrentAgreement(session, parties));
	}

	@Test
	public void testIsNotFinishedWithIncompleteVotes() {

		// We need at least two rounds for a finish.
		List<Round> rounds = new ArrayList<Round>();

		// 3 turns is apparently insufficient. Not clear why.
		// but protocol.isFinished should not crash on that.
		rounds.add(mockedRound(new Action[] { offer(), offer(), offer() }));
		// incomplete: only votes for bid 1.
		Round voteRound = mockedRound(new Action[] { REJECT, ACCEPT, ACCEPT });
		rounds.add(voteRound);

		when(session.getMostRecentRound()).thenReturn(voteRound);
		when(session.getRounds()).thenReturn(rounds);

		// we have round 0 and 1 done
		when(session.getRoundNumber()).thenReturn(2);
		when(session.getRounds()).thenReturn(rounds);

		assertFalse(protocol.isFinished(session, parties));
		assertEquals(null, protocol.getCurrentAgreement(session, parties));
	}

	@Test
	public void testIsFinishedWithIncompleteAcceptVotes2() {

		// We need at least two rounds for a finish.
		List<Round> rounds = new ArrayList<Round>();

		Offer firstoffer = offer();
		rounds.add(mockedRound(new Action[] { firstoffer, offer(), offer() }));
		// incomplete: only votes for bid 1 plus 1 extra.
		Round voteRound = mockedRound(new Action[] { ACCEPT, ACCEPT, ACCEPT, ACCEPT });
		rounds.add(voteRound);

		when(session.getMostRecentRound()).thenReturn(voteRound);
		when(session.getRounds()).thenReturn(rounds);

		// we have round 0 and 1 done
		when(session.getRoundNumber()).thenReturn(2);
		when(session.getRounds()).thenReturn(rounds);

		assertTrue(protocol.isFinished(session, parties));
		assertEquals(firstoffer.getBid(), protocol.getCurrentAgreement(session, parties));
	}

	@Test
	public void testIsFinishedWithTooFewVotes() {

		// We need at least two rounds for a finish.
		List<Round> rounds = new ArrayList<Round>();

		Offer firstoffer = offer();
		rounds.add(mockedRound(new Action[] { firstoffer, offer(), offer() }));
		// incomplete: not even enough to judge the first offer.
		Round voteRound = mockedRound(new Action[] { ACCEPT, ACCEPT });
		rounds.add(voteRound);

		when(session.getMostRecentRound()).thenReturn(voteRound);
		when(session.getRounds()).thenReturn(rounds);

		// we have round 0 and 1 done
		when(session.getRoundNumber()).thenReturn(2);
		when(session.getRounds()).thenReturn(rounds);

		assertFalse(protocol.isFinished(session, parties));
		assertNull(protocol.getCurrentAgreement(session, parties));
	}

	protected Offer offer() {
		Bid bid = new Bid(mock(Domain.class));
		Offer offer = new Offer(AGENTID, bid);
		return offer;
	}

	/**
	 * isFinished expects offers round, then vote round. But should bee robust
	 * for it
	 */
	@Test(expected = IllegalArgumentException.class)
	public void testIsFinishedWithVoteRoundsOnly() {

		// We need at least two rounds for a finish.
		List<Round> rounds = new ArrayList<Round>();

		Round voteRound = mockedRound(new Action[] { ACCEPT, ACCEPT, ACCEPT });
		rounds.add(voteRound);
		rounds.add(voteRound);

		when(session.getMostRecentRound()).thenReturn(voteRound);
		when(session.getRounds()).thenReturn(rounds);

		// we have round 0 and 1 done
		when(session.getRoundNumber()).thenReturn(2);
		when(session.getRounds()).thenReturn(rounds);

		assertTrue(protocol.isFinished(session, null));
	}

	@Test
	public void testIsFinishedWithCompleteAcceptVotes() {

		// We need at least two rounds for a finish.
		List<Round> rounds = new ArrayList<Round>();

		// 3 turns is apparently insufficient. Not clear why.
		// but protocol.isFinished should not crash on that.
		Offer firstoffer = offer();
		rounds.add(mockedRound(new Action[] { firstoffer, offer(), offer() }));
		Round voteround = mockedRound(
				new Action[] { ACCEPT, ACCEPT, ACCEPT, ACCEPT, ACCEPT, ACCEPT, ACCEPT, ACCEPT, ACCEPT });
		rounds.add(voteround);

		when(session.getMostRecentRound()).thenReturn(voteround);
		when(session.getRounds()).thenReturn(rounds);

		// we have round 0 and 1 done
		when(session.getRoundNumber()).thenReturn(2);
		when(session.getRounds()).thenReturn(rounds);

		assertTrue(protocol.isFinished(session, parties));
		assertEquals(firstoffer.getBid(), protocol.getCurrentAgreement(session, parties));
	}

	@Test
	public void testAcceptSecondOffer() {

		// We need at least two rounds for a finish.
		List<Round> rounds = new ArrayList<Round>();

		// 3 turns is apparently insufficient. Not clear why.
		// but protocol.isFinished should not crash on that.
		Offer acceptedoffer = offer();
		rounds.add(mockedRound(new Action[] { offer(), acceptedoffer, offer() }));
		Round voteround = mockedRound(
				new Action[] { ACCEPT, ACCEPT, REJECT, ACCEPT, ACCEPT, ACCEPT, REJECT, ACCEPT, ACCEPT });
		rounds.add(voteround);

		when(session.getMostRecentRound()).thenReturn(voteround);
		when(session.getRounds()).thenReturn(rounds);

		// we have round 0 and 1 done
		when(session.getRoundNumber()).thenReturn(2);
		when(session.getRounds()).thenReturn(rounds);

		assertTrue(protocol.isFinished(session, parties));
		assertEquals(acceptedoffer.getBid(), protocol.getCurrentAgreement(session, parties));
	}

	@Test
	public void testIsFinishedWithIncompleteRejectVotes() {

		// We need at least two rounds for a finish.
		List<Round> rounds = new ArrayList<Round>();

		// 3 turns is apparently insufficient. Not clear why.
		// but protocol.isFinished should not crash on that.
		rounds.add(mockedRound(mock(Offer.class)));
		Round voteRound = mockedRound(new Action[] { REJECT, REJECT, REJECT });
		rounds.add(voteRound);

		when(session.getMostRecentRound()).thenReturn(voteRound);
		when(session.getRounds()).thenReturn(rounds);

		// we have round 0 and 1 done
		when(session.getRoundNumber()).thenReturn(2);
		when(session.getRounds()).thenReturn(rounds);

		assertFalse(protocol.isFinished(session, parties));
		assertNull(protocol.getCurrentAgreement(session, parties));
	}

	/**
	 * @param action
	 *            the action to return for each of the turns in the action. Must
	 *            be real action , not a mock, as the protocol compares uses
	 *            instanceof to test action types
	 * @return Mocked {@link Round}.
	 */
	protected Round mockedRound(Action... actions) {
		Round round = mock(Round.class);
		List<Action> actionslist = new ArrayList<Action>();
		List<Turn> turns = new ArrayList<Turn>();
		for (int t = 0; t < actions.length; t++) {
			Action action = actions[t];
			Turn turn = mock(Turn.class);
			when(turn.getAction()).thenReturn(action);
			turns.add(turn);
			actionslist.add(action);
		}
		when(round.getTurns()).thenReturn(turns);
		when(round.getActions()).thenReturn(actionslist);
		return round;
	}

}
