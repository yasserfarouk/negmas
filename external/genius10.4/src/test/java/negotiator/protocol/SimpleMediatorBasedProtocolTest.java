package negotiator.protocol;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.Vote;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.InformVotingResult;
import genius.core.actions.OfferForVoting;
import genius.core.actions.Reject;
import genius.core.actions.VoteForOfferAcceptance;
import genius.core.parties.Mediator;
import genius.core.parties.NegotiationParty;
import genius.core.protocol.MediatorProtocol;
import genius.core.protocol.SimpleMediatorBasedProtocol;
import genius.core.session.Round;
import genius.core.session.Session;
import genius.core.session.Turn;

/**
 * Tests for the {@link SimpleMediatorBasedProtocol}.
 * 
 * @author W.Pasman
 *
 */
public class SimpleMediatorBasedProtocolTest {

	protected MediatorProtocol protocol;
	private static AgentID AGENTID = new AgentID("test");
	private static Bid receivedBid = mock(Bid.class);

	protected static Action ACCEPT = new Accept(AGENTID, receivedBid);
	protected static Action REJECT = new Reject(AGENTID, receivedBid);

	Session session = mock(Session.class);
	protected ArrayList<NegotiationParty> parties;
	private NegotiationParty party1;
	private NegotiationParty party2;
	private NegotiationParty party3;
	private NegotiationParty mediator;
	private List<Round> rounds;
	private InformVotingResult informVoteAccept;
	private Bid acceptedBid = mock(Bid.class);
	private InformVotingResult informVoteReject;
	private Bid rejectedBid = mock(Bid.class);
	private OfferForVoting partyVote;

	@Before
	public void init() {

		protocol = mock(getProtocol(), Mockito.CALLS_REAL_METHODS);

		parties = new ArrayList<NegotiationParty>();
		party1 = mock(NegotiationParty.class);
		party2 = mock(NegotiationParty.class);
		party3 = mock(NegotiationParty.class);
		mediator = mock(Mediator.class);

		parties.add(party1);
		parties.add(party2);
		parties.add(party3);
		parties.add(mediator);

		rounds = new ArrayList<Round>();
		when(session.getRounds()).thenReturn(rounds);

		informVoteAccept = mock(InformVotingResult.class);
		when(informVoteAccept.getVotingResult()).thenReturn(Vote.ACCEPT);
		when(informVoteAccept.getBid()).thenReturn(acceptedBid);

		informVoteReject = mock(InformVotingResult.class);
		when(informVoteReject.getVotingResult()).thenReturn(Vote.REJECT);
		when(informVoteReject.getBid()).thenReturn(rejectedBid);

		partyVote = mock(OfferForVoting.class);
	}

	protected Class<? extends MediatorProtocol> getProtocol() {
		return SimpleMediatorBasedProtocol.class;
	}

	@Test
	public void testGetMediator() {
		// strictly not a function of protocol, but still kind of accessable
		// through it.
		assertEquals(mediator, protocol.getMediator(parties));
	}

	@Test
	public void testGetNonMediators() {
		// strictly not a function of protocol, but still kind of accessable
		// through it.
		List<NegotiationParty> normalParties = protocol.getNonMediators(parties);
		assertTrue(normalParties.contains(party1));
		assertTrue(normalParties.contains(party2));
		assertTrue(normalParties.contains(party3));
	}

	@Test
	public void testGetRoundStructure() {
		Collection<Class<? extends Action>> acceptOrReject = new ArrayList<Class<? extends Action>>(2);
		acceptOrReject.add(Accept.class);
		acceptOrReject.add(Reject.class);

		Round round = protocol.getRoundStructure(parties, session);

		// check that round contains 5 turns (one offer, 3 votes, one inform)
		// and that they are associated correctly to our parties.
		List<Turn> turns = round.getTurns();
		assertEquals(5, turns.size());
		checkTurn(turns.get(0), mediator, OfferForVoting.class);
		checkTurn(turns.get(1), party1, VoteForOfferAcceptance.class);
		checkTurn(turns.get(2), party2, VoteForOfferAcceptance.class);
		checkTurn(turns.get(3), party3, VoteForOfferAcceptance.class);
		checkTurn(turns.get(4), mediator, InformVotingResult.class);
	}

	/**
	 * check that the mediator can hear all other parties
	 */
	@Test
	public void testMediatorHearsAll() {
		List<NegotiationParty> mediatorListensTo = protocol.getActionListeners(parties).get(mediator);

		for (NegotiationParty party : protocol.getNonMediators(parties)) {
			assertTrue("Mediator is not listening to " + party, mediatorListensTo.contains(party));
		}

	}

	/**
	 * check that parties can hear only the mediator.
	 */
	@Test
	public void testPartiesHearOnlyMediator() {
		for (NegotiationParty party : protocol.getNonMediators(parties)) {
			List<NegotiationParty> partyListensTo = protocol.getActionListeners(parties).get(party);
			assertEquals("Party listens to more than just the mediator:" + partyListensTo, 1, partyListensTo.size());
			assertTrue("Party is not listening to mediator", partyListensTo.contains(mediator));
		}

	}

	/**
	 * Call isFinished when in round 0 (initial situation). The round should not
	 * be finished, nothing happened yet.
	 */
	@Test
	public void isFinishedTestVoting() {

		assertFalse(protocol.isFinished(session, parties));
		assertNull(protocol.getCurrentAgreement(session, parties));
	}

	/**
	 * Call isFinished when in round 1. But there is no InformVotingResult in
	 * that round.
	 */
	@Test
	public void isFinishedRound1() {
		Round round1 = mock(Round.class);
		rounds.add(round1);

		assertFalse(protocol.isFinished(session, parties));
		assertNull(protocol.getCurrentAgreement(session, parties));
	}

	/**
	 * Check agreement with a accept and a reject. Always check we're not
	 * finished.
	 */
	@Test
	public void isFinishedWithOneAccept() {
		addRoundWithActions(informVoteAccept);

		assertEquals(acceptedBid, protocol.getCurrentAgreement(session, parties));
		// check that even when we have agreement, the protocol continues
		assertFalse(protocol.isFinished(session, parties));
	}

	/**
	 * Check agreement with a accept and a rejectt.
	 */
	@Test
	public void isFinishedWithAcceptReject() {
		addRoundWithActions(informVoteAccept);
		addRoundWithActions(informVoteReject);

		assertEquals(acceptedBid, protocol.getCurrentAgreement(session, parties));
		assertFalse(protocol.isFinished(session, parties));
	}

	/**
	 * Check agreement with a reject and an accept.
	 */
	@Test
	public void isFinishedWithRejectAccept() {
		addRoundWithActions(informVoteReject);
		addRoundWithActions(informVoteAccept);

		assertEquals(acceptedBid, protocol.getCurrentAgreement(session, parties));
		assertFalse(protocol.isFinished(session, parties));
	}

	/**
	 * Check agreement with a reject and an accept that are cluttered with party
	 * votes.
	 */
	@Test
	public void isFinishedWithRejectAcceptAndPartyVotes() {
		addRoundWithActions(partyVote, partyVote, partyVote, informVoteReject, partyVote, partyVote, partyVote);
		addRoundWithActions(partyVote, partyVote, partyVote, partyVote, partyVote, informVoteAccept, partyVote);

		assertEquals(acceptedBid, protocol.getCurrentAgreement(session, parties));
		assertFalse(protocol.isFinished(session, parties));
	}

	/**
	 * Check reject with two rejects
	 */
	@Test
	public void isFinishedWithTwoRejects() {
		addRoundWithActions(informVoteReject);
		addRoundWithActions(informVoteReject);

		assertNull(protocol.getCurrentAgreement(session, parties));
		assertFalse(protocol.isFinished(session, parties));
	}

	/**
	 * Check that the last accept is the agreement, within a set also containing
	 * rejects
	 */
	@Test
	public void isLastAcceptTheAgreement() {
		Bid acceptedBid2 = mock(Bid.class);
		InformVotingResult informVoteAccept2 = mock(InformVotingResult.class);
		when(informVoteAccept2.getVotingResult()).thenReturn(Vote.ACCEPT);
		when(informVoteAccept2.getBid()).thenReturn(acceptedBid2);

		addRoundWithActions(informVoteReject);
		addRoundWithActions(informVoteReject);
		addRoundWithActions(informVoteAccept);
		addRoundWithActions(informVoteReject);
		addRoundWithActions(informVoteReject);
		addRoundWithActions(informVoteReject);
		addRoundWithActions(informVoteReject);
		addRoundWithActions(informVoteAccept2);
		addRoundWithActions(informVoteReject);
		addRoundWithActions(informVoteReject);

		assertEquals(acceptedBid2, protocol.getCurrentAgreement(session, parties));
		assertFalse(protocol.isFinished(session, parties));
	}

	/******************** Support functions ********************/
	/**
	 * Checks that given turn comes from given party and contains given class.
	 * 
	 * @param turn
	 *            the turn to check
	 * @param party
	 *            the {@link NegotiationParty} that should be in this turn
	 * @param actionclass
	 *            the action type that should be in this turn
	 */
	private void checkTurn(Turn turn, NegotiationParty party, Class<? extends Action> actionclass) {
		assertTrue("Turn " + turn + " does not contain " + actionclass, turn.getValidActions().contains(actionclass));
		assertEquals(party, turn.getParty());
	}

	/**
	 * Mock the next round with some actions.
	 * 
	 * @param newturns
	 *            a list of new turns for the next round.
	 */
	private void addRoundWithActions(Action... newturns) {
		Round round = mock(Round.class);
		List<Turn> turns = new ArrayList<Turn>();
		List<Action> actions = new ArrayList<Action>();

		for (Action action : newturns) {
			actions.add(action);
			Turn turn = mock(Turn.class);
			when(turn.getAction()).thenReturn(action);
			turns.add(turn);
		}
		when(round.getTurns()).thenReturn(turns);
		when(round.getActions()).thenReturn(actions);

		rounds.add(round);
	}

}
