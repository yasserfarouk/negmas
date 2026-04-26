package negotiator.session;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import org.junit.Before;
import org.junit.Test;

import genius.core.Deadline;
import genius.core.DeadlineType;
import genius.core.session.Round;
import genius.core.session.Session;

/**
 * Tests the {@link Session} object. FIXME we currently ignore time related
 * calls as that part of the code needs to be cleaned up first.
 * 
 * @author W.Pasman
 *
 */
public class SessionTest {
	private Session session;
	private Deadline deadline = mock(Deadline.class);

	@Before
	public void before() {
		when(deadline.getType()).thenReturn(DeadlineType.ROUND);
		session = new Session(deadline, null);
	}

	@Test
	public void testGetDeadline() {
		assertEquals(deadline, session.getDeadlines());
	}

	@Test
	public void testGetLastRoundInitially() {
		assertNull(session.getMostRecentRound());
	}

	@Test
	public void testGetRoundNumberInitially() {
		assertEquals(0, session.getRoundNumber());
	}

	@Test
	public void testIsFirstRoundInitially() {
		assertFalse(session.isFirstRound());
	}

	@Test
	public void testGetRoundsInitially() {
		assertTrue(session.getRounds().isEmpty());
	}

	@Test
	public void testGetTurnNumberInitially() {
		assertEquals(0, session.getTurnNumber());
	}

	@Test
	public void testMostRecentActionInitially() {
		assertNull(session.getMostRecentAction());
	}

	@Test
	public void testIsDeadlineReachedInitially() {
		when(deadline.getValue()).thenReturn(10);
		assertFalse(session.isDeadlineReached());
	}

	@Test
	public void testGetRuntimeNanoInitially() {
		assertEquals(0, session.getRuntimeInNanoSeconds());
	}

	@Test
	public void testGetRuntimeInitially() {
		assertEquals(0, session.getRuntimeInSeconds(), 0.000000001);
	}

	@Test
	public void testGetLastRoundAfter1Round() {
		Round firstRound = mock(Round.class);
		session.startNewRound(firstRound);
		assertEquals(firstRound, session.getMostRecentRound());
	}

	@Test
	public void testIsDeadlineReachedAfter1Round() {
		Round firstRound = mock(Round.class);
		session.startNewRound(firstRound);
		when(deadline.getValue()).thenReturn(0);
		assertTrue(session.isDeadlineReached());
	}

	@Test
	public void testIsDeadlineReachedAfter1RoundB() {
		Round firstRound = mock(Round.class);
		session.startNewRound(firstRound);
		when(deadline.getValue()).thenReturn(10);
		assertFalse(session.isDeadlineReached());
	}

	@Test(expected = UnsupportedOperationException.class)
	public void testIsGetRoundNonmodifyable() {
		Round firstRound = mock(Round.class);
		session.startNewRound(firstRound);
		// this SHOULD throw unsipportedoperation
		session.getRounds().add(mock(Round.class));
	}

}
