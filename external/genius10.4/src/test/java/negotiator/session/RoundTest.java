package negotiator.session;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import org.junit.Before;
import org.junit.Test;

import genius.core.actions.Action;
import genius.core.session.Round;
import genius.core.session.Turn;

/**
 * Junit test for {@link Round}
 * 
 * @author W.Pasman
 *
 */
public class RoundTest {
	private Round round;

	@Before
	public void before() {
		round = new Round();
	}

	@Test
	public void testConstructor() {
		assertTrue("new round contains turns", round.getTurns().isEmpty());
		assertTrue("new round contains actions", round.getActions().isEmpty());
	}

	@Test
	public void testMostRecentActionWhenEmpty() {
		assertNull(round.getMostRecentAction());
	}

	@Test
	public void testAddTurn() {
		Turn turn = mock(Turn.class);
		Action action1 = mock(Action.class);
		when(turn.getAction()).thenReturn(action1);
		round.addTurn(turn);
		assertEquals(1, round.getTurns().size());
		assertEquals(1, round.getActions().size());
		assertEquals(action1, round.getActions().get(0));
	}

	@Test
	public void testgetAction() {
		Turn turn = mock(Turn.class);
		Action action1 = mock(Action.class);
		when(turn.getAction()).thenReturn(action1);
		round.addTurn(turn);
		assertEquals(1, round.getTurns().size());
		assertEquals(1, round.getActions().size());
		assertEquals(action1, round.getActions().get(0));
	}

	@Test
	public void testIterateOverRoundsWhileUpdating() {
		Turn turn = mock(Turn.class);
		Action action1 = mock(Action.class);
		when(turn.getAction()).thenReturn(action1);
		round.addTurn(turn);

		for (Turn t : round.getTurns()) {
			round.addTurn(mock(Turn.class));
		}

	}

	@Test
	public void testgetActionWithNull() {
		Turn turn = mock(Turn.class);
		when(turn.getAction()).thenReturn(null);
		round.addTurn(turn);
		assertEquals(1, round.getTurns().size());
		assertEquals(0, round.getActions().size());
	}

	@Test
	public void testGetLastAction() {
		Turn turn = mock(Turn.class);
		Action action1 = mock(Action.class);
		when(turn.getAction()).thenReturn(action1);
		round.addTurn(turn);
		Action lastAction = round.getMostRecentAction();
		assertEquals(action1, lastAction);
	}

}
