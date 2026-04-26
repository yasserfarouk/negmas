package genius.core.session;

import java.util.ArrayList;
import java.util.List;

import genius.core.actions.Action;

/**
 * Represents a single round in a negotiation session. A round is a part of a
 * negotiation where all participants can respond to the current state of the
 * negotiation. A {@link Round} consists of {@link Turn} which may or may not
 * have an {@link Action}. Rounds are contained in the {@link Session} object. A
 * party can have multiple turns in a single round.
 *
 * @author David Festen
 */
public class Round {
	/**
	 * holds the {@link Turn} objects of this round
	 */
	private List<Turn> turns;

	/**
	 * Creates a new instance of the {@link Round} object.
	 */
	public Round() {
		turns = new ArrayList<Turn>();
	}

	/**
	 * Creates a new instance of the {@link Round} object. This version of the
	 * constructor creates a shallow copy of the turns.
	 *
	 * @param round
	 *            An existing round object.
	 */
	public Round(Round round) {
		turns = round.getTurns();
	}

	/**
	 * Gets the turns in this round. See the {@link Turn} object for more
	 * information.
	 *
	 * @return copy of the {@link Turn} objects in this round
	 */
	public synchronized List<Turn> getTurns() {
		return new ArrayList<Turn>(turns);
	}

	/**
	 * Gets the actions in done in this round. If a turn is not executed, it
	 * shouldn't have an action. This means that in practice, you can use this
	 * method if you want to know the executed actions of this turn, even while
	 * it is still busy.
	 *
	 * @return A list of all actions done this turn.
	 */
	public synchronized List<Action> getActions() {
		List<Action> actions = new ArrayList<Action>(turns.size());
		for (Turn turn : turns)
			if (turn.getAction() != null)
				actions.add(turn.getAction());

		// return the actions
		return actions;
	}

	/**
	 * Add a turn to this round. See {@link Turn} for more information.
	 *
	 * @param turn
	 *            the turn to add.
	 */
	public synchronized void addTurn(Turn turn) {
		turns.add(turn);
	}

	/**
	 * get the last item of the {@link #getActions()} list, which in practice
	 * should be the most recent action of this round.
	 *
	 * @return The most recently executed action in this round. Null if no
	 *         action has been executed yet.
	 */
	public synchronized Action getMostRecentAction() {
		List<Action> actions = getActions();
		if (actions.isEmpty())
			return null;
		return actions.get(actions.size() - 1);
	}
}
