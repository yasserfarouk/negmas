package genius.core.session;

import static java.lang.Math.pow;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import genius.core.Deadline;
import genius.core.actions.Action;
import genius.core.parties.SessionsInfo;
import genius.core.protocol.MultilateralProtocol;
import genius.core.timeline.ContinuousTimeline;
import genius.core.timeline.DiscreteTimeline;
import genius.core.timeline.Timeline;

/**
 * Stores runtime info around a session. Mainly the rounds info. A
 * {@link Session} consists of {@link Round}s with in turn consists of
 * {@link Turn}. From this session object some information about the current
 * negotiation can be extracted. Important is the
 * {@link Session#startNewRound(Round)} method which is used to add a new round
 * to the session. At this moment (05-08-2014) adding new rounds to the session
 * is the responsibility of the {@link SessionManager}.
 *
 * @author David Festen
 */
public class Session {
	/**
	 * Holds the round objects of which this instance consists
	 */
	private ArrayList<Round> rounds;

	/**
	 * Holds the deadline constraints
	 */
	private Deadline deadlines;

	/**
	 * Holds a timestamp of the time started (used for runtime calculations)
	 * startTime should be updated using {@code System.nanoTime()}
	 */
	private long startTime;

	/**
	 * Holds a timestamp of the time started (used for runtime calculations)
	 * stopTime should be updated using {@code System.nanoTime()}
	 */
	private long stopTime;

	/**
	 * Holds a value to indicate whether the timer is running
	 */
	private boolean timerRunning;

	private Timeline timeline;

	private SessionsInfo sessionsInfo;

	/**
	 * Create a new instance of the session object. This should normally only
	 * happen once every session. See also the class documentation of
	 * {@link Session}. This also creates a timeline that starts running the
	 * moment Session is created.
	 *
	 * @param deadlines
	 *            Map of deadline constraints
	 * @param info
	 *            the global {@link SessionsInfo}
	 */
	public Session(Deadline deadlines, SessionsInfo info) {
		this.rounds = new ArrayList<Round>();
		this.deadlines = deadlines;
		this.sessionsInfo = info;
		switch (deadlines.getType()) {
		case ROUND:
			this.timeline = new DiscreteTimeline(deadlines.getValue());
			break;
		case TIME:
			this.timeline = new ContinuousTimeline(deadlines.getValue());
			break;
		}
	}

	/**
	 * Gets the deadline constraints
	 *
	 * @return a map of deadline types and values
	 */
	public Deadline getDeadlines() {
		return deadlines;
	}

	/**
	 * Updates the timestamp of this {@link Session}. Use just before starting
	 * the negotiation for most accurate timing. Timing is used in for example
	 * time deadline constraints. But might also be used in log messages as well
	 * as statistics.
	 */
	public void startTimer() {
		startTime = System.nanoTime();
		timerRunning = true;
		if (timeline instanceof ContinuousTimeline) {
			((ContinuousTimeline) timeline).reset();
		}
	}

	/**
	 * Updates the timestamp of this {@link Session}. Use just after finish the
	 * negotiation for most accurate timing. Timing is used in for example time
	 * deadline constraints. But might also be used in log messages as well as
	 * statistics. If you need to manually set it, consider using the
	 * {@link #setRuntimeInNanoSeconds(long)} function.
	 */
	public void stopTimer() {
		stopTime = System.nanoTime();
		timerRunning = false;
	}

	/**
	 * Gets the rounds currently in this session. When
	 * {@link #startNewRound(Round)} is called a new round will be added. Each
	 * round already includes all its {@link Turn}s, but some turns might not
	 * yet been done.
	 *
	 * @return list of rounds. Can be empty.
	 */
	public List<Round> getRounds() {
		return Collections.unmodifiableList(rounds);
	}

	/**
	 * Get the most recent round.
	 *
	 * @return The last round of the {@link #getRounds()} method, or null if no
	 *         round has been done yet.
	 */
	public Round getMostRecentRound() {
		if (rounds.isEmpty()) {
			return null;
		}
		return rounds.get(rounds.size() - 1);
	}

	/**
	 * Add a round to this session. Make sure it contains all the turns
	 * necessary to execute the rounds. Normally the new round will be created
	 * by using
	 * {@link MultilateralProtocol#getRoundStructure(java.util.List, Session)}
	 *
	 *
	 * @param round
	 *            The round to add to this session.
	 */
	public void startNewRound(Round round) {
		rounds.add(round);
	}

	/**
	 * Get the current round number. one-based (meaning first round is round 1)
	 *
	 * @return Integer representing the round number
	 */
	public int getRoundNumber() {
		return rounds.size();
	}

	/**
	 * Get the current turn number within the current round.
	 * 
	 * @return current turn within the round. 0 if no actions have been done yet
	 *         or if we are not even in a round.
	 */
	public int getTurnNumber() {
		if (rounds.isEmpty()) {
			return 0;
		}
		return getMostRecentRound().getActions().size();
	}

	/**
	 * Check whether this is the first round (round 1).
	 *
	 * @return true if {@link #getRoundNumber()} equals 1
	 */
	public boolean isFirstRound() {
		return getRoundNumber() == 1;
	}

	/**
	 * Removes the last (supposedly incomplete) round, if there is a last round
	 */
	public void removeLastRound() {
		if (!rounds.isEmpty()) {
			rounds.remove(rounds.size() - 1);
		}
	}

	/**
	 * Get the most recently executed action.
	 *
	 * @return The most recent action of the most recent round. Null if no
	 *         action has been done yet.
	 */
	public Action getMostRecentAction() {
		Round lastround = getMostRecentRound();
		if (lastround == null)
			return null;
		return getMostRecentRound().getMostRecentAction();
	}

	/**
	 * Check whether one of the deadlines is reached.
	 *
	 * @return true iff the deadline is reached. True if the runtime/round
	 *         number is bigger than the actual deadline.
	 */
	public boolean isDeadlineReached() {
		boolean deadlineReached = false;

		switch (deadlines.getType()) {
		case TIME:
			int timeDeadlineInSeconds = deadlines.getValue();
			double timeRanInSeconds = getRuntimeInSeconds();
			deadlineReached |= timeRanInSeconds > timeDeadlineInSeconds;
			break;
		case ROUND:
			int roundsDeadline = (Integer) deadlines.getValue();
			deadlineReached |= getRoundNumber() > roundsDeadline;
			break;
		}

		return deadlineReached;
	}

	public long getRuntimeInNanoSeconds() {
		if (timerRunning)
			return System.nanoTime() - startTime;
		else
			return stopTime - startTime;
	}

	public double getRuntimeInSeconds() {
		return (double) getRuntimeInNanoSeconds() / pow(10, 9); // ns -> s ( /
																// 10^9 )
	}

	public void setRuntimeInNanoSeconds(long nanoSeconds) {
		stopTime = System.nanoTime();
		startTime = stopTime - nanoSeconds;
	}

	public void setRuntimeInSeconds(double seconds) {
		setRuntimeInNanoSeconds(Math.round(seconds * pow(10, 9)));
	}

	public boolean isTimerRunning() {
		return timerRunning;
	}

	public Timeline getTimeline() {
		return timeline;
	}

	public void setTimeline(Timeline timeline) {
		this.timeline = timeline;
	}

	/**
	 * @return SessionsInfo containing all information that is fixed for all
	 *         sessions in the tournament/negotiation
	 */
	public SessionsInfo getInfo() {
		return sessionsInfo;
	}

}
