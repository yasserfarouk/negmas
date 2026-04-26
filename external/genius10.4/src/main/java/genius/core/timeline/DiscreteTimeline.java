package genius.core.timeline;

import genius.core.Agent;

/**
 * Implementation of the timeline in which time is divided in rounds. Time does
 * not pass within a round. Note that requesting the total time is in this case
 * undefined.
 * 
 * See also {@link #getOwnRoundsLeft()}
 */
@SuppressWarnings("serial")
public class DiscreteTimeline extends Timeline 
{
	/** With 3 rounds set in Genius, this is set to 4. */
	private int totalRounds;
	
	/** Current round. E.g. with 3 rounds, it takes the values 1, 2, 3, and on 4 is the deadline. */
	protected int cRound;

	/**
	 * @param the
	 *            total number of rounds allowed on this timeline. Creates a
	 *            timeline with a deadline of {@link #totalRounds} number of
	 *            rounds. If 3, the play rounds are 1,2,3 and when we are on
	 *            round 4 the deadline is reached. see also {@link #getTime()}
	 */
	public DiscreteTimeline(int totalRounds) {
		this.totalRounds = totalRounds + 1;
		hasDeadline = true;
		cRound = 1;
	}

	/**
	 * @return the time, running from t = 0 (start) to t = 1 (deadline). The
	 *         time is normalized, so agents need not be concerned with the
	 *         actual internal clock.
	 */
	@Override
	public double getTime() {
		double t = (double) cRound / (double) totalRounds;
		if (t > 1)
			t = 1;
		return t;
	}

	public void increment() {
		cRound++;
	}

	public void setcRound(int cRound) {
		this.cRound = cRound;
	}

	/**
	 * {@link Agent#sleep(double)} requires this method.
	 */
	@Override
	public double getTotalTime() {
		return totalRounds;
	}

	/**
	 * Starting to count from 1, until the total amount of rounds.
	 */
	public int getRound() {
		return cRound;
	}

	public int getRoundsLeft() {
		return totalRounds - cRound - 1;
	}

	/**
	 * Be careful, this is not equal to the initializing value!
	 */
	public int getTotalRounds() {
		return totalRounds;
	}

	/**
	 * The total number of rounds for ourself. Be careful, this is not equal to
	 * the initializing value!
	 */
	public int getOwnTotalRounds() {
		return (int) Math.floor(getTotalRounds() / 2);
	}

	/**
	 * The number of rounds left for ourself.
	 */
	public int getOwnRoundsLeft() {
		return (int) Math.floor(getRoundsLeft() / 2);
	}

	public Type getType() {
		return Type.Rounds;
	}

	@Override
	public double getCurrentTime() {
		return cRound;
	}
}