package genius.core.timeline;

/**
 * A time line, running from t = 0 (start) to t = 1 (deadline).
 */
public abstract class Timeline implements TimeLineInfo {
	protected boolean hasDeadline;
	protected boolean paused = false;

	/**
	 * In a time-based protocol, time passes within a round. In contrast, in a
	 * rounds-based protocol time only passes when the action is presented.
	 */
	public enum Type {
		/**
		 * Time passes with wall clock time. Usually the clock used is
		 * {@link System#nanoTime()} or similar so it also proceeds while others
		 * use the CPU.
		 */
		Time,
		/**
		 * time advances only when the action is presented. Time is frozen while
		 * an agent is computing its next step.
		 */
		Rounds;
	}

	/**
	 * Gets the time, running from t = 0 (start) to t = 1 (deadline). The time
	 * is normalized, so agents need not be concerned with the actual internal
	 * clock.
	 *
	 * @return current time in the interval [0, 1].
	 */
	public abstract double getTime();

	/**
	 * @return amount of time in seconds, or amount of rounds depending on
	 *         timeline type.
	 */
	public abstract double getTotalTime();

	/**
	 * @return amount of seconds passed, or amount of rounds passed depending on
	 *         the timeline type.
	 */
	public abstract double getCurrentTime();

	/**
	 * @return true if deadline is reached.
	 */
	public boolean isDeadlineReached() {
		return hasDeadline && (getTime() >= 1.0);
	}

	/**
	 * @return type of time: Type.Time or Type.Rounds.
	 */
	public Type getType() {
		return Type.Time;
	}

	/**
	 * @return true if timeline is paused.
	 */
	public boolean isPaused() {
		return paused;
	}
}