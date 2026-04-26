package genius.core.timeline;

/**
 * A time line, running from t = 0 (start) to t = 1 (deadline). The timeline
 * renormalizes real time.
 */
@SuppressWarnings("serial")
public class ContinuousTimeline extends Timeline {
	private final int totalSeconds;
	protected long startTime;

	/**
	 * Creates a timeline with a deadline of {@link #totalSeconds} number of
	 * seconds.
	 */
	public ContinuousTimeline(int totalSecs) {
		totalSeconds = totalSecs;
		startTime = System.nanoTime();
		hasDeadline = true;
	}

	/**
	 * Gets the elapsed time in seconds. Use {@link #getTime()} for a more
	 * generic version.
	 */
	public double getElapsedSeconds() {
		long t2 = System.nanoTime();
		return ((t2 - startTime) / 1000000000.0);
	}

	/**
	 * Gets the elapsed time in seconds. Use {@link #getTime()} for a more
	 * generic version.
	 */
	public double getElapsedMilliSeconds() {
		long t2 = System.nanoTime();
		return ((t2 - startTime) / 1000000.0);
	}

	/**
	 * Gets the total negotiation time in miliseconds
	 */
	public long getTotalMiliseconds() {
		return 1000 * totalSeconds;
	}

	/**
	 * Gets the total negotiation time in seconds
	 */
	public long getTotalSeconds() {
		return totalSeconds;
	}

	/**
	 * Gets the time, running from t = 0 (start) to t = 1 (deadline). The time
	 * is normalized, so agents need not be concerned with the actual internal
	 * clock.
	 */
	public double getTime() {
		double t = getElapsedSeconds() / (double) totalSeconds;
		if (t > 1)
			t = 1;
		return t;
	}

	@Override
	public double getTotalTime() {
		return getTotalSeconds();
	}

	@Override
	public double getCurrentTime() {
		return getElapsedSeconds();
	}

	public void reset() {
		startTime = System.nanoTime();
	}
}
