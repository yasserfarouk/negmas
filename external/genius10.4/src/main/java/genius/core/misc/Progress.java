package genius.core.misc;

import java.io.IOException;

/**
 * Contains the progress of something. This can be anything as long as it can be
 * counted with an integer and has a known integer end.
 *
 * For example we can count the progress in a tournament if we know that we doe
 * n negotiations in that tournament.
 */
public class Progress {

	/**
	 * Holds the current state of progress, the last state of progress, and
	 * spinner state
	 */
	private int current, total, spinnerState;

	/** Holds the nano time of when we started keeping progress */
	private long start;

	/**
	 * The spinner animation (can safely be changed to any other spinner
	 * animation of any length)
	 */
	private static String anim = "|/-\\";

	/** Flag for including current/total in the reporting (example: 5/20) */
	public static int CURRENT_TOTAL = 1;

	/** Flag for including percentage in the reporting (example: (25%)) */
	public static int PERCENTAGE = 2;

	/**
	 * Flag for including estimate time ahead (ETA) in report (example: ETA: 5
	 * minutes)
	 */
	public static int ETA = 4;

	/** Flag for including a spinner in report */
	public static int CHAR = 8;

	/** Flag for including all of the above in report */
	public static int ALL = CURRENT_TOTAL | PERCENTAGE | ETA | CHAR;

	/**
	 * Initializes a new instant of the Progress object. To use timing (ETA),
	 * you additionally should call the .start() function as closely to the
	 * function to be timed as possible.
	 *
	 * @param total
	 *            The total amount of progress that can be made
	 */
	public Progress(int total) {
		this.total = total;
	}

	/**
	 * Increases the progress by 1, but never beyond the total progress
	 */
	public void increment() {
		increment(1);
	}

	/**
	 * Increases the progress by n, but never beyond the total progress
	 * 
	 * @param n
	 *            the value to be added
	 */
	public void increment(int n) {
		current = Math.min(current + n, total);
	}

	/**
	 * Returns true if this progress is at last state.
	 *
	 * @return True if current &ge; total, false otherwise
	 */
	public boolean isDone() {
		return current >= total;
	}

	/**
	 * Mark now as the starting point in negotiation
	 */
	public void start() {
		start = System.nanoTime();
	}

	/**
	 * Gets the nano time passed since start of progress.
	 * 
	 * @return amount of nanoseconds since .start() was called
	 */
	public long getNowNano() {
		return System.nanoTime() - start;
	}

	/**
	 * Gets the time passed as an human readable string.
	 * 
	 * @param includeSeconds
	 *            if true, will print up to seconds. if false, will print up to
	 *            minutes.
	 * @return Time passed since .start() as a string (example: 2 hours, 5
	 *         minutes)
	 */
	public String getNowPretty(boolean includeSeconds) {
		return Time.prettyTimeSpan(getNowNano(), includeSeconds);
	}

	/**
	 * Get the final number for this progress
	 * 
	 * @return The number up to which to progress
	 */
	public int getTotal() {
		return total;
	}

	/**
	 * Gets the progress as a fraction
	 * 
	 * @return value between 0.0 (not started yet) and 1.0 (done)
	 */
	public double getProgress() {
		return (double) current / (double) total;
	}

	/**
	 * Gets the progress string. Will return depending on the used options. use
	 * the constant values in this class to generate the options. for example:
	 * getProgressString(Progress.ETA | Progress.CURRENT_TOTAL)
	 * <p>
	 * Progress.ETA: Report estimated time ahead Progress.CURRENT_TOTAL: Report
	 * current/total Progress.PERCENTAGE: Report percentage done Progress.CHAR:
	 * Show spinner char Progress.ALL: All of the above
	 * 
	 * @param options
	 *            See description
	 * @return String with the current progress
	 */
	public String getProgressString(int options) {
		String data = "";
		if ((options & CHAR) != 0) {
			data += anim.charAt(spinnerState) + " ";
			spinnerState = (spinnerState + 1) % anim.length();
		}
		if ((options & CURRENT_TOTAL) != 0) {
			data += String.format("%d/%d ", current, total);
		}
		if ((options & PERCENTAGE) != 0) {
			data += String.format("(%.0f%%) ", getProgress() * 100D);
		}
		if ((options & ETA) != 0) {
			data += String.format("ETA: %s", Time.prettyTimeSpan(getEta(), false));
		}
		return data;
	}

	/**
	 * Erases previously printed line and prints the progress string. Will print
	 * depending on the used options. use the constant values in this class to
	 * generate the options. for example: getProgressString(Progress.ETA |
	 * Progress.CURRENT_TOTAL)
	 * <p>
	 * Progress.ETA: Report estimated time ahead Progress.CURRENT_TOTAL: Report
	 * current/total Progress.PERCENTAGE: Report percentage done Progress.CHAR:
	 * Show spinner char Progress.ALL: All of the above
	 * 
	 * @param options
	 *            See description
	 */
	public void printProgressToConsole(int options) {
		try {
			String data = "\r                                                            "; // to
																							// make
																							// sure
																							// we
																							// overwrite
			data += "\r" + getProgressString(options);
			System.out.write(data.getBytes());
		} catch (IOException e) {
			// do nothing
		}
	}

	/**
	 * Gets the estimated time remaining based on the time we used so fare
	 * 
	 * @return estimated time remaining in nano seconds.
	 */
	public double getEta() {
		return (getNowNano() / getProgress()) - getNowNano();
	}

}
