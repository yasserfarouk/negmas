package genius.core.timeline;

import java.io.Serializable;

import genius.core.timeline.Timeline.Type;

/**
 * read-only access to the actual timeline information. The returned information
 * should be up to date at all times and automatically updated.
 * 
 * @author W.Pasman
 *
 */
public interface TimeLineInfo extends Serializable {
	/**
	 * In a time-based protocol, time passes within a round. In contrast, in a
	 * rounds-based protocol time only passes when the action is presented.
	 */

	public Type getType();

	/**
	 * Gets the time, running from t = 0 (start) to t = 1 (deadline). The time
	 * is normalized, so agents need not be concerned with the actual internal
	 * clock.
	 *
	 * @return current time in the interval [0, 1].
	 */
	public double getTime();

	/**
	 * @return amount of time in seconds, or amount of rounds depending on
	 *         timeline type.
	 */
	public double getTotalTime();

	/**
	 * @return amount of seconds passed, or amount of rounds passed depending on
	 *         the timeline type.
	 */
	public double getCurrentTime();

}
