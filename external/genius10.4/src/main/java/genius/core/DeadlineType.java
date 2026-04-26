package genius.core;

public enum DeadlineType {
	/**
	 * deadline based on maximum run time (seconds)
	 */
	TIME,
	/**
	 * Deadline with maximum number o rounds
	 */
	ROUND;

	public String units() {
		return this == TIME ? "s" : "rounds";
	}
}
