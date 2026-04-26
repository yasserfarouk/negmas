package genius.core.events;

/**
 * Indicates that a session started. You get this message only in tournaments,
 * where multiple sessions can be running.
 * 
 * @author W.Pasman 15jul15
 *
 */
public class TournamentSessionStartedEvent implements TournamentEvent {

	private final int currentSession;
	private final int totalSessions;

	/**
	 * @param session
	 *            First session has number 1.
	 * @param total
	 */
	public TournamentSessionStartedEvent(int session, int total) {
		currentSession = session;
		totalSessions = total;
	}

	public int getCurrentSession() {
		return currentSession;
	}

	public int getTotalSessions() {
		return totalSessions;
	}

}
