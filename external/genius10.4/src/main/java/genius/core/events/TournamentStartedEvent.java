package genius.core.events;

/**
 * Indicates that a tournament started.
 * 
 * @author W.Pasman 15jul15
 *
 */
public class TournamentStartedEvent implements TournamentEvent {

	private final int totalNumberOfSessions;
	private final int tournamentNumber;

	public TournamentStartedEvent(int tournament, int totalSessions) {
		tournamentNumber = tournament;
		totalNumberOfSessions = totalSessions;
	}

	public int getTotalNumberOfSessions() {
		return totalNumberOfSessions;
	}

	public int getTournamentNumber() {
		return tournamentNumber;
	}

}
