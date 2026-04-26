package genius.core.events;

import genius.core.parties.NegotiationParty;
import genius.core.parties.NegotiationPartyInternal;
import genius.core.parties.SessionsInfo;
import genius.core.session.Session;

/**
 * This exception indicates that something went wrong but we did an automatic
 * recovery. For example the session was succesfully completed but the aftermath
 * failed. This can happen when there is an issue while writing the
 * {@link SessionsInfo} or when the
 * {@link NegotiationParty#negotiationEnded(genius.core.Bid)} call failed.
 *
 */
public class RecoverableSessionErrorEvent implements NegotiationEvent {

	private Session session;
	private NegotiationPartyInternal party;
	private Exception exception;

	public RecoverableSessionErrorEvent(Session session, NegotiationPartyInternal party, Exception e) {
		this.session = session;
		this.party = party;
		this.exception = e;
	}

	public Exception getException() {
		return exception;
	}

	public NegotiationPartyInternal getParty() {
		return party;
	}

	public Session getSession() {
		return session;
	}
}
