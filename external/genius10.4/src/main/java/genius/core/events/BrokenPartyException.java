package genius.core.events;

import genius.core.parties.NegotiationPartyInternal;
import genius.core.session.Session;
import genius.core.session.SessionConfiguration;

/**
 * Thrown if construction of {@link NegotiationPartyInternal} fails. Contains
 * partial data scraps from the failed construction, which is needed for
 * logging. <br>
 * <b>Notice</b>: this contains only the {@link SessionConfiguration} but no
 * instantiated Profiles. This is because when this exception is thrown not all
 * profiles may have been created, eg when we have to time-out construction.
 * Therefore you will have to load the profiles (possibly again) if you want to
 * access them when handling this error.
 */
@SuppressWarnings("serial")
public class BrokenPartyException extends Exception {

	private Session session;
	private SessionConfiguration configuration;

	public BrokenPartyException(String mes, SessionConfiguration configuration, Session session, Throwable cause) {
		super(mes, cause);
		this.session = session;
		this.configuration = configuration;

	}

	public Session getSession() {
		return session;
	}

	public SessionConfiguration getConfiguration() {
		return configuration;
	}

}
