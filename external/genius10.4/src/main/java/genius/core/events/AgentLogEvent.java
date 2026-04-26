package genius.core.events;

import java.util.Map;

import genius.core.parties.NegotiationParty;

/**
 * Reports special log info returned by an agent. This is information that the
 * agent asked to log. See
 * {@link NegotiationParty#negotiationEnded(genius.core.Bid)}. Immutable
 */
public class AgentLogEvent implements NegotiationEvent {

	private Map<String, String> log;
	private String id;

	/**
	 * Contains log info from the agent, as received from the
	 * {@link NegotiationParty#negotiationEnded(genius.core.Bid)} call,
	 * 
	 * @param agent
	 *            the agent that returned the info
	 * @param logresult
	 *            see {@link NegotiationParty#negotiationEnded(genius.core.Bid)}
	 */
	public AgentLogEvent(String agent, Map<String, String> logresult) {
		log = logresult;
	}

	public Map<String, String> getLog() {
		return log;
	}

	public String getAgent() {
		return id;
	}

}
