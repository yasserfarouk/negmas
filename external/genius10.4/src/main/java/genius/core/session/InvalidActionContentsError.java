package genius.core.session;

import genius.core.AgentID;

@SuppressWarnings("serial")
public class InvalidActionContentsError extends ActionException {
	private AgentID agent;
	private String message;

	public InvalidActionContentsError(AgentID agent, String message) {
		this.agent = agent;
		this.message = message;
	}

	public String toString() {
		return "Agent " + agent + " created an action with invalid content: "
				+ message;
	}
}
