package genius.core.actions;

import javax.xml.bind.annotation.XmlRootElement;

import genius.core.Agent;
import genius.core.AgentID;

/**
 * immutable.
 *
 */
@XmlRootElement
public class NoAction extends DefaultAction {

	public NoAction(AgentID agent) {
		super(agent);
	}

	public NoAction(Agent agent) {
		this(agent.getAgentID());
	}

	public String toString() {
		return "(No additional action)";
	}
}
