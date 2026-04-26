package genius.core.actions;

import genius.core.AgentID;

/**
 * inform about some property. Immutable.
 * 
 * @author dfesten on 21-8-2014.
 * @author W.Pasman: made immutable
 */
public class Inform extends DefaultAction {
	private Object value;
	private String name;

	public Inform(AgentID id, String name, Object value) {
		super(id);
		this.name = name;
		this.value = value;
	}

	public Object getValue() {
		return value;
	}

	public String getName() {
		return name;
	}

	/**
	 * Enforces that actions implements a string-representation.
	 */
	@Override
	public String toString() {
		return name + ":" + value.toString();
	}
}
