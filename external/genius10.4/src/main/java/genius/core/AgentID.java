package genius.core;

import java.io.Serializable;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;

/**
 * Unique ID for an agent. Immutable. Not guaranteed to be unique.
 *
 */
@XmlRootElement
public class AgentID implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6858722448196458573L;

	@XmlAttribute
	String ID;

	private static int serialNr = 0; // for generating unique party ids.

	/**
	 * only for XML deserialization.
	 */
	@SuppressWarnings("unused")
	private AgentID() {

	}

	public AgentID(String id) {
		this.ID = id;
	}

	/**
	 * factory function
	 * 
	 * @param basename
	 *            the basename for the agent ID
	 * @return a new unique AgentID based on the basename plus a unique serial
	 *         number. These are unique only for a single run of the system, you
	 *         can not assume new generated IDs to be different from IDs
	 *         generated in earlier runs.
	 */
	public static AgentID generateID(String basename) {
		return new AgentID(basename + "@" + serialNr++);
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((ID == null) ? 0 : ID.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		AgentID other = (AgentID) obj;
		if (ID == null) {
			if (other.ID != null)
				return false;
		} else if (!ID.equals(other.ID))
			return false;
		return true;
	}

	@Override
	public String toString() {
		return ID;
	}

	/**
	 * 
	 * @return the agent's name, but with the serial number ("@X" with X a
	 *         integer) extension stripped if there is such an extension
	 */
	public String getName() {

		if (ID.matches(".+@[0-9]+")) {
			return ID.substring(0, ID.lastIndexOf("@"));
		}
		return ID;
	}
}
