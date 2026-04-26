package genius.core.repository;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;

/**
 * reference to a protocol class. Immutable.
 *
 */
@XmlRootElement
public class MultiPartyProtocolRepItem implements RepItem {

	@XmlAttribute
	private String protocolName;
	/**
	 * the key: short but unique name of the protocol as it will be known in the
	 * nego system. This is an arbitrary but unique label for this protocol.
	 */
	@XmlAttribute
	private String classPath;
	/** file path including the class name */
	@XmlAttribute
	private String description;
	/** description of this agent */

	@XmlAttribute
	// RA: For multiparty negotiation, there are two type of agents: mediator
	// and negotiating party
	private Boolean hasMediator;
	/** whether the protocol involves a mediator */

	@XmlAttribute
	// If there is a mediator, does it have any preference profile?
	private Boolean hasMediatorProfile;

	public MultiPartyProtocolRepItem() {

	}

	public MultiPartyProtocolRepItem(String protocolName, String classPath, String description, Boolean mediator,
			Boolean hasMediatorProfile) {
		super();
		this.protocolName = protocolName;
		this.classPath = classPath;
		this.description = description;
		this.hasMediator = mediator; // RA
		this.hasMediatorProfile = hasMediatorProfile;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((classPath == null) ? 0 : classPath.hashCode());
		result = prime * result + ((description == null) ? 0 : description.hashCode());
		result = prime * result + ((protocolName == null) ? 0 : protocolName.hashCode());
		result = prime * result + ((hasMediator == null) ? 0 : hasMediator.hashCode()); // RA
		result = prime * result + ((hasMediatorProfile == null) ? 0 : hasMediatorProfile.hashCode()); // RA
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
		MultiPartyProtocolRepItem other = (MultiPartyProtocolRepItem) obj;
		if (classPath == null) {
			if (other.classPath != null)
				return false;
		} else if (!classPath.equals(other.classPath))
			return false;

		if (description == null) {
			if (other.description != null)
				return false;
		} else if (!description.equals(other.description))
			return false;

		if (protocolName == null) {
			if (other.protocolName != null)
				return false;
		} else if (!protocolName.equals(other.protocolName))
			return false;

		if (hasMediator == null) { // RA
			if (other.hasMediator != null)
				return false;
		} else if (hasMediator != other.hasMediator)
			return false;

		return true;
	}

	/** Getters. No setters, this is immutable object **/
	public String getName() {
		return protocolName;
	}

	public String getClassPath() {
		return classPath;
	}

	public String getDescription() {
		return description;
	}

	public Boolean getHasMediator() { // RA
		return hasMediator;
	}

	@Override
	public String toString() {
		return getName();
	}

}
