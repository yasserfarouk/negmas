package genius.core.repository;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;

@XmlRootElement
public class ProtocolRepItem implements RepItem {

	@XmlAttribute
	String protocolName; 
	/**  the key: short but unique name of the protocol as it will be known in the nego system.
	 						* This is an arbitrary but unique label for this protocol.	 						 */
	@XmlAttribute
	String classPath; /** file path including the class name */
	@XmlAttribute
	String description; /** description of this agent */

	public ProtocolRepItem() {
	
	}
	


	public ProtocolRepItem(String protocolName, String classPath,
			String description) {
		super();
		this.protocolName = protocolName;
		this.classPath = classPath;
		this.description = description;
	}
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result
				+ ((classPath == null) ? 0 : classPath.hashCode());
		result = prime * result
				+ ((description == null) ? 0 : description.hashCode());
		result = prime * result
				+ ((protocolName == null) ? 0 : protocolName.hashCode());
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
		ProtocolRepItem other = (ProtocolRepItem) obj;
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
		return true;
	}
	/** Getters and Setters **/
	public String getName() {
		return protocolName;
	}

	public String getClassPath() {
		return classPath;
	}

	public String getDescription() {
		return description;
	}
}
