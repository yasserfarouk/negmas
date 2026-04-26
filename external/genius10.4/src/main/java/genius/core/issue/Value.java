package genius.core.issue;

import java.io.Serializable;

import javax.xml.bind.annotation.XmlSeeAlso;
import javax.xml.bind.annotation.XmlType;

/**
 * Specifies a generic value of an issue. This superclass needs to be extended
 * by a subclass.
 * <p>
 * Value objects are immutable.
 */
@XmlSeeAlso({ ValueInteger.class, ValueDiscrete.class, ValueReal.class })
@XmlType
public class Value implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1212374174018193000L;

	/**
	 * Empty constructor used to createFrom a new Value.
	 */
	public Value() {
	}

	/**
	 * @return type of the issue.
	 */
	public ISSUETYPE getType() {
		return ISSUETYPE.UNKNOWN;
	};

	public String toString() {
		return "unknown!";
	}
}
