package genius.core.issue;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;

/**
 * Specifies a discrete value. An example of a discrete value is the value "red"
 * for the issue "car color".
 */
@XmlRootElement
public class ValueDiscrete extends Value {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7357447303601851761L;
	/** Name of the value, for example "red". */
	@XmlAttribute
	private String value;

	@SuppressWarnings("unused")
	private ValueDiscrete() {
	}

	/**
	 * Creates a discrete value with a name.
	 * 
	 * @param name
	 *            of the value.
	 */
	public ValueDiscrete(String name) {
		value = name;
	}

	public final ISSUETYPE getType() {
		return ISSUETYPE.DISCRETE;
	}

	/**
	 * @return name of the value.
	 */
	public String getValue() {
		return value;
	}

	public String toString() {
		return value;
	}

	@Override
	public int hashCode() {
		return value.hashCode();
	}

	@Override
	public boolean equals(Object obj) {
		if (obj instanceof ValueDiscrete) {
			ValueDiscrete val = (ValueDiscrete) obj;
			return value.equals(val.getValue());
		} else if (obj instanceof String) {
			String val = (String) obj;
			return (value.equals(val));
		} else
			return false;
	}
}