package genius.core.issue;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;

/**
 * Specifies an integer value. An example of an integer value is the value 3 for
 * the issue price with range [0,10].
 */
@XmlRootElement
public class ValueInteger extends Value {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8706454849472666446L;
	@XmlAttribute
	private int value;

	/**
	 * only for XML deserialization.
	 */
	@SuppressWarnings("unused")
	private ValueInteger() {
	}

	/**
	 * Creates an integer value with a value.
	 * 
	 * @param i
	 *            value for an issue.
	 */
	public ValueInteger(int i) {
		value = i;
	}

	public ISSUETYPE getType() {
		return ISSUETYPE.INTEGER;
	}

	/**
	 * @return value of this issue.
	 */
	public int getValue() {
		return value;
	}

	public String toString() {
		return Integer.toString(value);
	}

	public boolean equals(Object pObject) {
		if (pObject instanceof ValueInteger) {
			ValueInteger val = (ValueInteger) pObject;
			return value == val.getValue();
		} else if (pObject instanceof Integer) {
			int val = (Integer) pObject;
			return value == val;
		} else
			return false;
	}

	@Override
	public int hashCode() {
		return 0;
	}
}