package genius.core.issue;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;

/**
 * @author Dmytro Tykhonov
 */
@XmlRootElement
public class ValueReal extends Value {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8695408707220542052L;

	@XmlAttribute
	private double value;

	/**
	 * only for XML deserialization.
	 */
	@SuppressWarnings("unused")
	private ValueReal() {
	}

	public ValueReal(double r) {
		value = r;
	}

	public ISSUETYPE getType() {
		return ISSUETYPE.REAL;
	}

	public double getValue() {
		return value;
	}

	public String toString() {
		return Double.toString(value);
	}

	public boolean equals(Object pObject) {
		if (pObject instanceof ValueReal) {
			ValueReal val = (ValueReal) pObject;
			return value == val.getValue();
		} else if (pObject instanceof Double) {
			double val = (Double) pObject;
			return value == val;
		} else
			return false;
	}
}