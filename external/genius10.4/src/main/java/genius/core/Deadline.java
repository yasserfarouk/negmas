package genius.core;

import java.io.Serializable;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;

/**
 * Contains the deadline - either rounds based or time based.
 * <p>
 * Deadline is a final object and can be serialized to xml. Immutable.
 * 
 * @author W.Pasman
 */
@SuppressWarnings("serial")
@XmlRootElement
@XmlAccessorType(XmlAccessType.FIELD)
public class Deadline implements Serializable {

	@XmlElement
	private final Integer value;
	@XmlElement
	private final DeadlineType type;

	/**
	 * Default timeout in seconds; also used as the realtime timeout 
	 * to abort the round-based negotiations.
	 */
	private final static Integer DEFAULT_TIME_OUT = 60;

	/**
	 * Create default value.
	 */
	public Deadline() {
		value = DEFAULT_TIME_OUT;
		type = DeadlineType.TIME;
	};

	public Deadline(int val, DeadlineType tp) {
		if (val <= 0) {
			throw new IllegalArgumentException("value must be >0 but got " + val);
		}
		if (tp == null) {
			throw new NullPointerException("type is null");
		}
		value = val;
		type = tp;
	}

	/**
	 * @return the total value of the deadline (seconds or rounds)
	 */
	public int getValue() {
		return value;
	}

	/**
	 * 
	 * @return the {@link DeadlineType} of this deadline
	 */
	public DeadlineType getType() {
		return type;
	}

	/**
	 * @return the default time-out for function calls in the agents
	 */
	public Integer getDefaultTimeout() {
		return DEFAULT_TIME_OUT;
	}

	public String toString() {
		return "Deadline:" + valueString();
	}

	/**
	 * @return just the value of this deadline, eg "10s".
	 */
	public String valueString() {
		return value + type.units();
	}

	/**
	 * 
	 * @return the time, or a default time time-out. This is needed to determine
	 *         the time-out for code execution with {@link DeadlineType#ROUND}
	 *         deadlines.
	 */
	public int getTimeOrDefaultTimeout() {
		if (type == DeadlineType.ROUND) {
			return DEFAULT_TIME_OUT;
		}
		return value;

	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((type == null) ? 0 : type.hashCode());
		result = prime * result + ((value == null) ? 0 : value.hashCode());
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
		Deadline other = (Deadline) obj;
		if (type != other.type)
			return false;
		if (value == null) {
			if (other.value != null)
				return false;
		} else if (!value.equals(other.value))
			return false;
		return true;
	}

}