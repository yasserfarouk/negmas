package genius.core.list;

import java.io.Serializable;

/**
 * tuple with two elements of different types. Immutable
 *
 * @param <T1>
 *            type of the first element of the tuple
 * @param <T2>
 *            type of the second element of the tuple
 */
@SuppressWarnings("serial")
public class Tuple<T1, T2> implements Serializable {
	private T1 element1;
	private T2 element2;

	public Tuple(T1 element1, T2 element2) {
		this.element1 = element1;
		this.element2 = element2;

	}

	public T1 get1() {
		return element1;
	}

	public T2 get2() {
		return element2;
	}

	@Override
	public String toString() {
		return "<" + element1 + "," + element2 + ">";
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((element1 == null) ? 0 : element1.hashCode());
		result = prime * result + ((element2 == null) ? 0 : element2.hashCode());
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
		Tuple other = (Tuple) obj;
		if (element1 == null) {
			if (other.element1 != null)
				return false;
		} else if (!element1.equals(other.element1))
			return false;
		if (element2 == null) {
			if (other.element2 != null)
				return false;
		} else if (!element2.equals(other.element2))
			return false;
		return true;
	}
}
