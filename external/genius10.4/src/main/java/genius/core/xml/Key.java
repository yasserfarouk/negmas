package genius.core.xml;

/**
 * Utility object for use with HashMaps. With this you can create a unique key
 * with a given string value. Allows us to use multiple different keys (equals
 * returs false) while the toString is in fact identical.
 */
public class Key {

	private String name;

	public Key(String string) {
		this.name = string;
	}

	@Override
	public int hashCode() {
		return 0;
	}

	@Override
	public boolean equals(Object obj) {
		return this == obj;
	}

	@Override
	public String toString() {
		return name;
	}
}
