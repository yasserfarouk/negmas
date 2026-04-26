package genius.core.persistent;

import java.io.Serializable;

public enum PersistentDataType implements Serializable {
	/**
	 * Data is not saved and null
	 */
	DISABLED,
	/**
	 * Container can hold any {@link Serializable} and is saved.
	 */
	SERIALIZABLE,

	/**
	 * Container holds {@link StandardInfoList} and can not be modified.
	 */
	STANDARD;
}
