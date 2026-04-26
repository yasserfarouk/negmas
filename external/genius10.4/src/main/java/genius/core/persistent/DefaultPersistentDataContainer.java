package genius.core.persistent;

import java.io.Serializable;

/**
 * Default implementation. Just stores the info locally. In fact these
 * containers are not 'persistent' but created every session and then re-loaded
 * with data if applicable.
 */
public class DefaultPersistentDataContainer implements PersistentDataContainer {
	/**
	 * 
	 */
	private static final long serialVersionUID = 6859456442618522085L;
	private Serializable data;
	private PersistentDataType type;

	public DefaultPersistentDataContainer(Serializable storage, PersistentDataType type) {
		if (type == null)
			throw new NullPointerException("type=null");
		this.data = storage;
		this.type = type;
	}

	@Override
	public Serializable get() {
		return data;
	}

	@Override
	public void put(Serializable data) {
		if (type != PersistentDataType.SERIALIZABLE) {
			throw new IllegalStateException(
					"put is allowed only for containers of type SERIALIZABLE. This type is " + type);
		}
		this.data = data;
	}

	@Override
	public PersistentDataType getPersistentDataType() {
		return type;
	}

}
