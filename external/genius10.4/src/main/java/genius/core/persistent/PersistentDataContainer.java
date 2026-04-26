package genius.core.persistent;

import java.io.Serializable;

/**
 * Container that may hold data saved from previous sessions..
 */
public interface PersistentDataContainer extends Serializable {
	/**
	 * @return data that is in this container, or null if there is no data. This
	 *         data may have been saved earlier through put, or may be a fixed
	 *         data object. See {@link #getPersistentDataType()} to see the type
	 *         of this container
	 */
	public Serializable get();

	/**
	 * 
	 * @param data
	 *            the data to save for the next time. If not null, this data may
	 *            be saved and delivered back to the party the next time it
	 *            runs. Only allowed for containers of type
	 *            {@link PersistentDataType#SERIALIZABLE}.
	 */
	public void put(Serializable data);

	/**
	 * the type of the data in this container. Data put in this container may be
	 * saved, depending on this.
	 */
	public PersistentDataType getPersistentDataType();

}
