package genius.gui.panels;

/**
 * Models can be lockable. The notification mechanism for lock info is not
 * specified here.
 */
public interface Lockable {
	/**
	 * 
	 * @param isLock
	 *            if true, the value can not be changed. The listeners of the
	 *            object should be notified about lock changes. The GUI should
	 *            also reflect this setting eg by greying out the item.
	 */
	public void setLock(boolean isLock);

	/**
	 * @return true if the value is locked and can't be changed.
	 */
	public boolean isLocked();

}
