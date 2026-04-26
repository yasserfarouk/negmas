package genius.gui.panels;

import genius.core.listener.DefaultListenable;

/**
 * Stores a listen-able boolean value. Listeners always hear the current value,
 * not the lock setting.
 *
 */
public class BooleanModel extends DefaultListenable<Boolean>
		implements Lockable {
	private boolean value = false;
	private boolean lock = false;

	/**
	 * 
	 * @param b
	 *            the initial value for the boolean
	 */
	public BooleanModel(boolean b) {
		this.value = b;
	}

	/**
	 * @param newValue
	 *            the new value. Check {@link #isLocked()} before attempting
	 *            this
	 * @throws IllegalStateException
	 *             if object is locked.
	 */
	public void setValue(boolean newValue) {
		if (lock) {
			throw new IllegalStateException("Value is locked");
		}
		if (value != newValue) {
			value = newValue;
			notifyChange(value);
		}
	}

	public boolean getValue() {
		return value;
	}

	/**
	 * 
	 * @param isLock
	 *            if true, the value can not be changed. The GUI should also
	 *            reflect this setting eg by greying out the item.
	 */
	@Override
	public void setLock(boolean isLock) {
		lock = isLock;
		notifyChange(value);
	}

	/**
	 * @return true if the value is locked and can't be changed.
	 */
	@Override
	public boolean isLocked() {
		return lock;
	}
}
