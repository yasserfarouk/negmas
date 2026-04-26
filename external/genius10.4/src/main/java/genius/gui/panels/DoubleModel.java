package genius.gui.panels;

import genius.core.listener.DefaultListenable;

/**
 * Model behind a double value (from a text field). Listeners receive the new string value as object.
 *
 */
public class DoubleModel extends DefaultListenable<Double> implements Lockable 
{
	private Double value;
	private boolean isLocked = false;

	public DoubleModel(Double initial) {
		this.value = initial;
	}

	public void setText(Double newvalue) 
	{
		if (!value.equals(newvalue)) {
			value = newvalue;
			notifyChange(value);
		}
	}

	public Double getValue() 
	{
		return value;
	}
	
	@Override
	public void setLock(boolean isLock) 
	{
		this.isLocked = isLock;
		notifyChange(value);
	}

	@Override
	public boolean isLocked() {
		return isLocked;
	}

}
