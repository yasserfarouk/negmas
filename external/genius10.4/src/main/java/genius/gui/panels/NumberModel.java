package genius.gui.panels;

import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import genius.core.listener.DefaultListenable;

/**
 * Model for {@link Integer} values. This improves type checking and also allows
 * us to use it with sliders. Changes on the lock and on value are reported.
 */
public class NumberModel extends DefaultListenable<Number> implements Lockable {

	private SpinnerNumberModel spinmodel;
	private boolean isLocked = false;

	public NumberModel(Number value, Comparable minimum, Comparable maximum,
			Number stepSize) {
		spinmodel = new SpinnerNumberModel(value, minimum, maximum, stepSize);
		spinmodel.addChangeListener(new ChangeListener() {
			@Override
			public void stateChanged(ChangeEvent e) {
				notifyChange((Number) spinmodel.getValue());
			}
		});
	}

	@SuppressWarnings("unchecked")
	public Number getMinimum() {
		return (Number) spinmodel.getMinimum();
	}

	@SuppressWarnings("unchecked")
	public Number getMaximum() {
		return (Number) spinmodel.getMaximum();
	}

	public javax.swing.SpinnerModel getSpinnerModel() {
		return spinmodel;
	}

	@SuppressWarnings("unchecked")
	public Number getValue() {
		return (Number) spinmodel.getValue();
	}

	public void setValue(Number value) {
		spinmodel.setValue(value);
	}

	public void setMinimum(Comparable newMinimum) {
		spinmodel.setMinimum(newMinimum);
	}

	public void setMaximum(Comparable newMaximum) {
		spinmodel.setMinimum(newMaximum);

	}

	@Override
	public void setLock(boolean isLock) {
		this.isLocked = isLock;
		notifyChange(((Number) spinmodel.getValue()));
	}

	@Override
	public boolean isLocked() {
		return isLocked;
	}

}
