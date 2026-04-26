package genius.gui.panels;

import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import genius.core.listener.DefaultListenable;

/**
 * Model for {@link Integer} values. This improves type checking and also allows
 * us to use it with sliders. Changes on the lock and on value are reported.
 */
public class IntegerModel extends DefaultListenable<Integer>
		implements Lockable {

	private SpinnerNumberModel spinmodel;
	private boolean isLocked = false;

	public IntegerModel(Integer value, Integer minimum, Integer maximum,
			Integer stepSize) {
		spinmodel = new SpinnerNumberModel(value, minimum, maximum, stepSize);
		spinmodel.addChangeListener(new ChangeListener() {
			@Override
			public void stateChanged(ChangeEvent e) {
				notifyChange((Integer) spinmodel.getValue());
			}
		});
	}

	@SuppressWarnings("unchecked")
	public Integer getMinimum() {
		return (Integer) spinmodel.getMinimum();
	}

	@SuppressWarnings("unchecked")
	public Integer getMaximum() {
		return (Integer) spinmodel.getMaximum();
	}

	public javax.swing.SpinnerModel getSpinnerModel() {
		return spinmodel;
	}

	@SuppressWarnings("unchecked")
	public Integer getValue() {
		return (Integer) spinmodel.getValue();
	}

	public void setValue(Integer value) {
		spinmodel.setValue(value);
	}

	public void setMinimum(Integer newMinimum) {
		spinmodel.setMinimum(newMinimum);
	}

	public void setMaximum(Integer newMaximum) {
		spinmodel.setMinimum(newMaximum);

	}

	@Override
	public void setLock(boolean isLock) {
		this.isLocked = isLock;
		notifyChange((Integer) spinmodel.getValue());
	}

	@Override
	public boolean isLocked() {
		return isLocked;
	}

}
