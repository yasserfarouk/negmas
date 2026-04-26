package genius.gui.panels;

import javax.swing.BoundedRangeModel;
import javax.swing.event.ChangeListener;

/**
 * Adapts {@link IntegerModel} to {@link BoundedRangeModel} so that we can use
 * it for sliders.
 */
public class IntegerToBoundeRangeAdapter implements BoundedRangeModel {

	private IntegerModel model;
	private boolean valueIsAdjusting = false;

	public IntegerToBoundeRangeAdapter(IntegerModel model) {
		this.model = model;
	}

	@Override
	public int getMinimum() {
		return model.getMinimum();
	}

	@Override
	public void setMinimum(int newMinimum) {
		// HACK. Does this work?? Problems down the line?
		model.setMinimum(newMinimum);
	}

	@Override
	public int getMaximum() {
		return model.getMaximum();
	}

	@Override
	public void setMaximum(int newMaximum) {
		model.setMaximum(newMaximum);

	}

	@Override
	public int getValue() {
		return model.getValue();
	}

	@Override
	public void setValue(int newValue) {
		model.setValue(newValue);
	}

	@Override
	public void setValueIsAdjusting(boolean b) {
		valueIsAdjusting = b;
	}

	@Override
	public boolean getValueIsAdjusting() {
		return valueIsAdjusting;
	}

	@Override
	public int getExtent() {
		return 0;
	}

	@Override
	public void setExtent(int newExtent) {
		// not supported
	}

	@Override
	public void setRangeProperties(int value, int extent, int min, int max,
			boolean adjusting) {
	}

	@Override
	public void addChangeListener(ChangeListener x) {
		model.getSpinnerModel().addChangeListener(x);
	}

	@Override
	public void removeChangeListener(ChangeListener x) {
		model.getSpinnerModel().addChangeListener(x);
	}

}
