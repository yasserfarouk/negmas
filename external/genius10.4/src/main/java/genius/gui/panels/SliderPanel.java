package genius.gui.panels;

import java.awt.BorderLayout;
import java.awt.Dimension;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.JSpinner;

import genius.core.listener.Listener;

/**
 * Shows slider plus spinner with optional ticks, optionally logarithmic, and
 * optionally with percentages
 *
 */
@SuppressWarnings("serial")
public class SliderPanel extends JPanel {
	private IntegerModel model;
	private JSlider slider;
	private JLabel label;
	private JSpinner spinner;

	public SliderPanel(String name, IntegerModel m) {
		model = m;
		setLayout(new BorderLayout());
		label = new JLabel(name);
		add(label, BorderLayout.WEST);

		slider = new JSlider(new IntegerToBoundeRangeAdapter(model));
		add(slider, BorderLayout.CENTER);
		spinner = new JSpinner(model.getSpinnerModel());
		spinner.setMaximumSize(new Dimension(300, 30));
		add(spinner, BorderLayout.EAST);

		// connect model->enabled to GUI appearance.
		model.addListener(new Listener<Integer>() {
			@Override
			public void notifyChange(Integer data) {
				updateEnabledness();
			}

		});
		updateEnabledness();
	}

	/**
	 * 
	 * @return the slider part. Used for testing, you should not need this and
	 *         communicate through the model you provided.
	 */
	public JSlider getSlider() {
		return slider;
	}

	private void updateEnabledness() {
		boolean enabled = !model.isLocked();
		setEnabled(enabled);
		label.setEnabled(enabled);
		slider.setEnabled(enabled);
		spinner.setEnabled(enabled);
	}

}
