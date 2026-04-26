package genius.gui.panels;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Dimension;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSpinner;

/**
 * Spinner but with text label.
 *
 */
@SuppressWarnings("serial")
public class SpinnerPanel extends JPanel {

	private final JSpinner spinner;
	private final NumberModel model;
	private final JLabel label;

	public SpinnerPanel(final String labeltext, final NumberModel model) {
		this.model = model;
		setLayout(new BorderLayout());
		label = new JLabel(labeltext);
		add(label, BorderLayout.WEST);
		label.setPreferredSize(new Dimension(120, 10));

		spinner = new JSpinner(model.getSpinnerModel());
		spinner.setMaximumSize(new Dimension(300, 30));
		add(spinner, BorderLayout.CENTER);
		// aligns the RIGHT side of the panel with the center of the parent.
		// This limits the total width
		setAlignmentX(Component.RIGHT_ALIGNMENT);
		setMaximumSize(new Dimension(3000000, 30));

		model.addListener(data -> enable1());
		enable1();

	}

	private void enable1() {
		boolean enabled = !model.isLocked();
		spinner.setEnabled(enabled);
		label.setEnabled(enabled);
	}
}
