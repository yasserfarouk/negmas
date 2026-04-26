package genius.gui.panels;

import java.awt.BorderLayout;
import java.awt.Checkbox;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;

import javax.swing.JLabel;
import javax.swing.JPanel;

import genius.core.listener.Listener;

/**
 * Panel showing a checkbox with a title and value. This panel supports a
 * BooleanModel (you can't with the default {@link Checkbox}).
 */
@SuppressWarnings("serial")
public class CheckboxPanel extends JPanel {

	private JLabel label;
	private Checkbox box;
	private BooleanModel model;

	/**
	 * 
	 * @param text
	 *            the text for the label
	 * @param boolModel
	 *            the boolean model
	 */
	public CheckboxPanel(final String text, final BooleanModel boolModel) {
		this.model = boolModel;

		setLayout(new BorderLayout());
		label = new JLabel(text);
		box = new Checkbox("", boolModel.getValue());
		enable1();

		// connect the box to the model,
		box.addItemListener(new ItemListener() {

			@Override
			public void itemStateChanged(ItemEvent e) {
				if (box.getState() != model.getValue()) {
					model.setValue(box.getState());
					enable1();
				}
			}
		});

		// and the model to the box
		model.addListener(new Listener<Boolean>() {

			@Override
			public void notifyChange(Boolean data) {
				// FIXME invokeLater
				box.setState(data);
				enable1();
			}
		});

		add(box, BorderLayout.WEST);
		add(label, BorderLayout.CENTER);
	}

	/**
	 * Update enabled-ness of panel. Just calling enable() doesn nothing?
	 */
	private void enable1() {
		boolean enabled = !model.isLocked();
		box.setEnabled(enabled);
		label.setEnabled(enabled);
	}

}
