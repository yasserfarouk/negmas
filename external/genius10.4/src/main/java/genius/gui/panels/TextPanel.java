package genius.gui.panels;

import java.awt.BorderLayout;

import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;

import genius.core.listener.Listener;

/**
 * Shows a single text line input area.
 */
@SuppressWarnings("serial")
public class TextPanel extends JPanel {
	private final JTextField textfield;
	
	public TextPanel(final TextModel model) {
		setLayout(new BorderLayout());
		 this.textfield = new JTextField(model.getText());
		add(textfield, BorderLayout.CENTER);
		// not working?
		// setMaximumSize(new Dimension(99999999, 30));
		textfield.getDocument().addDocumentListener(new DocumentListener() {

			@Override
			public void removeUpdate(DocumentEvent e) {
				if (!textfield.getText().isEmpty()) {
					model.setText(textfield.getText());
				}
				// we ignore empty text to avoid reacting on replace procedure
				// too early
			}

			@Override
			public void insertUpdate(DocumentEvent e) {
				model.setText(textfield.getText());
			}

			@Override
			public void changedUpdate(DocumentEvent e) {
				model.setText(textfield.getText());
			}
		});

		model.addListener(new Listener<String>() {
			@Override
			public void notifyChange(final String data) {
				SwingUtilities.invokeLater(new Runnable() {

					@Override
					public void run() {
						textfield.setText((String) data);
					}
				});
			}
		});

	}
	public JTextField getTextField() {
		return textfield;
	}
}
