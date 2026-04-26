package genius.gui.panels;

import java.awt.BorderLayout;
import java.awt.Dimension;

import javax.swing.JComponent;
import javax.swing.JFormattedTextField;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.JTextField;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;

import genius.core.listener.Listener;

/**
 * Shows a single text line input area for a double value.
 */
@SuppressWarnings("serial")
public class DoublePanel extends JPanel 
{
	private final JLabel jlabel;
	private final JTextField textfield;
	DoubleModel model;
	
	public DoublePanel(String label, final DoubleModel model) 
	{
		setLayout(new BorderLayout());
				
		this.model = model;
		setLayout(new BorderLayout());
		
		this.jlabel = new JLabel(label);
		jlabel.setHorizontalAlignment(SwingConstants.RIGHT);
		add(jlabel, BorderLayout.WEST);
		
		JSpinner jSpinner = new JSpinner(new SpinnerNumberModel((double) model.getValue(), 0.0, 1.0, 0.001));
		jSpinner.setMaximumSize(new Dimension(300, 30));
		JSpinner.NumberEditor numberEditor = new JSpinner.NumberEditor(jSpinner,"0.0000");
		jSpinner.setEditor(numberEditor);
		this.textfield = getTextField(jSpinner);
		this.textfield.setColumns(5);
		
		add(jSpinner, BorderLayout.EAST);
		
		// not working?
		// setMaximumSize(new Dimension(99999999, 30));
		textfield.getDocument().addDocumentListener(new DocumentListener() {

			@Override
			public void removeUpdate(DocumentEvent e) {
				if (!textfield.getText().isEmpty()) 
				{
					model.setText(parseDouble());
				}
				// we ignore empty text to avoid reacting on replace procedure
				// too early
			}

			@Override
			public void insertUpdate(DocumentEvent e) {
				model.setText(parseDouble());
			}

			@Override
			public void changedUpdate(DocumentEvent e) {
				model.setText(parseDouble());
			}
		});

		model.addListener(new Listener<Double>() {
			@Override
			public void notifyChange(final Double data) {
				SwingUtilities.invokeLater(new Runnable() {

					@Override
					public void run() {
						textfield.setText("" + data);
						updateEnabledness();
					}
				});
			}
		});
		
		updateEnabledness();
	}
	
	private void updateEnabledness() 
	{
		boolean enabled = !model.isLocked();
		setEnabled(enabled);
		textfield.setEnabled(enabled);
		jlabel.setEnabled(enabled);
	}
	
	private double parseDouble()
	{
		double d = 0;
		String text = textfield.getText();
		String err = "The value \"" + text + "\" is not a valid number.";
		try {
			d = Double.parseDouble(text);
			if (d < 0 || d > 1) {
				showError(err);
				return 0;
			}
		} catch (Exception e) {
			showError(err);
			return 0;
		}
		return d;
	}
	
	private void showError(String error) 
	{
		JOptionPane.showMessageDialog(null, error, "Edit error", 0);
	}	
	
	private JFormattedTextField getTextField(JSpinner spinner) {
	    JComponent editor = spinner.getEditor();
	    if (editor instanceof JSpinner.DefaultEditor) {
	        return ((JSpinner.DefaultEditor)editor).getTextField();
	    } else {
	        System.err.println("Unexpected editor type: "
	                           + spinner.getEditor().getClass()
	                           + " isn't a descendant of DefaultEditor");
	        return null;
	    }
	}
}
