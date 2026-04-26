package genius.gui.tournamentvars;

import java.awt.Frame;
import java.awt.Panel;

import javax.swing.BoxLayout;
import javax.swing.JTextField;

import genius.gui.panels.DefaultOKCancelDialog;

public class SingleStringVarUI extends DefaultOKCancelDialog 
{
	private static final long serialVersionUID = -6935071049618754059L;
	private JTextField textField;
	
	public SingleStringVarUI(Frame frame) {
		super(frame, "Number of sessions");
	}
	
	@Override
	public Panel getPanel() {
		textField = new JTextField();
		Panel panel = new Panel();
		panel.setLayout(new BoxLayout(panel,BoxLayout.Y_AXIS));
		panel.add(textField);
		return panel;

	}

	@Override
	public Object ok() {
		return textField.getText();
	}

}
