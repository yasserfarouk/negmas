package genius.gui.panels;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Dimension;

import javax.swing.JLabel;
import javax.swing.JPanel;

/**
 * Panel with a text on the left and an arbitrary component on the right
 *
 */

@SuppressWarnings("serial")
public class LabelAndComponent extends JPanel {
	JLabel label;
	
	public LabelAndComponent(String text, Component comp) {
		super(new BorderLayout());
		this.label = new JLabel(text);
		label.setPreferredSize(new Dimension(120, 10));
		add(label, BorderLayout.WEST);
		add(comp, BorderLayout.CENTER);
	}

	public JLabel getLabel() {
		return label;
	}
	
}
