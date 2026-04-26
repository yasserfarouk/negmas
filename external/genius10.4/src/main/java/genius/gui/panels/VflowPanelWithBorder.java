package genius.gui.panels;

import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JPanel;

/**
 * vertical flow panel that has a border and a title.
 *
 */
@SuppressWarnings("serial")
public class VflowPanelWithBorder extends JPanel {
	public VflowPanelWithBorder(String title) {
		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
		setBorder(BorderFactory.createTitledBorder(title));
	}
}