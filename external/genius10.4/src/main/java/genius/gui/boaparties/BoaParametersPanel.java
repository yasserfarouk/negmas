package genius.gui.boaparties;

import java.awt.BorderLayout;
import java.awt.Dimension;

import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;

/**
 * Editor for BOA parameters
 */
@SuppressWarnings("serial")
public class BoaParametersPanel extends JPanel {
	private BoaParametersModel model;

	public BoaParametersPanel(BoaParametersModel model) {
		this.model = model;
		setLayout(new BorderLayout());
		JScrollPane scrollpane = new JScrollPane();
		JTable table = new JTable(model);
		scrollpane.setPreferredSize(new Dimension(600, 200));
		scrollpane.getViewport().add(table);
		add(scrollpane, BorderLayout.CENTER);
	}

}
