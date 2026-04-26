package genius.gui.panels;

import java.awt.BorderLayout;
import java.util.Arrays;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.ListCellRenderer;

/**
 * A GUI to select a subset from a list of items. Just a bare list view. You
 * probably want to wrap this into a {@link JScrollPane}.
 */
@SuppressWarnings("serial")
public class SubsetSelectionPanel<ItemType> extends JPanel {
	private SubsetSelectionModel<ItemType> model;
	private JList<ItemType> list = new JList<ItemType>();

	public SubsetSelectionPanel(SubsetSelectionModel<ItemType> model) {
		this.model = model;
		initPanel();
	}

	/**
	 * Set basic panel contents: buttons, list area
	 */
	private void initPanel() {
		setLayout(new BorderLayout());
		list.setModel(new ListModelAdapter<>(model));
		list.setSelectionModel(new SelectionModelAdapter<ItemType>(model));
		add(list, BorderLayout.CENTER);

	}

	/**
	 * Set the cell renderer for the list items.
	 * 
	 * @param renderer
	 *            the cell renderer for the list items.
	 */
	public void setCellRenderer(ListCellRenderer<ItemType> renderer) {
		list.setCellRenderer(renderer);
	}

	/**
	 * simple stub to run this stand-alone (for testing).
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		final JFrame gui = new JFrame();
		gui.setLayout(new BorderLayout());
		List<String> allItems = Arrays.asList("een", "twee", "drie", "vier");
		SubsetSelectionModel<String> model = new SubsetSelectionModel<String>(allItems);
		gui.getContentPane().add(new SubsetSelectionPanel<String>(model), BorderLayout.CENTER);
		gui.pack();
		gui.setVisible(true);

	}
}
