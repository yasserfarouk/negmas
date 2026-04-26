package genius.gui.panels;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Arrays;
import java.util.List;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.ListCellRenderer;

import genius.core.listener.Listener;

/**
 * {@link SubsetSelectionPanel} with additional border, indicator of number of
 * selected items, and clear button
 * 
 */
@SuppressWarnings("serial")
public class SubsetSelectionPanelPlus<ItemType> extends VflowPanelWithBorder {

	private JLabel infolabel = new JLabel("0 selected");
	private SubsetSelectionModel<ItemType> model;
	private SubsetSelectionPanel<ItemType> listpanel;

	public SubsetSelectionPanelPlus(final String title, final SubsetSelectionModel<ItemType> model) {

		super(title);
		this.model = model;
		updateInfo();

		JPanel panel = new JPanel(new BorderLayout());
		panel.setMaximumSize(new Dimension(999999999, 30));
		panel.add(infolabel, BorderLayout.WEST);
		JButton clearButton = new JButton("Clear");
		panel.add(clearButton, BorderLayout.EAST);

		listpanel = new SubsetSelectionPanel<>(model);
		add(new JScrollPane(listpanel));
		add(panel);

		model.addListener(new Listener<ItemType>() {
			@Override
			public void notifyChange(ItemType data) {
				updateInfo();
			}
		});

		clearButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				model.clear();
			}
		});

	}

	/**
	 * Set the cell renderer.
	 * 
	 * @param renderer
	 *            the renderer for list elements
	 * 
	 */
	public void setCellRenderer(ListCellRenderer<ItemType> renderer) {
		listpanel.setCellRenderer(renderer);
	}

	private void updateInfo() {
		// FIXME invokelater
		infolabel.setText("" + model.getSelectedItems().size() + " selected");
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
		gui.getContentPane().add(new SubsetSelectionPanelPlus<String>("Paneel", model), BorderLayout.CENTER);
		gui.pack();
		gui.setVisible(true);

	}

}
