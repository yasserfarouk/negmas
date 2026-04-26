package genius.gui.repository;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

import javax.swing.JFileChooser;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.filechooser.FileFilter;
import javax.swing.table.AbstractTableModel;

import genius.core.Global;
import genius.core.repository.PartyRepItem;
import genius.core.repository.Repository;
import genius.core.repository.RepositoryFactory;
import genius.gui.panels.GenericFileFilter;

/**
 * A user interface to for RepItems, usable for AgentRepItem and PartyRepItem
 * repository
 * 
 * @author Wouter Pasman
 */
@SuppressWarnings("serial")
public class PartyRepositoryUI extends JPanel {
	private partyTableModel dataModel;
	private final JTable table = new JTable();
	private JScrollPane scrollpane = new JScrollPane();

	public PartyRepositoryUI() {
		setLayout(new BorderLayout());

		scrollpane.setViewportView(table);
		add(scrollpane, BorderLayout.CENTER);

		dataModel = new partyTableModel();

		table.setModel(dataModel);
		table.setShowVerticalLines(false);

		table.addKeyListener(new KeyAdapter() {
			public void keyReleased(KeyEvent ke) {
				if (ke.getKeyCode() == KeyEvent.VK_DELETE) {
					removeAction();
				}
			}
		});

		addPopupMenu(scrollpane);
		addPopupMenu(table);

	}

	/**
	 * User clicked in our panel. Show popup menu
	 * 
	 * if there is a row selected in the table, we also enable the Remove
	 * option.
	 * 
	 * @return popupmenu
	 */
	private JPopupMenu createPopupMenu() {
		JPopupMenu popup = new JPopupMenu();

		JMenuItem addAgent = new JMenuItem("Add new party");
		addAgent.addActionListener(new java.awt.event.ActionListener() {
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				addAction();
			}
		});
		popup.add(addAgent);

		if (table.getSelectedRow() > 0) {
			JMenuItem removeAgent = new JMenuItem("Remove party");
			removeAgent.addActionListener(new java.awt.event.ActionListener() {
				public void actionPerformed(java.awt.event.ActionEvent evt) {
					removeAction();
				}
			});
			popup.add(removeAgent);
		}

		return popup;
	}

	/**
	 * Add a popup menu to a component. We need to attach this to multiple
	 * components: the scrollpane and the table. This is because we want the
	 * menu also to appear if users click outside the table #858
	 * 
	 * @param component
	 *            the component to add the menu to.
	 * 
	 */
	private void addPopupMenu(Component component) {
		component.addMouseListener(new MouseAdapter() {

			// if Windows
			@Override
			public void mouseReleased(MouseEvent e) {
				mouseCode(e);
			}

			// if Linux
			public void mousePressed(MouseEvent e) {
				mouseCode(e);
			}

			private void mouseCode(MouseEvent e) {
				int r = table.rowAtPoint(e.getPoint());
				if (r >= 0 && r < table.getRowCount()) {
					table.setRowSelectionInterval(r, r);
				} else {
					table.clearSelection();
				}

				if (e.isPopupTrigger()) {// && e.getComponent() instanceof
											// JTable) {
					// if rowindex>0, we actually selected a row.
					JPopupMenu popup = createPopupMenu();
					popup.show(e.getComponent(), e.getX(), e.getY());
				}
			}
		});
	}

	/**
	 * Add new agent to repository. The party is expected to be a .class file
	 */
	public void addAction() {

		JFileChooser fc = new JFileChooser(System.getProperty("user.dir"));

		// Filter such that only directories and .class files are shown.
		FileFilter filter = new GenericFileFilter("class", "Java class files (.class)");
		fc.setFileFilter(filter);

		// Open the file picker
		int returnVal = fc.showOpenDialog(null);

		// If file selected
		if (returnVal == JFileChooser.APPROVE_OPTION) {
			try {
				dataModel.add(new PartyRepItem(fc.getSelectedFile().getPath()));
			} catch (Throwable e) {
				Global.showLoadError(fc.getSelectedFile(), e);
			}
		}

	}

	public void removeAction() {
		// CHECK is this ok? After removing a row, all row numbers change?
		for (int i = 0; i < table.getSelectedRows().length; i++) {
			dataModel.remove(table.getSelectedRows()[i]);
		}
	}

}

/**
 * Contains the data model for the Party Repository.
 * 
 * @author W.Pasman 3aug15
 *
 */
@SuppressWarnings("serial")
class partyTableModel extends AbstractTableModel {

	private Repository repository;
	final String columnnames[] = { "Party Name", "Description", "Protocol" };

	public partyTableModel() {
		repository = RepositoryFactory.get_party_repository();

	}

	public void add(PartyRepItem agentref) {
		repository.getItems().add(agentref);
		repository.save();
		fireTableDataChanged();
	}

	/**
	 * Remove row from the model.
	 * 
	 * @param i
	 *            the row to remove
	 */
	public void remove(int i) {
		repository.getItems().remove(i);
		repository.save();
		fireTableDataChanged();
	}

	public int getColumnCount() {
		return columnnames.length;
	}

	public int getRowCount() {
		return repository.getItems().size();
	}

	public Object getValueAt(int row, int col) {
		try {
			PartyRepItem party = (PartyRepItem) repository.getItems().get(row);
			switch (col) {
			case 0:
				return party.getName();
			case 1:
				return party.getDescription();
			case 2:
				return Class.forName(party.getProtocolClassPath()).getSimpleName();
			}
			return col;
		} catch (Exception e) {
			return e.toString();
		}
	}

	public String getColumnName(int column) {
		return columnnames[column];
	}

}