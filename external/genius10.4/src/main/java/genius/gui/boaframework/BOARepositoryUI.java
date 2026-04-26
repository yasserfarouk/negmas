package genius.gui.boaframework;

import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Map.Entry;

import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;
import javax.swing.JTable;
import javax.swing.table.AbstractTableModel;

import genius.core.boaframework.BoaType;
import genius.core.boaframework.repository.BOAagentRepository;
import genius.core.boaframework.repository.BOArepItem;
import genius.gui.GeniusAppInterface;

/**
 * A user interface to the agent repository
 */
@SuppressWarnings("serial")
public class BOARepositoryUI extends JTable {

	private static final String ADD_A_COMPONENT = "Add a component";
	private final BOAagentRepository boaRepository;
	private AbstractTableModel dataModel;
	private ArrayList<BOArepItem> items;
	private final GeniusAppInterface genius;

	public BOARepositoryUI(GeniusAppInterface genius) {
		this.genius = genius;
		boaRepository = BOAagentRepository.getInstance();
		items = new ArrayList<BOArepItem>();
		referenceComponents();
		initTable();
	}

	private JPopupMenu createPopupMenu() {
		JPopupMenu popup = new JPopupMenu();

		JMenuItem addComponent = new JMenuItem("Add new component");
		addComponent.addActionListener(new java.awt.event.ActionListener() {
			@Override
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				addAction();
			}
		});

		JMenuItem editComponent = new JMenuItem("Edit component");
		editComponent.addActionListener(new java.awt.event.ActionListener() {
			@Override
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				editAction();
			}
		});

		JMenuItem removeComponent = new JMenuItem("Remove component");
		removeComponent.addActionListener(new java.awt.event.ActionListener() {
			@Override
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				removeAction();
			}
		});

		popup.add(addComponent);
		popup.add(editComponent);
		popup.add(removeComponent);

		return popup;
	}

	private void referenceComponents() {
		items.clear();
		for (Entry<String, BOArepItem> entry : boaRepository
				.getOfferingStrategiesRepItems().entrySet()) {
			items.add(entry.getValue());
		}

		for (Entry<String, BOArepItem> entry : boaRepository
				.getAcceptanceStrategiesRepItems().entrySet()) {
			items.add(entry.getValue());
		}

		for (Entry<String, BOArepItem> entry : boaRepository
				.getOpponentModelsRepItems().entrySet()) {
			items.add(entry.getValue());
		}

		for (Entry<String, BOArepItem> entry : boaRepository
				.getOMStrategiesRepItems().entrySet()) {
			items.add(entry.getValue());
		}

		Collections.sort(items);

		if (items.size() == 0) {
			addTemporaryComponent();
		}
	}

	private void initTable() {
		dataModel = new AbstractTableModel() {
			private static final long serialVersionUID = -4985008096999143587L;
			final String columnnames[] = { "Type", "Name" };

			@Override
			public int getColumnCount() {
				return columnnames.length;
			}

			@Override
			public int getRowCount() {
				return items.size();
			}

			@Override
			public Object getValueAt(int row, int col) {
				BOArepItem boaComponent = items.get(row);
				switch (col) {
				case 0:
					return boaComponent.getTypeString();
				case 1:
					return boaComponent.getName();
				}
				return col;
			}

			@Override
			public String getColumnName(int column) {
				return columnnames[column];
			}
		};

		setModel(dataModel);
		setShowVerticalLines(false);
		addMouseListener(new MouseAdapter() {

			// if Windows
			@Override
			public void mouseReleased(MouseEvent e) {
				mouseCode(e);
			}

			// if Linux
			@Override
			public void mousePressed(MouseEvent e) {
				mouseCode(e);
			}

			private void mouseCode(MouseEvent e) {
				int r = rowAtPoint(e.getPoint());
				if (r >= 0 && r < getRowCount()) {
					setRowSelectionInterval(r, r);
				} else {
					clearSelection();
				}

				int rowindex = getSelectedRow();
				if (rowindex < 0)
					return;
				if (e.isPopupTrigger() && e.getComponent() instanceof JTable) {
					JPopupMenu popup = createPopupMenu();
					popup.show(e.getComponent(), e.getX(), e.getY());
				}
			}
		});

		addKeyListener(new KeyAdapter() {
			@Override
			public void keyReleased(KeyEvent ke) {
				if (ke.getKeyCode() == KeyEvent.VK_DELETE) {
					removeAction();
				}
			}
		});
	}

	public void addAction() {
		// shoud return boolean if added an item.
		// if so, sort items and display again
		BOAComponentEditor loader = new BOAComponentEditor(
				genius.getMainFrame(), "Add BOA component");
		BOArepItem item = loader.getResult(null);
		if (item != null) {
			items.add(item);
			Collections.sort(items);
			updateUI();
		}
	}

	public void editAction() {
		BOArepItem item = items.get(getSelectedRow());
		BOAComponentEditor loader = new BOAComponentEditor(
				genius.getMainFrame(), "Edit BOA component");
		BOArepItem result = loader.getResult(item);
		if (result != null) {
			items.remove(item);
			items.add(result);
			Collections.sort(items);
			updateUI();
		}
	}

	public void removeAction() {
		if (getSelectedRow() != -1) {
			BOArepItem removed = items.remove(getSelectedRow());
			if (dataModel.getRowCount() == 0) {
				addTemporaryComponent();
			}
			dataModel.fireTableDataChanged();
			if (removed.getType() != BoaType.UNKNOWN) {
				boaRepository.removeComponent(removed);
			}
		}
	}

	private void addTemporaryComponent() {
		if (items.size() == 0) {
			items.add(new BOArepItem(ADD_A_COMPONENT, "", BoaType.UNKNOWN));
		}
	}
}