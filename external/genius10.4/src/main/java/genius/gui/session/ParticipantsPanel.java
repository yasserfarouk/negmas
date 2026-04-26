package genius.gui.session;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.event.ListDataEvent;
import javax.swing.event.ListDataListener;
import javax.swing.event.TableModelListener;
import javax.swing.table.TableModel;

import genius.core.repository.RepItem;
import genius.core.session.Participant;
import genius.gui.panels.VflowPanelWithBorder;
import genius.gui.renderer.RepItemTableCellRenderer;

/**
 * An editor for a list of participants.
 *
 */
@SuppressWarnings("serial")
public class ParticipantsPanel extends VflowPanelWithBorder {

	public ParticipantsPanel(final ParticipantModel participantModel,
			final DefaultListModel<Participant> participantsModel) {
		super("Participants");

		add(new ParticipantPanel(participantModel));

		JPanel buttonsPanel = new JPanel(new BorderLayout());
		JButton addButton = new JButton("Add Party");
		JButton removeButton = new JButton("Remove Party");
		buttonsPanel.add(addButton, BorderLayout.WEST);
		buttonsPanel.add(removeButton, BorderLayout.EAST);
		buttonsPanel.setMaximumSize(new Dimension(99999999, 30));

		add(buttonsPanel);

		final JTable table = new JTable(new ParticipantTableAdapter(participantsModel));
		table.setDefaultRenderer(RepItem.class, new RepItemTableCellRenderer());
		add(new JScrollPane(table));

		addButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				participantsModel.addElement(participantModel.getParticipant());
				participantModel.increment();
			}
		});

		removeButton.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				if (table.getSelectedRow() != -1) {
					Participant participant = participantsModel.getElementAt(table.getSelectedRow());
					participantsModel.removeElement(participant);
				}
			}
		});
	}

}

/******************************************************************************************
 ************************************ INTERNAL ADAPTERS ***********************************
 ******************************************************************************************/
/**
 * Adapts the DefaultListModel<Participant> to show it in a 3-column table.
 */
class ParticipantTableAdapter implements TableModel {

	private DefaultListModel<Participant> model;

	public ParticipantTableAdapter(DefaultListModel<Participant> participantsModel) {
		this.model = participantsModel;
	}

	@Override
	public int getRowCount() {
		return model.getSize();
	}

	@Override
	public int getColumnCount() {
		return 3;
	}

	private static final String[] columns = { "Party ID", "Strategy", "Preference" };

	@Override
	public String getColumnName(int columnIndex) {
		return columns[columnIndex];
	}

	@Override
	public Class<?> getColumnClass(int columnIndex) {
		if (columnIndex == 0)
			return String.class;
		return RepItem.class;
	}

	@Override
	public boolean isCellEditable(int rowIndex, int columnIndex) {
		return false;
	}

	@Override
	public Object getValueAt(int rowIndex, int columnIndex) {
		Participant party = model.get(rowIndex);
		switch (columnIndex) {
		case 0:
			return party.getId();
		case 1:
			return party.getStrategy();
		case 2:
			return party.getProfile();
		}
		return "??";
	}

	@Override
	public void setValueAt(Object aValue, int rowIndex, int columnIndex) {
	}

	@Override
	public void addTableModelListener(TableModelListener l) {
		model.addListDataListener(new ListToTableListenerAdapter(l));
	}

	@Override
	public void removeTableModelListener(TableModelListener l) {
		model.removeListDataListener(new ListToTableListenerAdapter(l));
	}

}

/**
 * Bridges from ListDataListener to TableModelListeners so that a table can
 * listen to changes in a ListDataModel.
 *
 */
class ListToTableListenerAdapter implements ListDataListener {

	private TableModelListener tableListener;

	public ListToTableListenerAdapter(TableModelListener l) {
		tableListener = l;
	}

	@Override
	public void intervalAdded(ListDataEvent e) {
		tableListener.tableChanged(null);
	}

	@Override
	public void intervalRemoved(ListDataEvent e) {
		tableListener.tableChanged(null);
	}

	@Override
	public void contentsChanged(ListDataEvent e) {
		tableListener.tableChanged(null);
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((tableListener == null) ? 0 : tableListener.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		ListToTableListenerAdapter other = (ListToTableListenerAdapter) obj;
		if (tableListener == null) {
			if (other.tableListener != null)
				return false;
		} else if (!tableListener.equals(other.tableListener))
			return false;
		return true;
	}

}
