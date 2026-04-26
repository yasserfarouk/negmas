package genius.gui.progress.session;

import javax.swing.event.ListDataEvent;
import javax.swing.event.ListDataListener;
import javax.swing.event.TableModelEvent;
import javax.swing.event.TableModelListener;
import javax.swing.table.TableModel;

/**
 * Adapts the OutcomesModel to a table
 */
public class OutcomesModelToTableModelAdapter implements TableModel {

	private OutcomesListModel model;

	public OutcomesModelToTableModelAdapter(OutcomesListModel model) {
		this.model = model;
	}

	@Override
	public int getRowCount() {
		return model.getSize();
	}

	@Override
	public int getColumnCount() {
		// column 1 is for the round number
		return 1 + model.getParties().size();
	}

	@Override
	public String getColumnName(int columnIndex) {
		if (columnIndex == 0) {
			return "Round - Turn";
		}
		return model.getParties().get(columnIndex - 1).getID().toString();
	}

	@Override
	public Class<?> getColumnClass(int columnIndex) {
		return columnIndex == 0 ? String.class : Double.class;
	}

	@Override
	public boolean isCellEditable(int rowIndex, int columnIndex) {
		return false;
	}

	@Override
	public Object getValueAt(int rowIndex, int columnIndex) {
		Outcome outcome = model.get(rowIndex);
		if (columnIndex == 0) {
			return outcome.getRound() + "-" + outcome.getTurn();
		}
		return outcome.getDiscountedUtilities().get(columnIndex - 1);
	}

	@Override
	public void setValueAt(Object aValue, int rowIndex, int columnIndex) {
	}

	@Override
	public void addTableModelListener(TableModelListener l) {
		model.addListDataListener(new myListenerAdapter(l, this));
	}

	@Override
	public void removeTableModelListener(TableModelListener l) {
		model.removeListDataListener(new myListenerAdapter(l, this));

	}

}

/**
 * Adapts a ListDataListener to a TableModelListener
 */
class myListenerAdapter implements ListDataListener {

	private TableModelListener listener;
	private TableModel model;

	public myListenerAdapter(TableModelListener l, TableModel model) {
		this.listener = l;
		this.model = model;
	}

	@Override
	public void intervalAdded(ListDataEvent e) {
		listener.tableChanged(new TableModelEvent(model, e.getIndex0(), e.getIndex1(), TableModelEvent.ALL_COLUMNS,
				TableModelEvent.INSERT));
	}

	@Override
	public void intervalRemoved(ListDataEvent e) {
		// not supported. Just repaint all.
		listener.tableChanged(null);
	}

	@Override
	public void contentsChanged(ListDataEvent e) {
		listener.tableChanged(new TableModelEvent(null, e.getIndex0(), e.getIndex1(), TableModelEvent.ALL_COLUMNS,
				TableModelEvent.INSERT));
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((listener == null) ? 0 : listener.hashCode());
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
		myListenerAdapter other = (myListenerAdapter) obj;
		if (listener == null) {
			if (other.listener != null)
				return false;
		} else if (!listener.equals(other.listener))
			return false;
		return true;
	}

}