package genius.gui.boaparties;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import javax.swing.JTable;
import javax.swing.event.TableModelEvent;
import javax.swing.event.TableModelListener;
import javax.swing.table.TableModel;

import genius.core.boaframework.BOAparameter;
import genius.core.repository.boa.ParameterList;
import genius.core.repository.boa.ParameterRepItem;

/**
 * Holds set of BOA parameters as set by the user. Ready for use in a
 * {@link JTable}. This is mutable because it's a model.
 */
public class BoaParametersModel implements TableModel {

	private ArrayList<BOAparameter> parameters;

	private static final String[] columns = { "Name", "Description", "Lower bound", "Step size", "Upper bound" };

	/**
	 * listeners of the TableModel.
	 */
	private List<TableModelListener> listeners = new ArrayList<>();

	public BoaParametersModel(Set<BOAparameter> params) {
		parameters = new ArrayList<>(params);
	}

	/**
	 * Replace all the parameters in the model with a new set.
	 * 
	 * @param parameters2
	 */
	public void setParameters(Set<BOAparameter> parameters2) {
		this.parameters = new ArrayList<>(parameters2);
		notifyListeners(0);
	}

	@Override
	public int getRowCount() {
		return parameters.size();
	}

	@Override
	public int getColumnCount() {
		return columns.length;
	}

	@Override
	public String getColumnName(int columnIndex) {
		return columns[columnIndex];
	}

	@Override
	public Class<?> getColumnClass(int columnIndex) {
		return columnIndex < 2 ? String.class : Double.class;
	}

	@Override
	public boolean isCellEditable(int rowIndex, int columnIndex) {
		return columnIndex > 0;
	}

	@Override
	public Object getValueAt(int rowIndex, int columnIndex) {
		BOAparameter param = parameters.get(rowIndex);
		switch (columnIndex) {
		case 0:
			return param.getName();
		case 1:
			return param.getDescription();
		case 2:
			return param.getLow();
		case 3:
			return param.getStep();
		case 4:
			return param.getHigh();
		}
		return "???";

	}

	@Override
	public void setValueAt(Object aValue, int rowIndex, int columnIndex) {
		// System.out.println("User changed cell (" + rowIndex + "," +
		// columnIndex + ") to " + aValue);
		BOAparameter param = parameters.get(rowIndex);
		switch (columnIndex) {
		case 1:
			param = param.withDescription((String) aValue);
			break;
		case 2:
			param = param.withLow((Double) aValue);
			break;
		case 3:
			param = param.withStep((Double) aValue);
			break;
		case 4:
			param = param.withHigh((Double) aValue);
			break;
		}
		setParameter(rowIndex, param);
	}

	private void setParameter(int index, BOAparameter param) {
		System.out.println("new param for row" + index + ":" + param);
		parameters.set(index, param);
		notifyListeners(index);
	}

	/**
	 * Notify listeners of changed row
	 * 
	 * @param row
	 */
	private void notifyListeners(int row) {
		for (TableModelListener listener : listeners) {
			listener.tableChanged(new TableModelEvent(this, row));
		}
	}

	@Override
	public void addTableModelListener(TableModelListener l) {
		listeners.add(l);
	}

	@Override
	public void removeTableModelListener(TableModelListener l) {
		listeners.remove(l);
	}

	/**
	 * @return current settings of this model. A whole list of
	 *         {@link ParameterList}s is returned, one for each configuration
	 *         asked by the user.
	 */
	public List<ParameterList> getSettings() {
		return getSettings(parameters);
	}

	/**
	 * @param boaparams
	 *            Set of {@link BOAparameter}s with set of ranges for each
	 *            param.
	 * @return full product of all possible parameter combinations. If there are
	 *         no parameters at all, return a list with a single empty
	 *         ParameterList.
	 */
	protected static List<ParameterList> getSettings(List<BOAparameter> boaparams) {
		if (boaparams.isEmpty()) {
			ArrayList<ParameterList> list = new ArrayList<ParameterList>();
			list.add(new ParameterList());
			return list;
		}

		List<ParameterList> allParams = new ArrayList<>();
		BOAparameter boaparam1 = boaparams.get(0);
		List<ParameterList> allSubSettings = getSettings(boaparams.subList(1, boaparams.size()));

		BigDecimal endValue = BigDecimal.valueOf(boaparam1.getHigh());
		BigDecimal step = BigDecimal.valueOf(boaparam1.getStep());
		for (BigDecimal value = BigDecimal.valueOf(boaparam1.getLow()); value.compareTo(endValue) != 1; value = value
				.add(step)) {
			ParameterRepItem newParam = new ParameterRepItem(boaparam1.getName(), value.doubleValue());
			for (ParameterList subSetting : allSubSettings) {
				allParams.add(subSetting.include(newParam));
			}
		}

		return allParams;
	}

	@Override
	public String toString() {
		return parameters.toString();
	}

	/**
	 * @return current setting as set of {@link BOAparameter}s
	 */
	public ArrayList<BOAparameter> getSetting() {
		return parameters;
	}

}
