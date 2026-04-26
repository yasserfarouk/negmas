package genius.gui.progress;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import javax.swing.table.AbstractTableModel;

/**
 * Table model for showing results that takes {@link DataKey}s to enhance type
 * checking and support conversion and smart data inserting. This code
 * determines how data actually is ordered for display. Only adding of rows
 * (containing one result) is supported.
 * 
 * @author W.Pasman 16jul15
 *
 */
public class DataKeyTableModel extends AbstractTableModel {
	/**
	 * 
	 */
	private static final long serialVersionUID = 49325296249290419L;

	/**
	 * The column specification - the Integer is the multiplicity of each of the
	 * columns. Note that the default order of the keys is determined by the
	 * order of DataKey.
	 */
	private final LinkedHashMap<DataKey, Integer> columnSpec;

	/**
	 * name of the rows
	 */
	private final List<String> columns = new ArrayList<String>();
	private final List<List<Object>> rows = new ArrayList<List<Object>>();

	/**
	 * @param colspec
	 *            a {@link TreeMap} containing tuples [DataKey,Integer] that
	 *            specifies the columns for the table. The DataKey is the column
	 *            to show, the Integer is the maximum multiplicity of that
	 *            column. When a column multiplicity has been set to a value
	 *            &gt;1, and a row is added that has a {@link List} for that
	 *            {@link DataKey}, then the list is expanded to separate
	 *            columns. All multiplicities must be &ge; 1.
	 * 
	 *            The order of the DataKeys determines the order of the columns.
	 */
	public DataKeyTableModel(LinkedHashMap<DataKey, Integer> colspec) {
		super();
		if (colspec.isEmpty()) {
			throw new IllegalArgumentException("table contains no columns");
		}
		columnSpec = colspec;
		initColumns();
	}

	/**
	 * Get the real column names based on the {@link DataKey} and multiplicity
	 * of the {@link #columnSpec}.
	 */
	private void initColumns() {
		for (DataKey key : columnSpec.keySet()) {
			int multiplicity = columnSpec.get(key);
			if (multiplicity < 1) {
				throw new IllegalArgumentException("column multiplicity must be >=1, found " + multiplicity);
			}
			if (multiplicity > 1) {
				for (int n = 1; n <= multiplicity; n++) {
					columns.add(key.getName() + " " + n);
				}
			} else {
				columns.add(key.getName());
			}
		}
	}

	/**
	 * Adds a row with new values. The values from the [key,value] pairs in the
	 * map are inserted in the correct columns given the headers. When a column
	 * multiplicity has been set to a value &gt;1 for {@link DataKey} K, and the
	 * value for K in the given map is a {@link List}, then the list is expanded
	 * to separate columns.
	 * 
	 * @param newValues
	 *            a {@link Map} with DataKeys and objects as values.
	 */
	public void addRow(Map<DataKey, Object> newValues) {
		List<Object> row = new ArrayList<Object>();

		for (DataKey key : columnSpec.keySet()) {
			int multiplicity = columnSpec.get(key);
			row.addAll(makeColumns(multiplicity, newValues.get(key)));
		}
		rows.add(row);

		fireTableRowsInserted(rows.size() - 1, rows.size() - 1);
	}

	/**
	 * Converts an object into exactly #multiplicity objects. if
	 * multiplicity==1, the returned list contains just that object. if
	 * multiplicity&gt;1 but the object is not a list, the list contains the
	 * object as first element and filled out with empty elements.
	 * 
	 * If object is a List, then the columns are filled with subsequential
	 * elements from the list. Too short lists lead to empty columns. Too long
	 * lists results in dropped elements.
	 * 
	 * @param multiplicity
	 * @param object
	 * @return list with object as first element, or with the contents of object
	 *         if object is a list
	 */
	private ArrayList<Object> makeColumns(int multiplicity, Object object) {
		List<Object> objList = null;
		if (object instanceof List) {
			objList = (List<Object>) object;
		}

		ArrayList<Object> result = new ArrayList<Object>();
		if (multiplicity == 1) {
			result.add(object);
			return result;
		}
		for (int n = 0; n < multiplicity; n++) {
			if (objList == null) {
				if (n == 0) {
					result.add(object);
				} else {
					result.add("");
				}
			} else {
				// object is a list.
				if (n >= objList.size()) {
					result.add("");
				} else {
					result.add(objList.get(n));
				}
			}
		}
		return result;
	}

	public int getColumnCount() {
		return columns.size();
	}

	public int getRowCount() {
		return rows.size();
	}

	public String getColumnName(int col) {
		return columns.get(col);
	}

	public Object getValueAt(int row, int col) {
		return rows.get(row).get(col);
	}

}
