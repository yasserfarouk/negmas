package genius.gui.tree;

import java.awt.*;
import javax.swing.*;
import javax.swing.table.*;

/**
*
* @author Richard Noorlandt
* 
*/

public class IssueValueCellEditor extends AbstractCellEditor implements TableCellEditor, TableCellRenderer {

	private static final long serialVersionUID = -7575662197085471060L;
	//Attributes
	NegotiatorTreeTableModel tableModel;
	
	//Constructors
	public IssueValueCellEditor(NegotiatorTreeTableModel model) {
		super();
		tableModel = model;
	}
	
	//Methods
	public Object getCellEditorValue() {
		//TODO TEST CODE!!
		return new Integer(7);
	}
	
	public Component getTableCellEditorComponent(JTable table, Object value, boolean isSelected, int row, int column) {
		return (IssueValuePanel)value;
	}
	
	public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int column) {
		return (IssueValuePanel)value;
	}
}
