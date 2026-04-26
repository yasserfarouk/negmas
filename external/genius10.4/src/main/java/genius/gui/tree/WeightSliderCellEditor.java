package genius.gui.tree;

import java.awt.*;

import javax.swing.*;
import javax.swing.table.*;

/**
*
* @author Richard Noorlandt
* 
*/

public class WeightSliderCellEditor extends AbstractCellEditor implements TableCellEditor, TableCellRenderer {

	private static final long serialVersionUID = -18549774953873547L;
	//Attributes
	NegotiatorTreeTableModel tableModel;
	
	//Constructors
	
	public WeightSliderCellEditor(NegotiatorTreeTableModel model) {
		super();
		tableModel = model;
	}
	
	//Methods
	public Object getCellEditorValue() {
		return null;
	}
	
	public Component getTableCellEditorComponent(JTable table, Object value, boolean isSelected, int row, int column) {
		return (WeightSlider)value;
	}
	
	public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int column) {
		return (WeightSlider)value;
	}
}
