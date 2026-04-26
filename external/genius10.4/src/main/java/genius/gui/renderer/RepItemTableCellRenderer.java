package genius.gui.renderer;

import java.awt.Component;

import javax.swing.JLabel;
import javax.swing.JTable;
import javax.swing.table.DefaultTableCellRenderer;

import genius.core.repository.PartyRepItem;
import genius.core.repository.RepItem;

/**
 * Renders RepItems, using {@link PartyRepItem#getName()} if possible, or
 * toString otherwise. Identical to {@link RepItemListCellRenderer} except that
 * this is for tables.
 */

@SuppressWarnings("serial")
public class RepItemTableCellRenderer extends DefaultTableCellRenderer {

	@Override
	public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus,
			int row, int column) {
		JLabel comp = (JLabel) super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, column);
		comp.setText(RepItemListCellRenderer.getText((RepItem) value));
		return comp;
	}
}
