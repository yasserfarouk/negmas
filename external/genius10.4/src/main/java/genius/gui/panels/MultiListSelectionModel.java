package genius.gui.panels;

import javax.swing.DefaultListSelectionModel;

/**
 * Selection model for JList's which allows multiple elements to be selected
 * by toggling an item when clicked.
 * 
 * @author Mark Hendrikx (m.j.c.hendrikx@student.tudelft.nl)
 * @version 05/12/11
 */
public class MultiListSelectionModel extends DefaultListSelectionModel {
	private static final long serialVersionUID = 3815068403424389993L;
	private int i0 = -1;
	private int i1 = -1;

	public void setSelectionInterval(int index0, int index1) {
		if (i0 == index0 && i1 == index1) {
			if (getValueIsAdjusting()) {
				setValueIsAdjusting(false);
				setSelection(index0, index1);
			}
		} else {
			i0 = index0;
			i1 = index1;
			setValueIsAdjusting(false);
			setSelection(index0, index1);
		}
	}

	private void setSelection(int index0, int index1) {
		if (super.isSelectedIndex(index0)) {
			super.removeSelectionInterval(index0, index1);
		} else {
			super.addSelectionInterval(index0, index1);
		}
	}
}