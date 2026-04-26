package genius.gui.panels;

import java.util.ArrayList;
import java.util.List;

import javax.swing.AbstractListModel;
import javax.swing.ComboBoxModel;

/**
 * Extends the default ListModel by allowing it to be loaded afterwards with
 * data.
 * 
 * @author Mark Hendrikx (m.j.c.hendrikx@student.tudelft.nl)
 * @version 05/12/11
 */
public class ExtendedComboBoxModel<A> extends AbstractListModel<A> implements ComboBoxModel<A> {
	private static final long serialVersionUID = -8345719619830961700L;
	private List<A> items = new ArrayList<A>();
	private A selection;

	public void setInitialContent(List<A> items) {
		this.items = items;

	}

	public A getElementAt(int index) {
		if (index >= 0) {
			return items.get(index);
		}
		return null;
	}

	public int getSize() {
		return items.size();
	}

	public void removeElementAt(int i) {
		items.remove(i);
	}

	@Override
	public A getSelectedItem() {
		return selection;
	}

	@SuppressWarnings("unchecked")
	@Override
	public void setSelectedItem(Object anItem) {
		selection = (A) anItem;
	}
}