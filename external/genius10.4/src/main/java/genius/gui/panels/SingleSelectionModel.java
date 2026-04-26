package genius.gui.panels;

import java.util.ArrayList;
import java.util.List;

import javax.swing.DefaultComboBoxModel;
import javax.swing.event.ListDataListener;

/**
 * A model where the user can select a single item from a list of a given type.
 * This object can be listened for changes made by the user. Changes can also be
 * made programmatically.
 * 
 * <p>
 * To listen for selection changes, attach and use the callback in
 * {@link ListDataListener#contentsChanged(javax.swing.event.ListDataEvent)}.
 * 
 * @param <ItemType>
 *            the type of item that this model contains.
 */
@SuppressWarnings("serial")
public class SingleSelectionModel<ItemType> extends DefaultComboBoxModel<ItemType> {

	public SingleSelectionModel(List<ItemType> allItems) {
		setAllItems(allItems);
	}

	/**
	 * use new set of possible items.
	 * 
	 * @param allItems
	 */
	public void setAllItems(List<ItemType> allItems) {
		removeAllElements();
		for (ItemType item : allItems) {
			addElement(item);
		}
		if (!allItems.isEmpty()) {
			setSelectedItem(allItems.get(0));
		}
	}

	/**
	 * 
	 * @return all items that can be chosen.
	 */
	public List<ItemType> getAllItems() {
		ArrayList<ItemType> list = new ArrayList<ItemType>();
		for (int n = 0; n < getSize(); n++) {
			list.add(getElementAt(n));
		}
		return list;
	}

	/**
	 * Type-checked version of {@link #getSelectedItem()}
	 * 
	 * @return selected item.
	 */
	@SuppressWarnings("unchecked")
	public ItemType getSelection() {
		return (ItemType) getSelectedItem();
	}

	/**
	 * Select the next element in the list.
	 */
	public void increment() {
		int n = getIndexOf(getSelectedItem());
		if (n + 1 < getSize()) {
			setSelectedItem(getElementAt(n + 1));
		}
	}

}
