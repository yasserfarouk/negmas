package genius.gui.panels;

import java.util.ArrayList;
import java.util.List;

import genius.core.listener.DefaultListenable;

/**
 * A model for a list of Objects of given ItemType where the user can select a
 * subset. Used for {@link SubsetSelectionPanel}. We use {@link List} here so
 * that the order of the elements will not change (this might screw up AWT).
 *
 * @param <ItemType>
 *            the type of item that this model contains.
 */
public class SubsetSelectionModel<ItemType> extends DefaultListenable<ItemType> {
	/**
	 * All available items that can be chosen
	 */
	private List<ItemType> allItems;
	private List<ItemType> selectedItems;

	public SubsetSelectionModel(List<ItemType> allItems) {
		setAllItems(allItems);
	}

	public void setAllItems(List<ItemType> allItems) {
		this.allItems = allItems;
		clear();
	}

	/**
	 * @param items
	 *            the new selection
	 */
	public void select(List<ItemType> items) {
		// we might check if the selection is actually subset of allItems...
		selectedItems = items;
		notifyChange(null);
	}

	/**
	 * Clear the selection
	 */
	public void clear() {
		select(new ArrayList<ItemType>());
	}

	/**
	 * Remove items from the selection
	 * 
	 * @param items
	 */
	public void remove(List<ItemType> items) {
		selectedItems.removeAll(items);
		notifyChange(null);
	}

	public List<ItemType> getAllItems() {
		return allItems;
	}

	public List<ItemType> getSelectedItems() {
		return selectedItems;
	}

}
