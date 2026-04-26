package genius.gui.panels;

import java.util.ArrayList;
import java.util.List;

import javax.swing.ListSelectionModel;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;

import genius.core.listener.Listener;

/**
 * Adapts the {@link SubsetSelectionModel} to a {@link ListSelectionModel}.
 * 
 * The ListSelectionModel (jlist) supports quite complex selection modes:
 * ctrl-click , dragging, and combinations of these. We really only want to
 * support dragging so we ignore the ctrl- and removal stuff.
 */
public class SelectionModelAdapter<ItemType> implements ListSelectionModel {

	private SubsetSelectionModel<ItemType> model;

	/**
	 * drag start point. Valid while user is dragging
	 */
	private int anchorSelectionIndex = -1;
	/**
	 * current drag end point. Valid while user is dragging
	 */
	private int leadSelectionIndex = -1;
	/**
	 * True while the user is dragging.
	 */
	private boolean valueIsAdjusting;
	/**
	 * Selected selection mode.
	 */
	private int selectionMode;

	/**
	 * The selection in the model at the time the drag starts
	 */
	private List<ItemType> initialSelection;

	public SelectionModelAdapter(SubsetSelectionModel<ItemType> model) {
		this.model = model;

	}

	@Override
	public void setSelectionInterval(int index0, int index1) {
		if (anchorSelectionIndex == -1) {
			// user just started dragging
			anchorSelectionIndex = index0;
			initialSelection = model.getSelectedItems();
		}
		leadSelectionIndex = index1;
		reverseSelection();
	}

	/**
	 * Reverse the selection state of items in the range [anchor, lead]
	 */
	private void reverseSelection() {
		ArrayList<ItemType> newSelection = new ArrayList<ItemType>(initialSelection);
		for (int n = Math.min(anchorSelectionIndex, leadSelectionIndex); //
				n <= Math.max(anchorSelectionIndex, leadSelectionIndex); n++) {
			ItemType item = model.getAllItems().get(n);
			// reverse the selection of item.
			if (newSelection.contains(item)) {
				newSelection.remove(item);
			} else {
				newSelection.add(item);
			}
			model.select(newSelection);
		}

	}

	@Override
	public void addSelectionInterval(int index0, int index1) {
		System.out.println("ignore User added " + index0 + " " + index1);
	}

	@Override
	public void removeSelectionInterval(int index0, int index1) {
		System.out.println("ignore user removed " + index0 + " " + index1);
	}

	@Override
	public int getMinSelectionIndex() {
		if (model.getSelectedItems().isEmpty()) {
			return -1;
		}
		return model.getAllItems().indexOf(model.getSelectedItems().get(0));
	}

	@Override
	public int getMaxSelectionIndex() {
		return model.getAllItems().size() - 1;
	}

	@Override
	public boolean isSelectedIndex(int index) {
		return model.getSelectedItems().contains(model.getAllItems().get(index));
	}

	@Override
	public int getAnchorSelectionIndex() {
		return anchorSelectionIndex;
	}

	@Override
	public void setAnchorSelectionIndex(int index) {
		System.out.println("anchor set to " + index);
		anchorSelectionIndex = index;
	}

	@Override
	public int getLeadSelectionIndex() {
		return leadSelectionIndex;
	}

	@Override
	public void setLeadSelectionIndex(int index) {
		leadSelectionIndex = index;
	}

	@Override
	public void clearSelection() {
		model.select(new ArrayList<ItemType>());
	}

	@Override
	public boolean isSelectionEmpty() {
		return model.getSelectedItems().isEmpty();
	}

	@Override
	public void insertIndexInterval(int index, int length, boolean before) {
		System.out.println("insert index blabla" + index + length);
	}

	@Override
	public void removeIndexInterval(int index0, int index1) {
		// ???????
	}

	@Override
	public void setValueIsAdjusting(boolean valueIsAdjusting) {
		this.valueIsAdjusting = valueIsAdjusting;
		if (this.valueIsAdjusting) {
			// start drag
			anchorSelectionIndex = -1;
		}
	}

	@Override
	public boolean getValueIsAdjusting() {
		return valueIsAdjusting;
	}

	@Override
	public void setSelectionMode(int selectionMode) {
		this.selectionMode = selectionMode;
	}

	@Override
	public int getSelectionMode() {
		return selectionMode;
	}

	@Override
	public void addListSelectionListener(final ListSelectionListener x) {
		model.addListener(new ListSelectionListenerAdapter<ItemType>(x));
	}

	@Override
	public void removeListSelectionListener(ListSelectionListener x) {
		model.removeListener(new ListSelectionListenerAdapter<ItemType>(x));
	}

}

/**
 * adapts a {@link ListSelectionListener} to a {@link Listener}
 *
 * @param <ItemType>
 */
class ListSelectionListenerAdapter<ItemType> implements Listener<ItemType> {

	private ListSelectionListener listener;

	public ListSelectionListenerAdapter(ListSelectionListener x) {
		this.listener = x;
	}

	@Override
	public void notifyChange(Object data) {
		// our model does not report this kind of detail. Hack it.
		listener.valueChanged(new ListSelectionEvent(this, 0, 0, false));
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
		ListSelectionListenerAdapter other = (ListSelectionListenerAdapter) obj;
		if (listener == null) {
			if (other.listener != null)
				return false;
		} else if (!listener.equals(other.listener))
			return false;
		return true;
	}

}
