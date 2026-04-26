package genius.gui.panels;

import javax.swing.ListModel;
import javax.swing.event.ListDataListener;

/**
 * Adapter to map {@link SubsetSelectionModel} to {@link ListModel}
 *
 * @param <ItemType>
 */
public class ListModelAdapter<ItemType> implements ListModel<ItemType> {

	private SubsetSelectionModel<ItemType> model;

	public ListModelAdapter(SubsetSelectionModel<ItemType> model) {
		this.model = model;
	}

	@Override
	public int getSize() {
		return model.getAllItems().size();
	}

	@Override
	public ItemType getElementAt(int index) {
		return model.getAllItems().get(index);
	}

	@Override
	public void addListDataListener(ListDataListener l) {
		model.addListener(new ListDataListenerAdapter<ItemType>(l));
	}

	@Override
	public void removeListDataListener(ListDataListener l) {
		model.removeListener(new ListDataListenerAdapter<ItemType>(l));
	}

}