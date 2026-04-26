package genius.gui.panels;

import javax.swing.event.ListDataListener;

import genius.core.listener.Listener;

/**
 * Adapts a {@link Listener} into a {@link ListDataListener}.
 * 
 * @param <ItemType>
 */
public class ListDataListenerAdapter<ItemType> implements Listener<ItemType> {

	private ListDataListener listener;

	public ListDataListenerAdapter(ListDataListener l) {
		this.listener = l;
	}

	@Override
	public void notifyChange(Object data) {
		listener.contentsChanged(null);
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
		ListDataListenerAdapter other = (ListDataListenerAdapter) obj;
		if (listener == null) {
			if (other.listener != null)
				return false;
		} else if (!listener.equals(other.listener))
			return false;
		return true;
	}

}
