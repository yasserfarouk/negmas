package genius.core.repository.boa;

import java.util.ArrayList;

@SuppressWarnings({ "serial", "rawtypes" })
public class BoaRepItemList<T extends BoaRepItem> extends ArrayList<T> {
	/**
	 * @param name
	 *            name of item
	 * @return item that has given name, or null if no such item
	 */
	public T getName(String name) {
		for (T repItem : this) {
			if (repItem.getName().equals(name)) {
				return repItem;
			}
		}
		return null;
	}

	/**
	 * @param name
	 *            name of item
	 * @return item that has given name, or null if no such item
	 */

	public T getClassname(String name) {
		for (T repItem : this) {
			if (repItem.getClassPath().equals(name)) {
				return repItem;
			}
		}
		return null;
	}
}
