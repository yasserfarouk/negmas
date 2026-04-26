package genius.core.list;

import java.util.ArrayList;

/**
 * Adapts ImmutableList to java.util.List. But it stays immutable. Supports
 * quick adapter to old code. fetches all items from the old list. Try to avoid
 * this and use {@link ImmutableList} all the way.
 */
@SuppressWarnings("serial")
public class JavaList<E> extends ArrayList<E> {

	public JavaList(ImmutableList<E> list) {
		for (E item : list) {
			add(item);
		}
	}

}
