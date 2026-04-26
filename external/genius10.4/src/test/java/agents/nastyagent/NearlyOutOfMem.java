package agents.nastyagent;

import java.util.ArrayList;
import java.util.List;

import genius.core.actions.Action;

/**
 * Deliberately goes out of memory while choosing an action.
 * 
 * @author W.Pasman 25jan16
 *
 */
public class NearlyOutOfMem extends NastyAgent {
	private ArrayList<String> memory = new ArrayList<String>();
	private final static String block = "01234567890123456789123456789012";

	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {
		doubleMemory();
		return super.chooseAction(possibleActions);
	}

	/**
	 * Double the amount of used memory (starts with 32 bytes if no mem in use
	 * yet) until only 1000 bytes are left.
	 */
	private void doubleMemory() {
		long toAdd = memory.size();
		if (toAdd == 0) {
			toAdd = 1;
		}
		for (long i = 0; i < toAdd; i++) {
			memory.add(block);
			if (Runtime.getRuntime().freeMemory() < 10000)
				return;
		}
	}
}
