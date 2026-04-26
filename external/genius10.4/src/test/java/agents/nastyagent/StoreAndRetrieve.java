package agents.nastyagent;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Action;

/**
 * Checks if data is retained. This uses a hack. This agent must be used in only
 * 1 tournament.
 *
 * 
 */
public class StoreAndRetrieve extends NastyAgent {
	private static final String DATA = "data";

	static Set<String> runBefore = new HashSet<String>();

	@Override
	public void receiveMessage(AgentID sender, Action action) {
		this.lastReceivedAction = action;

		String utilSpaceName = utilitySpace.getFileName();

		if (!runBefore.contains(utilSpaceName)) {
			if (data.get() != null) {
				throw new IllegalStateException("Not run before but having data!");
			}
			data.put(DATA);
			runBefore.add(utilSpaceName);
		} else {
			if (!DATA.equals(data.get())) {
				throw new IllegalStateException("Data has not been retained!");
			}
		}

	}

	@Override
	public HashMap<String, String> negotiationEnded(Bid acceptedBid) {
		// we actually got here. Report it.
		super.negotiationEnded(acceptedBid);

		return null;
	}

}
