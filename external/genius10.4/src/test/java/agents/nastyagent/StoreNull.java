package agents.nastyagent;

import java.util.HashMap;

import genius.core.Bid;

/**
 * mangles the map
 *
 */
public class StoreNull extends NastyAgent {
	@Override
	public HashMap<String, String> negotiationEnded(Bid acceptedBid) {
		// we actually got here. Report it.
		super.negotiationEnded(acceptedBid);
		data.put(null);
		return null;
	}

}
