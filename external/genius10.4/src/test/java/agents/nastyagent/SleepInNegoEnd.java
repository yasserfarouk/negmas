package agents.nastyagent;

import java.util.HashMap;

import genius.core.Bid;

/**
 * sleeps when negotiationEnded is called.
 *
 */
public class SleepInNegoEnd extends NastyAgent {
	@Override
	public HashMap<String, String> negotiationEnded(Bid acceptedBid) {
		// we actually got here. Report it.
		super.negotiationEnded(acceptedBid);
		try {
			Thread.sleep(99999999);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		return null;
	}
}
