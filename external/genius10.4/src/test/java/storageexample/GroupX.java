package storageexample;

import java.util.List;

import java.util.HashMap;
import java.util.Map;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.list.Tuple;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.PersistentDataType;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;

/**
 * Sample party that accepts the Nth offer, where N is the number of sessions
 * this [agent-profile] already did.
 */
public class GroupX extends AbstractNegotiationParty {

	private Bid lastReceivedBid = null;
	private int nrChosenActions = 0; // number of times chosenAction was called.
	private StandardInfoList history;

	@Override
	public void init(NegotiationInfo info) {

		super.init(info);

		System.out.println("Discount Factor is " + getUtilitySpace().getDiscountFactor());
		System.out.println("Reservation Value is " + getUtilitySpace().getReservationValueUndiscounted());

		if (getData().getPersistentDataType() != PersistentDataType.STANDARD) {
			throw new IllegalStateException("need standard persistent data");
		}
		history = (StandardInfoList) getData().get();

		if (!history.isEmpty()) {
			// example of using the history. Compute for each party the maximum
			// utility of the bids in last session.
			Map<String, Double> maxutils = new HashMap<String, Double>();
			StandardInfo lastinfo = history.get(history.size() - 1);
			for (Tuple<String, Double> offered : lastinfo.getUtilities()) {
				String party = offered.get1();
				Double util = offered.get2();
				maxutils.put(party, maxutils.containsKey(party) ? Math.max(maxutils.get(party), util) : util);
			}
			System.out.println(maxutils); // notice tournament suppresses all
											// output.
		}
	}

	public Action chooseAction(List<Class<? extends Action>> validActions) {
		nrChosenActions++;
		if (nrChosenActions > history.size() & lastReceivedBid != null) {
			return new Accept(getPartyId(), lastReceivedBid);
		} else {
			return new Offer(getPartyId(), generateRandomBid());
		}
	}

	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		if (action instanceof Offer) {
			lastReceivedBid = ((Offer) action).getBid();
		}
	}

	public String getDescription() {
		return "accept Nth offer";
	}

}
