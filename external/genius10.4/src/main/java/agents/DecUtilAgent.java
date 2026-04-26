package agents;

import java.util.ArrayList;
import java.util.Collections;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Offer;

/**
 * This agent just offers all bids in decreasing utility order to opponent. The
 * acceptance criterion is the same as with the SimpleAgent.
 * 
 * @author W.Pasman
 * 
 */
public class DecUtilAgent extends Agent {
	private Action actionOfPartner = null;
	ArrayList<Bid> bids = new ArrayList<Bid>();
	int nextBidIndex = 0; // what's the next bid from bids to be done.

	@Override
	public void init() {

		BidIterator biter = new BidIterator(utilitySpace.getDomain());
		while (biter.hasNext())
			bids.add(biter.next());
		Collections.sort(bids, new BidComparator(utilitySpace));
	}

	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
	}

	public Action chooseAction() {
		Action action = null;
		try {
			if (actionOfPartner == null)
				action = new Offer(getAgentID(), bids.get(nextBidIndex++));
			if (actionOfPartner instanceof Offer) {
				action = new Offer(getAgentID(), bids.get(nextBidIndex++));
			}
			// Thread.sleep(300); // 3 bids per second is good enough.
		} catch (Exception e) {
			System.out.println("Exception in ChooseAction:" + e.getMessage());
			// best guess if things go wrong.
			action = new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid());
		}
		return action;
	}

	/**
	 * This function determines the accept probability for an offer. At t=0 it
	 * will prefer high-utility offers. As t gets closer to 1, it will accept
	 * lower utility offers with increasing probability. it will never accept
	 * offers with utility 0.
	 * 
	 * @param u
	 *            is the utility
	 * @param t
	 *            is the time as fraction of the total available time (t=0 at
	 *            start, and t=1 at end time)
	 * @return the probability of an accept at time t
	 * @throws Exception
	 *             if you use wrong values for u or t.
	 * 
	 */
	double Paccept(double u, double t1) throws Exception {
		double t = t1 * t1 * t1; // steeper increase when deadline approaches.
		if (u < 0 || u > 1.05)
			throw new Exception("utility " + u + " outside [0,1]");
		// normalization may be slightly off, therefore we have a broad boundary
		// up to 1.05
		if (t < 0 || t > 1)
			throw new Exception("time " + t + " outside [0,1]");
		if (u > 1.)
			u = 1;
		if (t == 0.5)
			return u;
		return (u - 2. * u * t + 2. * (-1. + t + Math.sqrt(sq(-1. + t) + u
				* (-1. + 2 * t))))
				/ (-1. + 2 * t);
	}

	double sq(double x) {
		return x * x;
	}
}