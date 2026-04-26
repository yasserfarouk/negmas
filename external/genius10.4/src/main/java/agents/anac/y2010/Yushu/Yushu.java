package agents.anac.y2010.Yushu;

import java.util.Collections;
import java.util.Date;
import java.util.LinkedList;
import java.util.Random;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.SupportedNegotiationSetting;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.bidding.BidDetailsStrictSorterUtility;

/**
 * ANAC2010 competitor Yushu.
 */
public class Yushu extends Agent {
	private double eagerness = 1.2;
	private Action actionOfOpponent = null;
	private Bid myLastBid = null;
	private LinkedList<Bid> opponentHistory; // LinkedHashMap<Date,Bid>
	private LinkedList<Double> opponentUtis;
	private LinkedList<Integer> bestTenBids; // decreasing
	private LinkedList<Bid> myHistory;
	private LinkedList<Double> resptimes;
	Bid suggestBid = null; // the bid suggested based on op's bids
	Bid BESTBID = null;
	double topponentBidU = 0;
	double roundleft;
	Date rstime = null; // the time when I send out a
	double HPOSSIBLEU = 0;// the highest utility that can be achieved
	double ACCEPTABLEU = 1; // if oppennent's utility is higher than this,
							// accept it
	double previousTime = 0;
	Random random100;
	Random random200;
	private final boolean TEST_EQUIVALENCE = false;

	@Override
	public void init() {
		// if(utilitySpace.getReservationValue()!=null)
		// Utility.MINIMUM_BID_UTILITY = utilitySpace.getReservationValue();
		actionOfOpponent = null;
		myLastBid = null;
		opponentHistory = new LinkedList<Bid>();
		opponentUtis = new LinkedList<Double>();
		myHistory = new LinkedList<Bid>();
		resptimes = new LinkedList<Double>();
		suggestBid = null;
		bestTenBids = new LinkedList<Integer>();
		topponentBidU = 0;
		rstime = null;
		ACCEPTABLEU = 1;
		if (TEST_EQUIVALENCE) {
			random100 = new Random(100);
			random200 = new Random(200);
		} else {
			random100 = new Random();
			random200 = new Random();
		}
		try {
			BESTBID = utilitySpace.getMaxUtilityBid();
			HPOSSIBLEU = utilitySpace.getUtility(BESTBID);
		} catch (Exception e) {
			HPOSSIBLEU = 1;
		}
	}

	@Override
	public String getVersion() {
		return "2";
	}

	@Override
	public String getName() {
		return "Yushu";
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		actionOfOpponent = opponentAction;
	}

	@Override
	public Action chooseAction() {
		double targetuti = -1;
		resptimes.add(timeline.getTime() - previousTime);
		previousTime = timeline.getTime();
		Action action = null;
		try {
			if (actionOfOpponent == null) {
				Bid initialbid = BESTBID;
				action = new Offer(getAgentID(), initialbid);
			} else if (actionOfOpponent instanceof Offer) {
				Bid opponentBid = ((Offer) actionOfOpponent).getBid();
				double utiop = utilitySpace.getUtility(opponentBid);
				this.opponentHistory.add(opponentBid);
				this.opponentUtis.add(utiop);
				topponentBidU = topponentBidU + utiop;
				updateBelief(opponentBid, utiop);

				if (this.myLastBid == null)
					targetuti = 1;
				else
					targetuti = getTargetUtility();

				boolean accept = false;
				if ((utiop >= targetuti) | (utiop >= ACCEPTABLEU)) {

					if (this.suggestBid == null)
						accept = true;
					if (roundleft < 8)
						accept = true;
					if ((this.suggestBid != null) & (roundleft > 8)) {
						double utit = utilitySpace.getUtility(this.suggestBid);
						if (utit <= utiop)
							accept = true;
					}
				}
				if (accept) {
					action = new Accept(getAgentID(), opponentBid);

				} else {
					Bid mynextBid;
					if (this.suggestBid != null) {
						mynextBid = this.suggestBid;
					} else {
						if (targetuti >= HPOSSIBLEU)
							mynextBid = utilitySpace.getMaxUtilityBid();
						else {
							mynextBid = getNextBid(targetuti);
						}
					}
					action = new Offer(getAgentID(), mynextBid);
				}
			}
			// Thread.sleep(1000); //
		} catch (Exception e) {
			e.printStackTrace();
			targetuti = getTargetUtility();
			try {
				Bid mynextBid = getNextBid(targetuti);
				action = new Offer(getAgentID(), mynextBid);
			} catch (Exception es) {
				e.printStackTrace();
				// best guess if things go wrong.
				action = new Accept(getAgentID(),
						((ActionWithBid) actionOfOpponent).getBid());
			}
			// action=new Accept(getAgentID()); // best guess if things go
			// wrong.
		}

		if (action instanceof Offer) {
			this.myLastBid = ((Offer) action).getBid();
			myHistory.add(this.myLastBid);
		}
		return action;
	}

	// generate the objective utitility of my next offer
	public double getTargetUtility() {
		double tround = Math.max(averResT(), averLastTResT(3));

		double lefttime = 1 - timeline.getTime();

		roundleft = lefttime / tround;

		if (roundleft > 6.7)
			Utility.MINIMUM_BID_UTILITY = 0.93 * HPOSSIBLEU;
		else if (roundleft > 5)
			Utility.MINIMUM_BID_UTILITY = 0.90 * HPOSSIBLEU;
		else if (lefttime > 3 * tround)
			Utility.MINIMUM_BID_UTILITY = 0.86 * HPOSSIBLEU;
		else if (lefttime > 2.3 * tround)
			Utility.MINIMUM_BID_UTILITY = 0.8 * HPOSSIBLEU;
		else
			Utility.MINIMUM_BID_UTILITY = 0.6 * HPOSSIBLEU;
		if (lefttime < 15 * tround)
			ACCEPTABLEU = 0.92 * HPOSSIBLEU;
		else
			ACCEPTABLEU = 0.96 * HPOSSIBLEU;

		// consider the domain competition
		double averopu = 0, averopui = 0;
		if (this.opponentHistory.size() > 0)
			averopui = this.topponentBidU / this.opponentHistory.size();
		averopu = Math.max(0.30, averopui);

		double rte = 20 + (1 - averopu) / 0.10 * 20;
		if ((lefttime < rte * tround) & (this.opponentHistory.size() > 3)
				& (averopu < 0.75)) {
			Utility.MINIMUM_BID_UTILITY = Utility.MINIMUM_BID_UTILITY
					- (0.75 - averopu) / 2.5;
		}
		Utility.MINIMUM_BID_UTILITY = Math.max(0.50,
				Utility.MINIMUM_BID_UTILITY); // no
												// less
												// than
												// 0.5
		double time = timeline.getTime();
		// if(time>0.5)
		Utility.MINIMUM_BID_UTILITY = Utility.MINIMUM_BID_UTILITY
				* (Math.min(0.75, averopu) / 3 + 0.75);
		Utility.MINIMUM_BID_UTILITY = Math.max(Utility.MINIMUM_BID_UTILITY,
				averopu);
		double targetuti = HPOSSIBLEU
				- (HPOSSIBLEU - Utility.MINIMUM_BID_UTILITY)
						* Math.pow(time, eagerness);
		// Debug.LOG("CASE 4-"+this.opponentUtis.size()+" "+rte+"
		// roundleft-"+roundleft);

		if (lefttime < 1.6 * tround) {
			suggestBid = this.opponentHistory.get(this.bestTenBids.getFirst());
			return targetuti;
		}

		// consider op's best past offer
		if (lefttime > 50 * tround)
			targetuti = Math.max(targetuti,
					opponentUtis.get(this.bestTenBids.getFirst()) * 1.001);
		if (((lefttime < 10 * tround) & (opponentUtis
				.get(this.bestTenBids.getFirst()) > targetuti * 0.95))
				| opponentUtis.get(this.bestTenBids.getFirst()) >= targetuti) {
			double newtargetuti = targetuti;
			if ((lefttime < 10 * tround) & (opponentUtis
					.get(this.bestTenBids.getFirst()) > targetuti * 0.95))
				newtargetuti = targetuti * 0.95;

			// check whether the best op bid was offered in the last 4 rouds
			boolean offered = false;
			int length = Math.min(this.myHistory.size(), 4);

			for (int i = this.myHistory.size() - 1; i >= this.myHistory.size()
					- length; i--) {
				if (this.myHistory.get(i).equals(this.opponentHistory
						.get(this.bestTenBids.getFirst()))) {
					offered = true;
				}
			}
			if (offered) {
				LinkedList<Integer> candidates = new LinkedList<Integer>();
				for (int i = 0; i < bestTenBids.size(); i++)
					if (this.opponentUtis
							.get(bestTenBids.get(i)) >= newtargetuti)
						candidates.add(bestTenBids.get(i));
				int indexc = (int) (random100.nextDouble() * candidates.size());
				suggestBid = this.opponentHistory.get(candidates.get(indexc));

			} else {
				suggestBid = this.opponentHistory
						.get(this.bestTenBids.getFirst());
			}
			targetuti = newtargetuti;
		}
		return targetuti;
	}

	// generate the next bid satisfing the
	private Bid getNextBid(double targetuti) throws Exception {
		Bid nextBid = null;
		double maxdiff = Double.MAX_VALUE;
		double tempmaxdiff = Double.MAX_VALUE;

		LinkedList<BidDetails> candidates = new LinkedList<BidDetails>();
		BidIterator bidsIter = new BidIterator(utilitySpace.getDomain());
		while (bidsIter.hasNext()) {
			Bid tmpBid = bidsIter.next();
			double utitemp = utilitySpace.getUtility(tmpBid);
			double vlowbound;
			if (roundleft > 30)
				vlowbound = Math.max(this.opponentUtis.get(bestTenBids.get(0)),
						targetuti);
			else
				vlowbound = 0.96 * targetuti;
			if ((utitemp > vlowbound) & (utitemp < 1.08 * targetuti))
				candidates.add(new BidDetails(tmpBid,
						utilitySpace.getUtility(tmpBid), timeline.getTime()));
			double currentdiff = Math.abs(utitemp - targetuti);
			if (currentdiff < tempmaxdiff) {
				tempmaxdiff = currentdiff;
			}

			if ((currentdiff < maxdiff) & (utitemp > targetuti)) {
				maxdiff = currentdiff;
				nextBid = tmpBid;
			}
		}
		if (this.myHistory.size() > 10) {
			candidates.add(new BidDetails(nextBid,
					utilitySpace.getUtility(nextBid), timeline.getTime()));
			if (TEST_EQUIVALENCE) {
				Collections.sort(candidates,
						new BidDetailsStrictSorterUtility());
			}
			int indexc = (int) (random200.nextDouble() * candidates.size());
			nextBid = candidates.get(indexc).getBid();
		}

		return nextBid;
	}

	// learning --to complete
	public void updateBelief(Bid opponentBid, double uti) {
		int index = this.opponentHistory.size() - 1;
		if (bestTenBids.size() == 0)
			bestTenBids.add(index);
		else {
			LinkedList<Integer> newlist = new LinkedList<Integer>();
			if (uti > this.opponentUtis.get(bestTenBids.getFirst())) {
				newlist.add(index);
				for (int j = 0; j < bestTenBids.size(); j++)
					newlist.add(bestTenBids.get(j));
				bestTenBids = newlist;
			} else if (uti <= this.opponentUtis.get(bestTenBids.getLast()))
				bestTenBids.add(index);
			else {
				for (int i = 1; i < bestTenBids.size(); i++) {
					if ((uti <= this.opponentUtis.get(bestTenBids.get(i - 1)))
							& (uti > this.opponentUtis
									.get(bestTenBids.get(i)))) {
						for (int j = 0; j < i; j++)
							newlist.add(bestTenBids.get(j));
						newlist.add(index);
						for (int j = i; j < bestTenBids.size(); j++)
							newlist.add(bestTenBids.get(j));
						break;
					}
				}
				// Debug.LOG("CASE 3-"+this.opponentUtis.size());
				bestTenBids = newlist;
			}
		}
		if (bestTenBids.size() > 10)
			bestTenBids.removeLast();
	}

	public double averResT() {
		if (resptimes.size() == 0)
			return 0;
		double total = 0;
		for (int i = 0; i < resptimes.size(); i++)
			total = total + resptimes.get(i);
		return total / resptimes.size();
	}

	public double averLastTResT(int length) {
		if (resptimes.size() < length)
			return 0;
		double total = 0;
		for (int i = resptimes.size() - 1; i > resptimes.size() - length
				- 1; i--)
			total = total + resptimes.get(i);
		return total / length;
	}

	@Override
	public SupportedNegotiationSetting getSupportedNegotiationSetting() {
		return SupportedNegotiationSetting.getLinearUtilitySpaceInstance();
	}

	@Override
	public String getDescription() {
		return "ANAC2010";
	}
}