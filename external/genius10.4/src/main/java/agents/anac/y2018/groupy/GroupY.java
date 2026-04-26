package agents.anac.y2018.groupy;
import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.list.Tuple;
import genius.core.misc.Range;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.PersistentDataType;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;

/**
 * Sample party that accepts the Nth offer, where N is the number of sessions
 * this [agent-profile] already did.
 */
public class GroupY extends AbstractNegotiationParty {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private double timeTreshhold;
	private Bid lastReceivedBid = null;
	private int nrChosenActions = 0; // number of times chosenAction was called.
	private StandardInfoList history;
	private Bid Max = null;
	private Double reservationV;
	private Bid Min = null;
	private Range r = new Range(0, 0);
	private OpponentModelHolder hold;
	private ArrayList<AgentID> agentId;
	SortedOutcomeSpace sos;
	private ArrayList<Bid> loopB = new ArrayList<Bid>();
	int loopCounter = 0;
	private boolean first = true;

	@Override
	public void init(NegotiationInfo info) {

		super.init(info);
		hold = new OpponentModelHolder();
		agentId = new ArrayList<AgentID>();
		sos = new SortedOutcomeSpace(utilitySpace);

		try {
			Max = utilitySpace.getMaxUtilityBid();
			Min = utilitySpace.getMinUtilityBid();
			r.setUpperbound(getUtility(Max));
			reservationV = utilitySpace.getReservationValue();

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("Discount Factor is " + info.getUtilitySpace().getDiscountFactor());
		System.out.println("Reservation Value is " + info.getUtilitySpace().getReservationValueUndiscounted());

		if (getData().getPersistentDataType() != PersistentDataType.STANDARD) {
			throw new IllegalStateException("need standard persistent data");
		}
		history = (StandardInfoList) getData().get();
			
		if (!history.isEmpty()) {

			Map<String, Double> maxutils = new HashMap<String, Double>();
			StandardInfo lastinfo = history.get(history.size() - 1);

			if (hold.getAgentModels().get(getPartyId()) == null) {
				hold.getAgentModels().put(getPartyId(), new OpponentModel(utilitySpace, timeline.getTime()));
				agentId.add(getPartyId());

			}
			int remember =0;
			double normalize=102;
			for (Tuple<String, Double> offered : lastinfo.getUtilities()) {
				
				String party = offered.get1();
				Double util = offered.get2();
				Bid bid = sos.getBidNearUtility(util).getBid();
				hold.getAgentModels().get(getPartyId()).updateModel(bid,(int)normalize);
				if(remember==100000)break;
				// maxutils.put(party, maxutils.containsKey(party) ?
				// Math.max(maxutils.get(party), util) : util);
				remember++;
				normalize=normalize-0.001;
			}
			// System.out.println(maxutils); // notice tournament suppresses all
			// output.
		}
	}

	public Action chooseAction(List<Class<? extends Action>> validActions) {
		nrChosenActions++;

		if (nrChosenActions < 4) {
			loopCounter = loopCounter + 1;
			if (lastReceivedBid != null && getUtility(lastReceivedBid) == getUtility(Max)) {

				return new Accept(getPartyId(), lastReceivedBid);

			}
			loopB.add(Max);
			return new Offer(getPartyId(), Max);

		} else {
			// if generatebid<lastReceivedBid
			// reservation value must change to undiscounted
			if (timeline.getTime() < 0.99) {
				if (lastReceivedBid != null && getUtility(lastReceivedBid) >= getUtility(getSuperOffer())
						&& reservationV < getUtility(lastReceivedBid)) {
					return new Accept(getPartyId(), lastReceivedBid);

				} else {
					loopCounter = loopCounter + 1;
					if (first) {
						loopB.add(getSuperOffer());
						first = false;
					}

					if (loopCounter == 5) {
						loopCounter = 1;
						if (!loopB.isEmpty())
							loopB.remove(0);
						loopB.add(getSuperOffer());
					}

					// return new Offer(getPartyId(), Max);
					return new Offer(getPartyId(), loopB.get(loopCounter - 1));

				}
			} else {
				if (0.65 <= getUtility(lastReceivedBid))
					return new Accept(getPartyId(), lastReceivedBid);
				return new Offer(getPartyId(), sos.getBidNearUtility(0.7).getBid());

			}

		}

	}

	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		if (action instanceof Offer) {
			lastReceivedBid = ((Offer) action).getBid();

			if (hold.getAgentModels().get(sender) == null) {
				hold.getAgentModels().put(sender, new OpponentModel(utilitySpace, timeline.getTime()));
				agentId.add(sender);

			}

			hold.getAgentModels().get(sender).updateModel(lastReceivedBid,1);
		}

	}

    @Override
    public String getDescription() {
        return "ANAC2018";
    }

	/*
	 * public double timeDeal() {
	 * 
	 * Double remTimeRatio = 1 - timeline.getTime();
	 * 
	 * double minUtility = 0 + getUtility(Min); if (remTimeRatio > 0.5) { minUtility
	 * = 0.8 + getUtility(Min); } else if (remTimeRatio > 0.1) { minUtility = 0.6 +
	 * getUtility(Min); } else { minUtility = 0.3 + getUtility(Min); }
	 * 
	 * return minUtility + (getUtility(Max) - minUtility) * Math.pow(remTimeRatio, 1
	 * / Math.E); }
	 */
	public double timeDeal() {

		Double remTimeRatio = 1 - timeline.getTime();

		double minUtility = 0 + getUtility(Min);
		if (remTimeRatio > 0.5) {
			minUtility = 0.85 + getUtility(Min);
		} else if (remTimeRatio > 0.2) {
			minUtility = 0.65 + getUtility(Min);
		} else {
			minUtility = 0.45 + getUtility(Min);
		}

		return minUtility + (getUtility(Max) - minUtility) * Math.pow(remTimeRatio, 1 / Math.E);
	}

	/*
	 * public Bid getSuperOffer() {
	 * 
	 * double utilBid; timeTreshhold = timeDeal(); r.setLowerbound(timeTreshhold);
	 * List<BidDetails> bids = sos.getBidsinRange(r); Set set =
	 * hold.getAgentModels().entrySet(); Iterator iterator = set.iterator(); Bid
	 * bestBid = bids.get(0).getBid(); double bestBidUtil = 0; for (BidDetails
	 * bidTot : bids) { utilBid = 0; while (iterator.hasNext()) { Map.Entry entry =
	 * (Map.Entry) iterator.next(); OpponentModel op = (OpponentModel)
	 * entry.getValue(); utilBid = utilBid + op.ExpectedUtility(bidTot.getBid());
	 * 
	 * } if (bestBidUtil < utilBid) bestBid = bidTot.getBid();
	 * 
	 * }
	 * 
	 * return bestBid; }
	 */
	public Bid getSuperOffer() {

		double utilBid;
		timeTreshhold = timeDeal();
		r.setLowerbound(timeTreshhold);

		List<BidDetails> bids = sos.getBidsinRange(r);

		Bid bestBid = bids.get(0).getBid();
		if (bestBid == null)
			bestBid = Max;
		double bestBidUtil = 0;
		for (BidDetails bidTot : bids) {
			utilBid = 0;
			for (AgentID id : agentId) {

				OpponentModel op = hold.getAgentModels().get(id);

				utilBid = utilBid + op.ExpectedUtility(bidTot.getBid());

			}
			if (bestBidUtil < utilBid) {
				bestBid = bidTot.getBid();
				bestBidUtil = utilBid;
			}
		}

		// System.out.println("bestbid: " + bestBid);

		return bestBid;
	}

}
