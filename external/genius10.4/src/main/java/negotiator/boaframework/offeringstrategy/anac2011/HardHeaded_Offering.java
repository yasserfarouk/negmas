package negotiator.boaframework.offeringstrategy.anac2011;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Map;
import java.util.Map.Entry;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.utility.AdditiveUtilitySpace;

import java.util.Random;
import java.util.TreeMap;

import negotiator.boaframework.offeringstrategy.anac2011.hardheaded.BidSelector;
import negotiator.boaframework.opponentmodel.DefaultModel;
import negotiator.boaframework.opponentmodel.HardHeadedFrequencyModel;
import negotiator.boaframework.sharedagentstate.anac2011.HardHeadedSAS;

/**
 * This is the decoupled Offering Strategy for HardHeaded (ANAC2011). The code
 * was taken from the ANAC2011 HardHeaded and adapted to work within the BOA
 * framework.
 * 
 * >>Extension of opponent model>> As default the agent uses the frequency model
 * to select the best bid from a small set of bids. The agent is extended such
 * that other models can be used. If no opponent model is given, then a random
 * bid is chosen.
 * 
 * DEFAULT OM: HardHeadedFrequencyModel with default parameters
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class HardHeaded_Offering extends OfferingStrategy {

	private double Ka = 0.05;
	private double e = 0.05;
	private double maxUtil = 1;
	private double MINIMUM_BID_UTILITY = 0.585D;
	private double minUtil = MINIMUM_BID_UTILITY;
	private LinkedList<Entry<Double, Bid>> offerQueue;
	private final double UTILITY_TOLORANCE = 0.01D;
	private final int TOP_SELECTED_BIDS = 4;
	private double discountF = 1D;
	int round = 0;
	private BidSelector BSelector;
	private Bid opponentbestbid = null;
	private Entry<Double, Bid> opponentbestentry;
	private boolean firstRound = true;
	private final boolean TEST_EQUIVALENCE = false;
	private Random random100;
	private Random random200;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public HardHeaded_Offering() {
	}

	@Override
	public void init(NegotiationSession negotiationSession, OpponentModel model, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {
		if (model instanceof DefaultModel) {
			model = new HardHeadedFrequencyModel();
			model.init(negotiationSession, null);
			oms.setOpponentModel(model);
		}
		initializeAgent(negotiationSession, model, oms);
	}

	public void initializeAgent(NegotiationSession negoSession, OpponentModel model, OMStrategy oms) {
		this.negotiationSession = negoSession;
		BSelector = new BidSelector((AdditiveUtilitySpace) negotiationSession.getUtilitySpace());
		offerQueue = new LinkedList<Entry<Double, Bid>>();
		this.opponentModel = model;
		this.omStrategy = oms;
		helper = new HardHeadedSAS(negoSession);
		Entry<Double, Bid> highestBid = BSelector.getBidList().lastEntry();
		try {
			maxUtil = negotiationSession.getUtilitySpace().getUtility(highestBid.getValue());
		} catch (Exception e) {
			e.printStackTrace();
		}
		if (negoSession.getDiscountFactor() <= 1D && negoSession.getDiscountFactor() > 0D)
			discountF = negoSession.getDiscountFactor();

		if (TEST_EQUIVALENCE) {
			random100 = new Random(100);
			random200 = new Random(200);
		} else {
			random100 = new Random();
			random200 = new Random();
		}
	}

	@Override
	public BidDetails determineNextBid() {

		round++;

		double p = get_p();

		Entry<Double, Bid> potentialNextBid = null;

		double opbestvalue;
		if (!negotiationSession.getOpponentBidHistory().getHistory().isEmpty()) {
			Bid opponentLastBid = negotiationSession.getOpponentBidHistory().getLastBidDetails().getBid();
			try {
				if (opponentbestbid == null)
					opponentbestbid = opponentLastBid;
				else if (negotiationSession.getUtilitySpace().getUtility(opponentLastBid) > negotiationSession
						.getUtilitySpace().getUtility(opponentbestbid)) {
					opponentbestbid = opponentLastBid;
				}

				opbestvalue = BSelector.getBidList()
						.floorEntry(negotiationSession.getUtilitySpace().getUtility(opponentbestbid)).getKey();

				while (!BSelector.getBidList().floorEntry(opbestvalue).getValue().equals(opponentbestbid)) {
					opbestvalue = BSelector.getBidList().lowerEntry(opbestvalue).getKey();
				}
				opponentbestentry = BSelector.getBidList().floorEntry(opbestvalue);
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		}

		if (firstRound) {
			firstRound = !firstRound;
			potentialNextBid = BSelector.getBidList().lastEntry();
			offerQueue.add(potentialNextBid);
		} else if (offerQueue == null || offerQueue.isEmpty()) {
			// calculations of concession step according to time
			TreeMap<Double, Bid> newBids = new TreeMap<Double, Bid>();

			potentialNextBid = BSelector.getBidList()
					.lowerEntry(negotiationSession.getOwnBidHistory().getLastBidDetails().getMyUndiscountedUtil());
			newBids.put(potentialNextBid.getKey(), potentialNextBid.getValue());

			if (potentialNextBid.getKey() < p) {
				int indexer = negotiationSession.getOwnBidHistory().size();
				indexer = (int) Math.floor(indexer * random100.nextDouble());
				newBids.remove(potentialNextBid.getKey());
				BidDetails selected = negotiationSession.getOwnBidHistory().getHistory().get(indexer);
				newBids.put(selected.getMyUndiscountedUtil(), selected.getBid());
			}

			double firstUtil = potentialNextBid.getKey();
			Entry<Double, Bid> addBid = BSelector.getBidList().lowerEntry(firstUtil);
			double addUtil = addBid.getKey();

			while ((firstUtil - addUtil) < UTILITY_TOLORANCE && addUtil >= p) {
				newBids.put(addUtil, addBid.getValue());
				addBid = BSelector.getBidList().lowerEntry(addUtil);
				addUtil = addBid.getKey();
			}

			// adding selected bids to offering queue
			if (newBids.size() <= TOP_SELECTED_BIDS) {

				offerQueue.addAll(newBids.entrySet());
			} else {

				int addedSofar = 0;
				Entry<Double, Bid> bestBid = null;
				while (addedSofar <= TOP_SELECTED_BIDS) {
					bestBid = newBids.lastEntry();

					if (opponentModel instanceof NoModel) {
						// Do nothing, using the last entry is random enough
					} else {
						ArrayList<BidDetails> bids = new ArrayList<BidDetails>();
						for (Entry<Double, Bid> entry : newBids.entrySet()) {
							bids.add(new BidDetails(entry.getValue(), entry.getKey(), -1));
						}
						BidDetails selectedBid = omStrategy.getBid(bids);
						try {
							bestBid = new MyEntry<Double, Bid>(
									negotiationSession.getUtilitySpace().getUtility(selectedBid.getBid()),
									selectedBid.getBid());
						} catch (Exception e) {
							e.printStackTrace();
						}
					}
					offerQueue.add(bestBid);

					newBids.remove(bestBid.getKey());
					addedSofar++;

				}

			}

			// if opponentbest entry is better for us then the offer que then
			// replace the top entry
			if (offerQueue.getFirst().getKey() < opponentbestentry.getKey()) {
				offerQueue.addFirst(opponentbestentry);
			}
		}

		// if no bids are selected there must be a problem
		if (offerQueue.isEmpty() || offerQueue == null) {

			Bid bestBid1 = negotiationSession.getUtilitySpace().getDomain().getRandomBid(random200);
			BidDetails opponentLastBid = negotiationSession.getOpponentBidHistory().getLastBidDetails();
			try {
				if (opponentLastBid != null
						&& negotiationSession.getUtilitySpace().getUtility(bestBid1) <= negotiationSession
								.getUtilitySpace().getUtility(opponentLastBid.getBid())) {
					return null;
				} else {
					BidDetails offer = new BidDetails(bestBid1,
							negotiationSession.getUtilitySpace().getUtility(bestBid1), negotiationSession.getTime());
					if (offer.getMyUndiscountedUtil() < ((HardHeadedSAS) helper).getLowestYetUtility()) {
						((HardHeadedSAS) helper).setLowestYetUtility(offer.getMyUndiscountedUtil());
					}
					nextBid = offer;
					return offer;
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else {

			Entry<Double, Bid> toOffer = offerQueue.remove();

			BidDetails offer = new BidDetails(toOffer.getValue(), toOffer.getKey(), negotiationSession.getTime());
			if (offer.getMyUndiscountedUtil() < ((HardHeadedSAS) helper).getLowestYetUtility()) {

				try {
					((HardHeadedSAS) helper)
							.setLowestYetUtility(negotiationSession.getUtilitySpace().getUtility(offer.getBid()));
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			try {

				nextBid = offer;
				return nextBid;

			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return null;
	}

	@Override
	public BidDetails determineOpeningBid() {
		return determineNextBid();
	}

	/**
	 * This function calculates the concession amount based on remaining time,
	 * initial parameters, and, the discount factor.
	 * 
	 * @return double: concession step
	 */
	public double get_p() {

		double time = negotiationSession.getTime();
		double Fa;
		double p = 1D;
		double step_point = discountF;
		double tempMax = maxUtil;
		double tempMin = minUtil;
		double tempE = e;
		double ignoreDiscountThreshold = 0.9D;

		if (step_point >= ignoreDiscountThreshold) {
			Fa = Ka + (1 - Ka) * Math.pow(time / step_point, 1D / e);
			p = minUtil + (1 - Fa) * (maxUtil - minUtil);
		} else if (time <= step_point) {
			tempE = e / step_point;
			Fa = Ka + (1 - Ka) * Math.pow(time / step_point, 1D / tempE);
			tempMin += Math.abs(tempMax - tempMin) * step_point;
			p = tempMin + (1 - Fa) * (tempMax - tempMin);
		} else {
			// Ka = (maxUtil - (tempMax -
			// tempMin*step_point))/(maxUtil-minUtil);
			tempE = 30D;
			Fa = (Ka + (1 - Ka) * Math.pow((time - step_point) / (1 - step_point), 1D / tempE));
			tempMax = tempMin + Math.abs(tempMax - tempMin) * step_point;
			p = tempMin + (1 - Fa) * (tempMax - tempMin);
		}
		return p;
	}

	private class MyEntry<K, V> implements Map.Entry<K, V> {
		private final K key;
		private V value;

		public MyEntry(K key, V value) {
			this.key = key;
			this.value = value;
		}

		@Override
		public K getKey() {
			return key;
		}

		@Override
		public V getValue() {
			return value;
		}

		@Override
		public V setValue(V value) {
			V old = this.value;
			this.value = value;
			return old;
		}
	}

	@Override
	public String getName() {
		return "2011 - HardHeaded";
	}
}
