package agents.anac.y2014.SimpaticoAgent;

import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;

/**
 * @author Bedour Alrayes, Paulo Ricca, Ozgur Kafali, Kostas Stathis
 * @institution Royal Holloway University of London
 *
 *
 */
public class Simpatico extends Agent {
	private Action actionOfPartner = null;
	private double initialMinimumBidUtility = 0.9;
	private double minimumBidUtility = initialMinimumBidUtility;
	private double acceptanceBidUtility = 0.85;
	private BidTuple bestOpponentOffer = null;
	private ArrayList<BidTuple> bestOpponentOffers = new ArrayList<Simpatico.BidTuple>();
	private int bidToSubmitBack = 0; // used in the last moments, we propose
										// back the best of the opponents bids

	// Search Parameters
	private int initialSearchDepth = 3;
	private int maximumSearchDepth = initialSearchDepth;
	private int maximumSearchDepthForOurBids = 2;
	private float percentageOfIssuesToChange = 0.5f;
	private float randomSearchRatio = 0.3f;

	// Decision making parameters
	double maximumRoundTime = 0;
	double lastRoundTime = 0;
	boolean opponentIsCooperative = true;

	int countOpponentOffers = 0;
	double cooperationthreshold = 0.5;
	double minimumCooperativeUtility = 0.5;
	double numberOfCooperativeUtilities;

	@Override
	public void init() {
		actionOfPartner = null;
		initialMinimumBidUtility = 0.9;
		minimumBidUtility = initialMinimumBidUtility;
		acceptanceBidUtility = 0.85;
		bestOpponentOffer = null;
		bestOpponentOffers = new ArrayList<Simpatico.BidTuple>();
		bidToSubmitBack = 0; // used in the last moments, we propose back the
								// best of the opponents bids

		// Search Parameters
		initialSearchDepth = 3;
		maximumSearchDepth = initialSearchDepth;
		maximumSearchDepthForOurBids = 2;
		percentageOfIssuesToChange = 0.5f;
		randomSearchRatio = 0.3f;

		// Decision making parameters
		maximumRoundTime = 0;
		lastRoundTime = 0;
		opponentIsCooperative = true;

		countOpponentOffers = 0;
		cooperationthreshold = 0.5;
		minimumCooperativeUtility = 0.5;
	}

	private BidTuple SearchNeighbourhood(BidTuple initialBid) {
		return SearchNeighbourhood(initialBid, maximumSearchDepth);
	}

	@Override
	public String getVersion() {
		return "1.0";
	}

	@Override
	public String getName() {
		return "Simpatico";
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
	}

	private BidTuple SearchNeighbourhood(BidTuple initialBid, int depth) {
		if (depth == 0) {
			return initialBid;
		}
		BidTuple bestBid = initialBid;
		int nIssues = initialBid.bid.getIssues().size();

		int numberOfIssuesToChange = (int) (nIssues
				* percentageOfIssuesToChange);
		if (numberOfIssuesToChange > 5)
			numberOfIssuesToChange = 5;
		debug("NUMBER OF ISSUES: " + numberOfIssuesToChange, 3);

		int[] issuesToChange = new int[numberOfIssuesToChange];
		Random rand = new Random();
		for (int i = 0; i < issuesToChange.length; i++) {
			boolean alreadyChosen;
			boolean assigned = false;
			do {
				alreadyChosen = false;
				int issueIndexChosen = 1 + rand.nextInt(nIssues - 1);
				for (int j = 0; j <= i - 1; j++) {
					if (issuesToChange[j] == issueIndexChosen) {
						alreadyChosen = true;
					}
				}
				if (!alreadyChosen) {
					issuesToChange[i] = issueIndexChosen;
					assigned = true;
				}

			} while (!assigned && alreadyChosen);
		}
		try {
			for (int i = 0; i < issuesToChange.length; i++) {
				BidTuple bidTupleTmp;
				switch (initialBid.bid.getIssues().get(issuesToChange[i])
						.getType()) {
				case DISCRETE:
					bidTupleTmp = getDiscreteIssueVariation(initialBid,
							issuesToChange[i], 1);
					if (bidTupleTmp != null) {
						if (bidTupleTmp.utility > bestBid.utility)
							bestBid = bidTupleTmp;
						BidTuple returnedBid = SearchNeighbourhood(bidTupleTmp,
								depth - 1);
						if (returnedBid.utility > bestBid.utility)
							bestBid = returnedBid;
					}

					bidTupleTmp = getDiscreteIssueVariation(initialBid,
							issuesToChange[i], -1);
					if (bidTupleTmp != null) {
						if (bidTupleTmp.utility > bestBid.utility)
							bestBid = bidTupleTmp;
						BidTuple returnedBid = SearchNeighbourhood(bidTupleTmp,
								depth - 1);
						if (returnedBid.utility > bestBid.utility)
							bestBid = returnedBid;
					}
					break;
				case REAL:
					IssueReal issueReal = (IssueReal) initialBid.bid.getIssues()
							.get(issuesToChange[i]);
					double range = issueReal.getUpperBound()
							- issueReal.getLowerBound();
					double variation = range * 0.05;

					bidTupleTmp = getRealIssueVariation(initialBid,
							issuesToChange[i], rand.nextFloat() * variation);
					if (bidTupleTmp != null) {
						if (bidTupleTmp.utility > bestBid.utility)
							bestBid = bidTupleTmp;
						BidTuple returnedBid = SearchNeighbourhood(bidTupleTmp,
								depth - 1);
						if (returnedBid.utility > bestBid.utility)
							bestBid = returnedBid;
					}

					bidTupleTmp = getRealIssueVariation(initialBid,
							issuesToChange[i], -rand.nextFloat() * variation);
					if (bidTupleTmp != null) {
						if (bidTupleTmp.utility > bestBid.utility)
							bestBid = bidTupleTmp;
						BidTuple returnedBid = SearchNeighbourhood(bidTupleTmp,
								depth - 1);
						if (returnedBid.utility > bestBid.utility)
							bestBid = returnedBid;
					}
					break;
				case INTEGER:
					bidTupleTmp = getIntegerIssueVariation(initialBid,
							issuesToChange[i], 1);
					if (bidTupleTmp != null) {
						if (bidTupleTmp.utility > bestBid.utility)
							bestBid = bidTupleTmp;
						BidTuple returnedBid = SearchNeighbourhood(bidTupleTmp,
								depth - 1);
						if (returnedBid.utility > bestBid.utility)
							bestBid = returnedBid;
					}

					bidTupleTmp = getIntegerIssueVariation(initialBid,
							issuesToChange[i], -1);
					if (bidTupleTmp != null) {
						if (bidTupleTmp.utility > bestBid.utility)
							bestBid = bidTupleTmp;
						BidTuple returnedBid = SearchNeighbourhood(bidTupleTmp,
								depth - 1);
						if (returnedBid.utility > bestBid.utility)
							bestBid = returnedBid;
					}
					break;
				default:
					throw new Exception(
							"issue type not supported by " + getName());
				}
			}
		} catch (Exception e) {
			debug(e.getMessage());
			e.printStackTrace();
		}
		return bestBid;
	}

	private void saveBestOpponentBids(BidTuple bid) {
		int insertPosition = 0;
		for (int i = 0; i < bestOpponentOffers.size(); i++) {
			if (bestOpponentOffers.get(i).utility > bid.utility)
				insertPosition++;
		}
		bestOpponentOffers.add(insertPosition, bid);
		if (bestOpponentOffers.size() > 10)
			bestOpponentOffers.remove(10);
	}

	private BidTuple getDiscreteIssueVariation(BidTuple bid, int issueId,
			int variation) throws Exception {
		Bid tmpBid = new Bid(bid.bid);
		BidTuple bidTupleTmp = null;
		IssueDiscrete issueDiscrete = (IssueDiscrete) tmpBid.getIssues()
				.get(issueId);
		int temIssueIndex = issueDiscrete.getIndex(issueDiscrete);
		if (issueDiscrete.getNumberOfValues() >= temIssueIndex + variation
				&& temIssueIndex + variation >= 0) {
			ValueDiscrete value = issueDiscrete
					.getValue(temIssueIndex + variation);
			tmpBid = tmpBid.putValue(issueId, value);
			double util = getUtility(tmpBid);
			bidTupleTmp = new BidTuple(tmpBid, util);
		}
		return bidTupleTmp;
	}

	private BidTuple getRealIssueVariation(BidTuple bid, int issueId,
			double variation) throws Exception {
		Bid tmpBid = new Bid(bid.bid);
		BidTuple bidTupleTmp = null;
		IssueReal issueReal = (IssueReal) tmpBid.getIssues().get(issueId);
		int temIssueValue = ((ValueInteger) tmpBid.getValue(issueId))
				.getValue();
		if (issueReal.getUpperBound() >= temIssueValue + variation
				&& issueReal.getLowerBound() <= temIssueValue + variation) {
			ValueReal value = new ValueReal(temIssueValue + variation);
			tmpBid = tmpBid.putValue(issueId, value);
			double util = getUtility(tmpBid);
			bidTupleTmp = new BidTuple(tmpBid, util);
		}
		return bidTupleTmp;
	}

	private BidTuple getIntegerIssueVariation(BidTuple bid, int issueId,
			int variation) throws Exception {
		Bid tmpBid = new Bid(bid.bid);
		BidTuple bidTupleTmp = null;
		IssueInteger issueInteger = (IssueInteger) tmpBid.getIssues()
				.get(issueId);
		int temIssueValue = ((ValueInteger) tmpBid.getValue(issueId))
				.getValue();
		if (issueInteger.getUpperBound() >= temIssueValue + variation
				&& issueInteger.getLowerBound() <= temIssueValue + variation) {
			ValueInteger value = new ValueInteger(temIssueValue + variation);
			tmpBid = tmpBid.putValue(issueId, value);
			double util = getUtility(tmpBid);
			bidTupleTmp = new BidTuple(tmpBid, util);
		}
		return bidTupleTmp;
	}

	@Override
	public Action chooseAction() {
		boolean NeverFoundHighUtilityInNeighbourhood = true;
		debug("1. choosing action");
		// get current time
		double time = timeline.getTime();
		// calculate maximum round time
		if (actionOfPartner != null) {
			if (maximumRoundTime < time - lastRoundTime)
				maximumRoundTime = time - lastRoundTime;
			debug("maximumRoundTime: " + maximumRoundTime);
			lastRoundTime = time;
		}
		Action action = null;
		try {
			if (actionOfPartner == null) {
				lastRoundTime = time;
				debug("OFFER: FIRST RANDOM OFFER", 3);
				return action = chooseRandomBidAction();
			}
			if (actionOfPartner instanceof Offer) {
				Bid partnerBid = ((Offer) actionOfPartner).getBid();
				double offeredUtilFromOpponent = getUtility(partnerBid);

				if (time >= 0.99 && offeredUtilFromOpponent >= 0.75) {
					debug("ime >= 0.99 &&  offeredUtilFromOpponent>= 0.75");
					debug("OFFER: STRAIGHT ACCEPT, TIME IS RUNNING OUT, GOOD OFFER",
							3);
					return action = new Accept(getAgentID(), partnerBid);
				}

				BidTuple opponentBid = new BidTuple(partnerBid,
						offeredUtilFromOpponent);
				saveBestOpponentBids(opponentBid);

				if (bestOpponentOffer == null
						|| bestOpponentOffer.utility < offeredUtilFromOpponent)
					bestOpponentOffer = opponentBid;

				// Last moments, submit back the opponents best bids
				if ((time >= 1 - (maximumRoundTime * 1.09))) {
					debug("final round:" + maximumRoundTime);
					debug("(time>=1-(maximumRoundTime*1.05)) && bestOpponentOffer.utility >= 0.5");
					if (bidToSubmitBack >= bestOpponentOffers.size())
						bidToSubmitBack = 0;
					debug("OFFER: BEST OF THE OPPONENTS NR " + bidToSubmitBack,
							3);
					return action = new Offer(getAgentID(),
							bestOpponentOffers.get(bidToSubmitBack++).bid);
				}

				updateOpponentProfile(partnerBid, offeredUtilFromOpponent);
				updateMinimumBidUtility(partnerBid, offeredUtilFromOpponent);

				debug("3. searching the vicinity of the opp bid");
				debug("MINIMUM UTILITY: " + minimumBidUtility, 3);
				BidTuple bestBidInTheVicinity = SearchNeighbourhood(
						opponentBid);
				debug("bestBidInTheVicinity.utility"
						+ bestBidInTheVicinity.utility);
				if (bestBidInTheVicinity.utility < minimumBidUtility) {
					debug("4. bad search, going to pick a random");
					action = chooseRandomBidAction();

					// if the random returned a null, which means we couldnt
					// find a bid before time ran out,
					// we offer back the best opponent bid
					if (((Offer) action).getBid() == null
							&& bestOpponentOffer.utility >= 0.5) {
						debug("action == null && bestOpponentOffer.utility >= 0.5");
						return action = new Offer(getAgentID(),
								bestOpponentOffer.bid);
					}

					debug("bid Random");
					debug("OFFER: RANDOM BID", 3);

				} else {
					debug("5. found a good bid in the vicinity");
					NeverFoundHighUtilityInNeighbourhood = false;
					action = new Offer(getAgentID(), bestBidInTheVicinity.bid);
					debug("Bid in bestBidInTheVicinity");
					debug("OFFER: OPPONENT VICINITY", 3);
				}

				if (NeverFoundHighUtilityInNeighbourhood) {
					if (time > 0.8) {
						debug("6. time > 0.8, maximumsearchdepth = "
								+ (int) (initialSearchDepth * 1.5));
						maximumSearchDepth = (int) (initialSearchDepth * 1.5);
					} else if (time > 0.5) {
						debug("7. time > 0.5, maximumsearchdepth = "
								+ (int) (initialSearchDepth * 1.2));
						maximumSearchDepth = (int) (initialSearchDepth * 1.2);
					}
				}

				Bid myBid = ((Offer) action).getBid();
				double myOfferedUtil = getUtility(myBid);

				// accept under certain circumstances
				if (isAcceptable(offeredUtilFromOpponent, myOfferedUtil,
						time)) {
					debug("8. opponents bid is acceptable");
					debug("Accept");
					debug("OFFER: STRAIGHT ACCEPT", 3);
					action = new Accept(getAgentID(), partnerBid);
				}

			}

		} catch (Exception e) {
			debug("Exception in ChooseAction:" + e.getMessage());
			return action = new Offer(getAgentID(), bestOpponentOffer.bid); // best
																			// guess
																			// if
																			// things
																			// go
																			// wrong.
		}
		debug("9. end of choose action");
		return action;
	}

	private void updateOpponentProfile(Bid partnerBid,
			double offeredUtilFromOpponent) {
		countOpponentOffers++;
		if (offeredUtilFromOpponent > minimumCooperativeUtility) {
			numberOfCooperativeUtilities++;
		}
		if ((numberOfCooperativeUtilities
				/ countOpponentOffers) > cooperationthreshold) {
			opponentIsCooperative = true;
			debug("-- OPPONENT IS COOPERATIVE", 1);
		} else {
			opponentIsCooperative = false;
			debug("-- OPPONENT IS HARD HEADED", 1);
		}
	}

	private boolean isAcceptable(double offeredUtilFromOpponent,
			double myOfferedUtil, double time) throws Exception {
		double discount_utility_factor = utilitySpace.getDiscountFactor();
		acceptanceBidUtility = initialMinimumBidUtility
				* Math.pow(discount_utility_factor, time);

		if (time >= 0.9) {
			initialMinimumBidUtility = 0.85;
		}

		if (offeredUtilFromOpponent >= acceptanceBidUtility) {
			debug("offeredUtilFromOpponent >= " + acceptanceBidUtility);
			return true;
		}

		if (offeredUtilFromOpponent >= myOfferedUtil) {
			debug("offeredUtilFromOpponent >= myOfferedUtil");
			return true;
		}

		return false;
	}

	private Action chooseRandomBidAction() {
		Bid nextBid = null;
		try {
			nextBid = getRandomBid();
		} catch (Exception e) {
			debug("Problem with received bid:" + e.getMessage()
					+ ". cancelling bidding");
		}
		return (new Offer(getAgentID(), nextBid));
	}

	private Bid getRandomBid() throws Exception {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>(); // pairs
																		// <issuenumber,chosen
																		// value
																		// string>
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random randomnr = new Random();

		debug("R 1.beginning Random pick");

		Bid bid = null;

		int tries = 0;
		int absoluteTries = 0;
		float minimumUtilTolerance = 0;
		do {
			tries++;
			absoluteTries++;
			if (tries > 2000)// && minimumUtilTolerance<0.09)
			{
				debug("R 1.increasing tolerance");
				minimumUtilTolerance += 0.001;
				tries = 0;
			}

			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					int optionIndex = randomnr
							.nextInt(lIssueDiscrete.getNumberOfValues());
					values.put(lIssue.getNumber(),
							lIssueDiscrete.getValue(optionIndex));
					break;
				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					int optionInd = randomnr.nextInt(
							lIssueReal.getNumberOfDiscretizationSteps() - 1);
					values.put(lIssueReal.getNumber(), new ValueReal(lIssueReal
							.getLowerBound()
							+ (lIssueReal.getUpperBound()
									- lIssueReal.getLowerBound()) * (optionInd)
									/ (lIssueReal
											.getNumberOfDiscretizationSteps())));
					break;
				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					int optionIndex2 = lIssueInteger.getLowerBound()
							+ randomnr.nextInt(lIssueInteger.getUpperBound()
									- lIssueInteger.getLowerBound());
					values.put(lIssueInteger.getNumber(),
							new ValueInteger(optionIndex2));
					break;
				default:
					throw new Exception("issue type " + lIssue.getType()
							+ " not supported by SimpleAgent2");
				}
			}
			bid = new Bid(utilitySpace.getDomain(), values);

		} while (getUtility(bid) < minimumBidUtility - minimumUtilTolerance);
		debug("R 2.Generated a random bid. current utility: " + getUtility(bid)
				+ ", tolerance: " + minimumUtilTolerance + "; tries: "
				+ absoluteTries);

		Bid randomBid = bid;

		// // search the vicinity of the random offer
		if (randomnr.nextFloat() < randomSearchRatio) {
			debug("R 3.Going to search the vicinity of the random pick");
			bid = SearchNeighbourhood(new BidTuple(bid, getUtility(bid)),
					maximumSearchDepthForOurBids).bid;
		}

		if (bid != randomBid) {
			debug("OFFER: RANDOM VICINITY BID", 3);
		} else {
			debug("OFFER: RANDOM BID", 3);
		}

		debug("R 4.Random pick finished");
		return bid;
	}

	private void updateMinimumBidUtility(Bid partnerBid,
			double offeredUtilFromOpponent) {

		double time = timeline.getTime();
		double discount_utility_factor = utilitySpace.getDiscountFactor();
		if (time >= 0.5 && opponentIsCooperative) {
			minimumBidUtility = 0.88 * Math.pow(discount_utility_factor, time);
			debug("2. ADJUSTING MINIMUM UTILITY: " + minimumBidUtility);
		}
		debug("2.  MINIMUM UTILITY: " + minimumBidUtility);
		debug("2. discount factor: " + discount_utility_factor);
	}

	private void debug(String line) {
		debug(line, 0);
	}

	private void debug(String line, int priority) {
		if (priority >= 4)
			System.out.println(line);
	}

	private class BidTuple {
		public Bid bid;
		public double utility;

		public BidTuple(Bid bid, double utility) {
			this.bid = bid;
			this.utility = utility;
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2014 compatible with non-linear utility spaces";
	}

}
