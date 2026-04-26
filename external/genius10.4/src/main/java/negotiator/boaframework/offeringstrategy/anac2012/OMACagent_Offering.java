package negotiator.boaframework.offeringstrategy.anac2012;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import agents.anac.y2012.OMACagent.TimeBidHistory;
import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.misc.Range;
import genius.core.utility.AdditiveUtilitySpace;
import negotiator.boaframework.opponentmodel.DefaultModel;

/**
 * This is the decoupled Bidding Strategy of OMACAgent
 *
 * For the opponent model extension a range of bids is found near the target
 * utility. The opponent model strategy uses the OM to select a bid from this
 * range of bids.
 * 
 * DEFAULT OM: None
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class OMACagent_Offering extends OfferingStrategy {

	private double MINIMUM_UTILITY = 0.59; // 0.618
	private double resU = 0.0;
	private double EU = 0.95; // a threshold of expected utility for a session,
								// in use now
	private double est_t = 0;
	private double est_u = 0;
	private TimeBidHistory mBidHistory; // new
	private int intervals = 100;
	private double timeInt = 1.0 / (double) intervals;
	private double discount = 1.0;
	private Bid maxBid = null; // the maxBid which can be made by itself
	private double maxBidU = 0.0; // the uti of the bid above
	private double cTime = 0.0;
	private int tCount = 0;
	private double nextUti = 0.96;
	private double discountThreshold = 0.845D;
	private double exma;
	private double est_mu;
	private double est_mt;
	private double maxTime = 180.0;
	private AdditiveUtilitySpace utilitySpace;
	private SortedOutcomeSpace outcomespace;

	private Random randomnr;
	private final boolean TEST_EQUIVALENCE = true;

	public OMACagent_Offering() {
	}

	public OMACagent_Offering(NegotiationSession negoSession, OpponentModel model, OMStrategy oms) throws Exception {
		init(negoSession, model, oms, null);
	}

	/**
	 * Init required for the Decoupled Framework.
	 */
	@Override
	public void init(NegotiationSession negoSession, OpponentModel model, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {
		if (model instanceof DefaultModel) {
			model = new NoModel();
		}
		if (!(model instanceof NoModel)) {
			outcomespace = new SortedOutcomeSpace(negoSession.getUtilitySpace());
		}
		super.init(negoSession, model, oms, parameters);
		utilitySpace = (AdditiveUtilitySpace) negoSession.getUtilitySpace();
		if (utilitySpace.getReservationValue() != null) {
			resU = utilitySpace.getReservationValue();
			if (MINIMUM_UTILITY < resU)
				MINIMUM_UTILITY = resU * 1.06;
		}

		if (utilitySpace.getDiscountFactor() <= 1D && utilitySpace.getDiscountFactor() > 0D)
			discount = utilitySpace.getDiscountFactor();

		try {
			maxBid = utilitySpace.getMaxUtilityBid();
			maxBidU = utilitySpace.getUtility(maxBid);
			EU = EU * maxBidU;
		} catch (Exception e) {
			// System.out.println("Errors in ini process!");
		}

		mBidHistory = new TimeBidHistory(this.utilitySpace, discount);
		if (TEST_EQUIVALENCE) {
			randomnr = new Random(100);
		} else {
			randomnr = new Random();

		}
	}

	@Override
	public BidDetails determineOpeningBid() {
		return determineNextBid();
	}

	@Override
	public BidDetails determineNextBid() {
		Bid bidToOffer = null;

		try {
			if (negotiationSession.getOpponentBidHistory().size() == 0) {
				bidToOffer = chooseBidAction();
			} else {
				cTime = negotiationSession.getTime();
				Bid partnerBid = negotiationSession.getOpponentBidHistory().getLastBid();
				double offeredUtilFromOpponent = getUtility(partnerBid);
				mBidHistory.addOpponentBidnTime(offeredUtilFromOpponent, partnerBid, cTime);

				bidToOffer = chooseBidAction();
			}
		} catch (Exception e) {
			e.printStackTrace();
			/*
			 * if (resU != 0){ bidToOffer=new EndNegotiation(); }else{
			 * bidToOffer=new Accept(getAgentID()); }
			 */
		}
		try {
			nextBid = new BidDetails(bidToOffer, negotiationSession.getUtilitySpace().getUtility(bidToOffer));
		} catch (Exception e) {
			e.printStackTrace();
		}
		return nextBid;
	}

	private Bid chooseBidAction() {
		Bid possibleNextBid = null;
		try {
			if (cTime <= 0.02) {
				possibleNextBid = maxBid;
			} else {
				possibleNextBid = getFinalBid();
			}
		} catch (Exception e) {

		}

		if (possibleNextBid == null) {
			// return (new Accept(getAgentID())); //need to break off
			// negotiation now
			endNegotiation = true;
			return possibleNextBid;
		}

		mBidHistory.addMyBid(possibleNextBid);
		// System.out.println("Decoupled action: " + possibleNextBid);

		return possibleNextBid;
	}

	private Bid getFinalBid() {
		Bid bid = null;
		double upper = 1.01;
		double lower = 0.99;
		double splitFactor = 3.0;
		double val = 0.0;
		double dval = 0.0;
		int delay = 75;
		double laTime = 0D;
		double adp = 1.2;

		if (discount >= discountThreshold) {
			// System.out.println("Decoupled discount > threshold");

			if (cTime <= (double) delay / 100.0) {
				if (resU <= 0.3) {
					return maxBid;
				} else {
					val = EU;
				}

				dval = val * Math.pow(discount, cTime);
				if (opponentModel instanceof NoModel) {
					bid = genRanBid(val * lower, val * upper);
				} else {
					Range range = new Range(val * lower, val * upper);
					bid = omStrategy.getBid(outcomespace, range).getBid();
				}

				return bid;
			} else if (cTime > 0.01 * (tCount + delay)) {
				nextUti = getNextUti();
				tCount++;
			}
		} else {
			if (cTime <= discount / splitFactor) {
				if (resU <= 0.3) {
					return maxBid;
				} else {
					val = EU;
				}

				dval = val * Math.pow(discount, cTime);

				if (opponentModel instanceof NoModel) {
					bid = genRanBid(val * (1.0 - 0.02), val * (1.0 + 0.02));
				} else {
					Range range = new Range(val * (1.0 - 0.02), val * (1.0 + 0.02));
					bid = omStrategy.getBid(outcomespace, range).getBid();
				}
				return bid;
			} else if (cTime > 0.01 * (tCount + (int) Math.floor(discount / splitFactor * 100))) {
				nextUti = getNextUti();
				tCount++;
			}
		}

		if (nextUti == -3.0) {
			if (resU <= 0.3) {
				return maxBid;
			} else {
				val = EU;
			}
		} else if (nextUti == -2.0) {
			val = est_mu + (cTime - est_mt) * (est_u - est_mu) / (est_t - est_mt);
		} else if (nextUti == -1.0) {
			val = getOriU(cTime);
		}

		laTime = mBidHistory.getMCtime() * maxTime;
		if (cTime * maxTime - laTime > 1.5 || cTime > 0.995) {
			dval = val * Math.pow(discount, cTime);

			if (opponentModel instanceof NoModel) {
				bid = genRanBid(dval * lower, dval * upper);
			} else {
				Range range = new Range(dval * lower, dval * upper);
				bid = omStrategy.getBid(outcomespace, range).getBid();
			}

			// System.out.println("Decoupled genRandomBid3: " + bid);

		} else {
			if (val * lower * adp >= maxBidU) {
				bid = maxBid;
				// System.out.println("Decoupled return maxBid3");
			} else {
				dval = adp * val * Math.pow(discount, cTime);
				if (opponentModel instanceof NoModel) {
					bid = genRanBid(dval * lower, dval * upper);
				} else {
					Range range = new Range(dval * lower, dval * upper);
					bid = omStrategy.getBid(outcomespace, range).getBid();
				}
				// System.out.println("Decoupled genRandomBid4: " + bid);
			}
		}

		if (bid == null)
			bid = mBidHistory.getMyLastBid();

		if (getUtility(mBidHistory.bestOppBid) >= getUtility(bid)) {
			// System.out.println("Decoupled bestOppBid1: " +
			// mBidHistory.bestOppBid);
			return mBidHistory.bestOppBid;
		}

		if (cTime > 0.999 && getUtility(mBidHistory.bestOppBid) > MINIMUM_UTILITY * Math.pow(discount, cTime)) {
			// System.out.println("Decoupled bestOppBid2: " +
			// mBidHistory.bestOppBid);

			return mBidHistory.bestOppBid;
		}

		return bid;
	}

	private double getOriU(double t) {
		double exp = 1D;
		double maxUtil = maxBidU;
		double minUtil = 0.69; // 0.7
		if (minUtil < MINIMUM_UTILITY)
			minUtil = MINIMUM_UTILITY * 1.05;

		double e1 = 0.033;
		double e2 = 0.04;
		double time = t;
		double tMax = maxBidU;
		double tMin = MINIMUM_UTILITY * 1.05; // Math.pow(discount, 0.33);

		if (discount >= discountThreshold) {
			exp = minUtil + (1 - Math.pow(time, 1D / e1)) * (maxUtil - minUtil);
		} else {
			tMax = Math.pow(discount, 0.2);
			exp = tMin + (1 - Math.pow(time, 1D / e2)) * (tMax - tMin);
		}

		return exp;
	}

	private double getPre() {
		int len = 3;
		double[] pmaxList = mBidHistory.getTimeBlockList();
		int lenA = (int) Math.floor(cTime * 100); // don't count the current
													// second as it is not
													// complete
		if (lenA < len) {
			return -1.0;
		}

		double[] maxList = new double[lenA];
		double[] ma = new double[lenA];
		double[] res = new double[lenA];
		double exma = 0.0;

		for (int i = 0; i < lenA; i++) {
			maxList[i] = pmaxList[i];
		}

		for (int i = 0; i < len - 1; i++) {
			ma[i] = 0;
		}

		for (int i = len - 1; i < lenA; i++) {
			ma[i] = (maxList[i] + maxList[i - 1] + maxList[i - 2]) / 3.0;
		}

		for (int i = 0; i < lenA; i++) {
			res[i] = maxList[i] - ma[i];
		}

		exma = ma[lenA - 1] + avg(res)
				+ std(res) * (1.0 - Math.pow(maxList[lenA - 1], 4)) * (1.3 + 0.66 * Math.pow(1 - cTime * cTime, 0.4));

		return exma;
	}

	public static double sum(double[] arr) {
		double sum = 0.0;
		int len = arr.length;
		for (int i = 0; i < len; i++) {
			sum += arr[i];
		}
		return sum;
	}

	public static double avg(double[] arr) {
		double avg = 0.0;
		int len = arr.length;
		avg = sum(arr) / (double) len;
		return avg;
	}

	public static double std(double[] arr) {
		double std = 0.0;
		int len = arr.length;
		double ssum = 0.0;
		for (int i = 0; i < len; i++) {
			ssum += arr[i] * arr[i];
		}

		std = ((double) len / (double) (len - 1.0)) * (ssum / (double) len - Math.pow(avg(arr), 2));
		return Math.sqrt(std);
	}

	private double getNextUti() {
		double utiO = getOriU(cTime + timeInt);
		exma = getPre();

		if (exma >= 1.0)
			return -3.0;

		if (exma > utiO) {
			est_t = cTime + timeInt;
			est_u = exma;
			est_mu = getOriU(cTime);
			est_mt = cTime;
			return -2.0;
		} else {
			return -1.0;
		}

	}

	private int getMaxPoint(int cur, float ar[]) {
		int temp = cur;

		for (int i = cur; i < 99; i++) {
			if (ar[temp] < ar[i + 1])
				temp = i + 1;
		}

		return temp;
	}

	private int getMinPoint(int cur, float ar[]) {
		int temp = cur;

		for (int i = cur; i < 99; i++) {
			if (ar[temp] > ar[i + 1])
				temp = i + 1;
		}

		return temp;
	}

	private Bid genRanBid(double min, double max) {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>(); // pairs
																		// <issuenumber,chosen
																		// value
																		// string>
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		int counter = 0;
		int limit = 1000;
		double fmax = max;

		Bid bid = null;
		do {
			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					int optionIndex = randomnr.nextInt(lIssueDiscrete.getNumberOfValues());
					values.put(lIssue.getNumber(), lIssueDiscrete.getValue(optionIndex));
					break;
				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					int optionInd = randomnr.nextInt(lIssueReal.getNumberOfDiscretizationSteps() - 1);
					values.put(lIssueReal.getNumber(),
							new ValueReal(lIssueReal.getLowerBound()
									+ (lIssueReal.getUpperBound() - lIssueReal.getLowerBound()) * (double) (optionInd)
											/ (double) (lIssueReal.getNumberOfDiscretizationSteps())));
					break;
				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					int optionIndex2 = lIssueInteger.getLowerBound()
							+ randomnr.nextInt(lIssueInteger.getUpperBound() - lIssueInteger.getLowerBound());
					values.put(lIssueInteger.getNumber(), new ValueInteger(optionIndex2));
					break;
				default:
					//
				}
			}

			try {
				bid = new Bid(utilitySpace.getDomain(), values);
			} catch (Exception e) {
				System.out.println("error in generating random bids");
			}

			counter++;
			if (counter > limit) {
				limit = limit + 500;
				fmax += 0.005;
				// return mBidHistory.getMyLastBid();
			}

			if (counter > 4000)
				return mBidHistory.getMyLastBid();

		} while (getUtility(bid) < min || getUtility(bid) > fmax);

		return bid;
	}

	public double getUtility(Bid bid) {
		return utilitySpace.getUtilityWithDiscount(bid, negotiationSession.getTimeline());
	}

	@Override
	public String getName() {
		return "2012 - OMACagent";
	}

}
