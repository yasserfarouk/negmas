package negotiator.boaframework.acceptanceconditions.anac2012;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.utility.AdditiveUtilitySpace;
import negotiator.boaframework.sharedagentstate.anac2012.AgentMRSAS;

/**
 * This is the Acceptance condition of AgentMR Due to the connectedness of the
 * original code if another Bidding strategy is being used, the Bidding Strategy
 * of AgentMR will be run in the background
 * 
 * @author Alexander Dirkzwager
 */
public class AC_AgentMR extends AcceptanceStrategy {

	private boolean EQUIVALENCE_TEST = true;
	private Random random100;
	private ArrayList<Double> observationUtility = new ArrayList<Double>();
	private HashMap<Bid, Double> bidTables = new HashMap<Bid, Double>();
	private static boolean firstOffer;
	private static boolean forecastTime = true;
	private static boolean discountFactor;
	private static double offereMaxUtility;
	private int currentBidNumber = 0;
	private AdditiveUtilitySpace utilitySpace;
	private boolean alreadyDone = false;
	private Actions nextAction;

	private boolean activeHelper = false;
	private static final double MINIMUM_ACCEPT_P = 0.965;

	public AC_AgentMR() {
	}

	public AC_AgentMR(NegotiationSession negoSession, OfferingStrategy strat) throws Exception {
		initializeAgent(negoSession, strat);
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		initializeAgent(negoSession, strat);
	}

	public void initializeAgent(NegotiationSession negotiationSession, OfferingStrategy os) throws Exception {
		this.negotiationSession = negotiationSession;
		this.offeringStrategy = os;

		if (offeringStrategy.getHelper() == null || (!offeringStrategy.getHelper().getName().equals("AgentMR"))) {
			helper = new AgentMRSAS(negotiationSession);
			activeHelper = true;
		} else {
			helper = (AgentMRSAS) offeringStrategy.getHelper();
		}

		if (discountFactor) {
			((AgentMRSAS) helper).setSigmoidGain(-3.0);
			((AgentMRSAS) helper).setPercent(0.55);
		} else {
			((AgentMRSAS) helper).setSigmoidGain(-5.0);
			((AgentMRSAS) helper).setPercent(0.70);
		}

		if (activeHelper) {
			firstOffer = true;
			try {
				utilitySpace = (AdditiveUtilitySpace) negotiationSession.getUtilitySpace();
				getDiscountFactor();
				getReservationFactor();

				Bid b = negotiationSession.getMaxBidinDomain().getBid();
				bidTables.put(b, getUtility(b));
				((AgentMRSAS) helper).getBidRunk().add(b);

				if (EQUIVALENCE_TEST) {
					random100 = new Random(100);
				} else {
					random100 = new Random();
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	@Override
	public Actions determineAcceptability() {
		if (activeHelper)
			nextAction = activeDetermineAcceptability();
		else
			nextAction = regularDetermineAcceptability();
		return nextAction;
	}

	private Actions activeDetermineAcceptability() {
		nextAction = Actions.Reject;
		if (negotiationSession.getOpponentBidHistory().getHistory().isEmpty()) {
			if (!alreadyDone) {
				((AgentMRSAS) helper).updateMinimumBidUtility(0);
				alreadyDone = true;
			}
			return Actions.Reject;
		}

		try {
			BidDetails partnerBid;
			if (firstOffer) {
				partnerBid = negotiationSession.getOpponentBidHistory().getHistory().get(0);
			} else {
				partnerBid = negotiationSession.getOpponentBidHistory().getLastBidDetails();
			}

			// get current time
			double time = negotiationSession.getTime();
			// System.out.println("test: " +
			// negotiationSession.getDiscountedUtility(negotiationSession.getOpponentBidHistory().getFirstBidDetails().getBid(),
			// negotiationSession.getOpponentBidHistory().getFirstBidDetails().getTime()));
			double offeredutil;
			if (discountFactor) {
				offeredutil = getUtility(partnerBid.getBid())
						* (1 / Math.pow(negotiationSession.getUtilitySpace().getDiscountFactor(), time));
			} else {
				offeredutil = getUtility(partnerBid.getBid());

			}
			if (firstOffer) {
				offereMaxUtility = offeredutil;
				((AgentMRSAS) helper).setFirstOffereUtility(offeredutil);

				observationUtility.add(offeredutil); // addObservation
				if (offeredutil > 0.5) {
					((AgentMRSAS) helper).setP(0.90);
				} else {
					((AgentMRSAS) helper).setP(0.80);
				}
				firstOffer = !firstOffer;
			}
			((AgentMRSAS) helper).updateMinimumBidUtility(time);

			/*
			 * if (partnerBid.equals(previousPartnerBid)) { if (currentBidNumber
			 * > 0 && 0.5 > 0.65) { currentBidNumber--; //
			 * ç¢ºçŽ‡çš„ã�«Bidã‚’ç§»å‹• } }
			 */
			if (offeredutil > offereMaxUtility) {
				offereMaxUtility = offeredutil;
				// addObservation
				observationUtility.add(offeredutil);
				if ((time > 0.5) && !discountFactor) {
					newupdateSigmoidFunction();
				}
			}

			// forecasting
			if ((time > 0.5) && forecastTime) {
				updateSigmoidFunction();
				forecastTime = !forecastTime;
			}

			double P = Paccept(offeredutil, time);

			if ((P > MINIMUM_ACCEPT_P)
					|| (negotiationSession.getOpponentBidHistory().getLastBidDetails()
							.getMyUndiscountedUtil() > ((AgentMRSAS) helper).getMinimumBidUtility())
					|| ((AgentMRSAS) helper).getBidRunk()
							.contains(negotiationSession.getOpponentBidHistory().getLastBid())) {
				nextAction = Actions.Accept;
			} else {
				if (offeredutil > ((AgentMRSAS) helper).getMinimumOffereDutil()) {
					HashMap<Bid, Double> getBids = getBidTable(1);
					if (getBids.size() >= 1) {
						// BidTable
						currentBidNumber = 0;
						((AgentMRSAS) helper).getBidRunk().clear();
						bidTables = getBids;
						sortBid(getBids); // Sort BidTable
					} else {
						getBids = getBidTable(2);
						if (getBids.size() >= 1) {
							sortBid(getBids); // Sort BidTable
							Bid maxBid = getMaxBidUtility(getBids);
							currentBidNumber = ((AgentMRSAS) helper).getBidRunk().indexOf(maxBid);
						}
					}
					if (currentBidNumber + 1 < ((AgentMRSAS) helper).getBidRunk().size()) {
						// System.out.println("Decoupled
						// currentBidNumberChange1");

						currentBidNumber++;
					}

				} else {
					HashMap<Bid, Double> getBids = getBidTable(2);

					if (getBids.size() >= 1) {
						sortBid(getBids); // Sort BidTable
						Bid maxBid = getMaxBidUtility(getBids);
						currentBidNumber = ((AgentMRSAS) helper).getBidRunk().indexOf(maxBid);
					}

					if (currentBidNumber + 1 < ((AgentMRSAS) helper).getBidRunk().size()) {
						currentBidNumber++;
					} else {
						currentBidNumber = 0;
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return nextAction;
	}

	private Actions regularDetermineAcceptability() {

		if (!(negotiationSession.getOpponentBidHistory().isEmpty())) {
			double offeredutil = negotiationSession.getOpponentBidHistory().getFirstBidDetails()
					.getMyUndiscountedUtil();
			((AgentMRSAS) helper).setFirstOffereUtility(offeredutil);
		}
		double P;
		try {
			if (negotiationSession.getOpponentBidHistory().isEmpty()) {
				return Actions.Reject;
			}
			P = Paccept(negotiationSession.getOpponentBidHistory().getLastBidDetails().getMyUndiscountedUtil(),
					negotiationSession.getTime());

			if (activeHelper) {
				((AgentMRSAS) helper).updateMinimumBidUtility(negotiationSession.getTime());
			}

			// System.out.println("Decoupled condition1: " + (P >
			// MINIMUM_ACCEPT_P));
			// System.out.println("Decoupled condition2: " +
			// (negotiationSession.getOpponentBidHistory().getLastBidDetails().getMyUndiscountedUtil()
			// > ((AgentMRSAS) helper).getMinimumBidUtility()));
			// System.out.println("Decoupled condition3: " +((AgentMRSAS)
			// helper).getBidRunk().contains(negotiationSession.getOpponentBidHistory().getLastBid()));
			// Accept
			if ((P > MINIMUM_ACCEPT_P)
					|| (negotiationSession.getOpponentBidHistory().getLastBidDetails()
							.getMyUndiscountedUtil() > ((AgentMRSAS) helper).getMinimumBidUtility())
					|| ((AgentMRSAS) helper).getBidRunk()
							.contains(negotiationSession.getOpponentBidHistory().getLastBid())) {
				return Actions.Accept;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		return Actions.Reject;
	}

	private double getUtility(Bid bid) {
		return negotiationSession.getUtilitySpace().getUtilityWithDiscount(bid, negotiationSession.getTimeline());
	}

	private void getReservationFactor() {
		if (utilitySpace.getReservationValue() != null) {
			((AgentMRSAS) helper).setReservation(utilitySpace.getReservationValue());
		}
	}

	private void getDiscountFactor() {
		discountFactor = utilitySpace.isDiscounted();
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
		return (u - 2. * u * t + 2. * (-1. + t + Math.sqrt(sq(-1. + t) + u * (-1. + 2 * t)))) / (-1. + 2 * t);
	}

	double sq(double x) {
		return x * x;
	}

	private void newupdateSigmoidFunction() {
		double latestObservation = observationUtility.get(observationUtility.size() - 1);
		double concessionPercent = Math.abs(latestObservation - ((AgentMRSAS) helper).getFirstOffereUtility())
				/ (1.0 - ((AgentMRSAS) helper).getFirstOffereUtility());
		double modPercent = Math
				.abs(((AgentMRSAS) helper).getMinimumOffereDutil() - ((AgentMRSAS) helper).getFirstOffereUtility())
				/ (1.0 - ((AgentMRSAS) helper).getFirstOffereUtility());

		if (modPercent < concessionPercent) {
			((AgentMRSAS) helper).setPercent(concessionPercent);
		}
	}

	private void updateSigmoidFunction() {
		int observationSize = observationUtility.size();
		double latestObservation = observationUtility.get(observationSize - 1); // æœ€æ–°ã�®ç›¸æ‰‹BidUtil
		double concessionPercent = Math.abs(latestObservation - ((AgentMRSAS) helper).getFirstOffereUtility())
				/ (1.0 - ((AgentMRSAS) helper).getFirstOffereUtility());

		if (discountFactor) {
			if ((concessionPercent < 0.20) || (observationSize < 3)) {
				((AgentMRSAS) helper).setPercent(0.35);
				((AgentMRSAS) helper).setSigmoidGain(-2);
			} else {
				((AgentMRSAS) helper).setPercent(0.45);
			}
		} else {
			if ((concessionPercent < 0.20) || (observationSize < 3)) {
				((AgentMRSAS) helper).setPercent(0.50);
				((AgentMRSAS) helper).setSigmoidGain(-4);
			} else if (concessionPercent > 0.60) {
				((AgentMRSAS) helper).setPercent(0.80);
				((AgentMRSAS) helper).setSigmoidGain(-6);
			} else {
				((AgentMRSAS) helper).setPercent(0.60);
			}
		}
	}

	/**
	 * @param maxBid
	 * @return
	 * @throws Exception
	 */
	private HashMap<Bid, Double> getBidTable(int flag) throws Exception {
		HashMap<Bid, Double> getBids = new HashMap<Bid, Double>();

		// Random randomnr = new Random();
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Bid standardBid = null;

		for (Issue lIssue : issues) {
			switch (lIssue.getType()) {
			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
				for (ValueDiscrete value : lIssueDiscrete.getValues()) {
					if (flag == 0) {
						standardBid = utilitySpace.getMaxUtilityBid(); // è‡ªåˆ†ã�®æœ€é«˜å€¤
					} else if (flag == 1) {
						standardBid = negotiationSession.getOpponentBidHistory().getLastBid();
					} else {
						standardBid = ((AgentMRSAS) helper).getBidRunk().get(currentBidNumber);
					}
					standardBid = clone(standardBid);
					standardBid = standardBid.putValue(lIssue.getNumber(), value);
					double utility = getUtility(standardBid);
					// System.out.println("Decoupled minimumBidUtility: " +
					// ((AgentMRSAS) helper).getMinimumBidUtility());
					if ((utility > ((AgentMRSAS) helper).getMinimumBidUtility())
							&& (!((AgentMRSAS) helper).getBidRunk().contains(standardBid))) {
						getBids.put(standardBid, utility);
					}
				}
				break;
			case REAL:
				IssueReal lIssueReal = (IssueReal) lIssue;
				int optionInd = random100.nextInt(lIssueReal.getNumberOfDiscretizationSteps() - 1);
				Value pValue = new ValueReal(
						lIssueReal.getLowerBound() + (lIssueReal.getUpperBound() - lIssueReal.getLowerBound())
								* (double) (optionInd) / (double) (lIssueReal.getNumberOfDiscretizationSteps()));
				standardBid = standardBid.putValue(lIssueReal.getNumber(), pValue);
				double utility = getUtility(standardBid);
				getBids.put(standardBid, utility);
				break;
			case INTEGER:
				IssueInteger lIssueInteger = (IssueInteger) lIssue;
				int optionIndex2 = lIssueInteger.getLowerBound()
						+ random100.nextInt(lIssueInteger.getUpperBound() - lIssueInteger.getLowerBound());
				Value pValue2 = new ValueInteger(optionIndex2);
				standardBid = standardBid.putValue(lIssueInteger.getNumber(), pValue2);
				double utility2 = getUtility(standardBid);
				getBids.put(standardBid, utility2);
				break;
			default:
				throw new Exception("issue type " + lIssue.getType() + " not supported by AgentMR");
			}
		}

		return getBids;
	}

	/**
	 * BidTable
	 *
	 * @param bidTable
	 */
	private void sortBid(final HashMap<Bid, Double> getBids) {

		for (Bid bid : getBids.keySet()) {
			bidTables.put(bid, getUtility(bid));
			((AgentMRSAS) helper).getBidRunk().add(bid); // Add bidRunk
		}

		if (!EQUIVALENCE_TEST) {
			// Bidã‚½ãƒ¼ãƒˆå‡¦ç�†
			Collections.sort(((AgentMRSAS) helper).getBidRunk(), new Comparator<Bid>() {
				@Override
				public int compare(Bid o1, Bid o2) {
					return (int) Math.ceil(-(bidTables.get(o1) - bidTables.get(o2)));
				}
			});
		}
	}

	private Bid getMaxBidUtility(HashMap<Bid, Double> bidTable) {
		Double maxBidUtility = 0.0;
		Bid maxBid = null;
		for (Bid b : bidTable.keySet()) {
			if (getUtility(b) > maxBidUtility) {
				maxBidUtility = getUtility(b);
				maxBid = b;
			}
		}
		return maxBid;
	}

	private Bid clone(Bid source) throws Exception {
		HashMap<Integer, Value> hash = new HashMap<Integer, Value>();
		for (Issue i : utilitySpace.getDomain().getIssues()) {
			hash.put(i.getNumber(), source.getValue(i.getNumber()));
		}
		return new Bid(utilitySpace.getDomain(), hash);
	}

	@Override
	public String getName() {
		return "2012 - AgentMR";
	}

}
