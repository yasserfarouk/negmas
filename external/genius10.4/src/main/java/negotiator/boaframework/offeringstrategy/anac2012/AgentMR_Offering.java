package negotiator.boaframework.offeringstrategy.anac2012;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

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
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.utility.AdditiveUtilitySpace;
import negotiator.boaframework.opponentmodel.DefaultModel;
import negotiator.boaframework.sharedagentstate.anac2012.AgentMRSAS;

/**
 * This is the decoupled Bidding Strategy of AgentMR
 * 
 * @author Alex Dirkzwager
 */
public class AgentMR_Offering extends OfferingStrategy {

	private boolean EQUIVALENCE_TEST = false;
	private Random random100;
	private ArrayList<Double> observationUtility = new ArrayList<Double>();
	private HashMap<Bid, Double> bidTables = new HashMap<Bid, Double>();
	private static boolean firstOffer;
	private static boolean forecastTime = true;
	private static boolean discountFactor;
	private static BidDetails offereMaxBid = null;
	private static double offereMaxUtility;
	private int currentBidNumber = 0;
	private int lastBidNumber = 1;
	private AdditiveUtilitySpace utilitySpace;
	private boolean alreadyDone = false;
	private SortedOutcomeSpace outcomeSpace;

	public AgentMR_Offering() {
	}

	public AgentMR_Offering(NegotiationSession negoSession, OpponentModel model, OMStrategy oms) throws Exception {
		init(negoSession, model, oms, null);
	}

	/**
	 * Init required for the Decoupled Framework.
	 */
	@Override
	public void init(NegotiationSession negoSession, OpponentModel model, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {
		super.init(negoSession, model, omStrategy, parameters);
		if (model instanceof DefaultModel) {
			model = new NoModel();
		}
		if (!(model instanceof NoModel)) {
			outcomeSpace = new SortedOutcomeSpace(negoSession.getUtilitySpace());
		}
		this.opponentModel = model;
		this.omStrategy = oms;

		helper = new AgentMRSAS(negotiationSession);
		firstOffer = true;
		try {
			utilitySpace = (AdditiveUtilitySpace) negoSession.getUtilitySpace();
			getDiscountFactor();
			getReservationFactor();

			Bid b = negoSession.getMaxBidinDomain().getBid();
			bidTables.put(b, getUtility(b));
			((AgentMRSAS) helper).getBidRunk().add(b);
			if (discountFactor) {
				((AgentMRSAS) helper).setSigmoidGain(-3.0);
				((AgentMRSAS) helper).setPercent(0.55);
			} else {
				((AgentMRSAS) helper).setSigmoidGain(-5.0);
				((AgentMRSAS) helper).setPercent(0.70);
			}
			if (EQUIVALENCE_TEST) {
				random100 = new Random(100);
			} else {
				random100 = new Random();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	@Override
	public BidDetails determineOpeningBid() {

		return determineNextBid();
	}

	@Override
	public BidDetails determineNextBid() {
		if (negotiationSession.getOpponentBidHistory().getHistory().isEmpty()) {
			if (!alreadyDone) {
				((AgentMRSAS) helper).updateMinimumBidUtility(0);
				alreadyDone = true;
			}

			return negotiationSession.getMaxBidinDomain();

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
			double offeredutil;
			if (discountFactor) {
				offeredutil = getUtility(partnerBid.getBid())
						* (1 / Math.pow(negotiationSession.getUtilitySpace().getDiscountFactor(), time));

			} else {
				offeredutil = getUtility(partnerBid.getBid());

			}
			// System.out.println(firstOffer);
			if (firstOffer) {
				// System.out.println("Decoupled partnerBid: " +
				// partnerBid.getBid());
				// System.out.println("Decoupled offeredutil: " + offeredutil);

				offereMaxBid = partnerBid;
				offereMaxUtility = offeredutil;
				((AgentMRSAS) helper).setFirstOffereUtility(offeredutil);

				observationUtility.add(offeredutil);
				if (offeredutil > 0.5) {
					((AgentMRSAS) helper).setP(0.90);
				} else {
					((AgentMRSAS) helper).setP(0.80);
				}
				firstOffer = !firstOffer;
			}
			((AgentMRSAS) helper).updateMinimumBidUtility(time);

			if (offeredutil > offereMaxUtility) {
				offereMaxBid = partnerBid;
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

			if (offereMaxUtility > ((AgentMRSAS) helper).getMinimumBidUtility()) {
				nextBid = offereMaxBid;
			} else if (time > 0.985) {
				if (offereMaxUtility > ((AgentMRSAS) helper).getReservation()) {
					nextBid = offereMaxBid;
				} else {
					Bid nBid = ((AgentMRSAS) helper).getBidRunk()
							.get(((AgentMRSAS) helper).getBidRunk().size() - lastBidNumber);
					nextBid = new BidDetails(nBid, negotiationSession.getUtilitySpace().getUtility(nBid));
					lastBidNumber++;
				}
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
					Bid nBid = ((AgentMRSAS) helper).getBidRunk().get(currentBidNumber);
					nextBid = new BidDetails(nBid, negotiationSession.getUtilitySpace().getUtility(nBid));

					if (currentBidNumber + 1 < ((AgentMRSAS) helper).getBidRunk().size()) {
						currentBidNumber++;
					}

				} else {
					HashMap<Bid, Double> getBids = getBidTable(2);
					if (getBids.size() >= 1) {
						sortBid(getBids); // Sort BidTable
						Bid maxBid = getMaxBidUtility(getBids);
						currentBidNumber = ((AgentMRSAS) helper).getBidRunk().indexOf(maxBid);
					}

					Bid nBid = ((AgentMRSAS) helper).getBidRunk().get(currentBidNumber);
					nextBid = new BidDetails(nBid, negotiationSession.getUtilitySpace().getUtility(nBid));

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

		if (!(opponentModel instanceof NoModel)) {
			try {
				nextBid = omStrategy.getBid(outcomeSpace, utilitySpace.getUtility(nextBid.getBid()));
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return nextBid;

	}

	private void getReservationFactor() {
		if (utilitySpace.getReservationValue() != null) {
			((AgentMRSAS) helper).setReservation(utilitySpace.getReservationValue());
		}
	}

	private void getDiscountFactor() {
		discountFactor = utilitySpace.isDiscounted();
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
			Collections.sort(((AgentMRSAS) helper).getBidRunk(), new Comparator<Bid>() {
				@Override
				public int compare(Bid o1, Bid o2) {
					return (int) Math.ceil(-(bidTables.get(o1) - bidTables.get(o2)));
				}
			});
		}
	}

	private Bid clone(Bid source) throws Exception {
		HashMap<Integer, Value> hash = new HashMap<Integer, Value>();
		for (Issue i : utilitySpace.getDomain().getIssues()) {
			hash.put(i.getNumber(), source.getValue(i.getNumber()));
		}
		return new Bid(utilitySpace.getDomain(), hash);
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
						standardBid = utilitySpace.getMaxUtilityBid();
					} else if (flag == 1) {
						standardBid = negotiationSession.getOpponentBidHistory().getLastBid();
					} else {
						standardBid = ((AgentMRSAS) helper).getBidRunk().get(currentBidNumber);
					}
					standardBid = clone(standardBid);
					standardBid = standardBid.putValue(lIssue.getNumber(), value);
					double utility = getUtility(standardBid);
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

	public double getUtility(Bid bid) {
		return negotiationSession.getUtilitySpace().getUtilityWithDiscount(bid, negotiationSession.getTimeline());
	}

	private void updateSigmoidFunction() {
		int observationSize = observationUtility.size();
		double latestObservation = observationUtility.get(observationSize - 1);
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

	@Override
	public String getName() {
		return "2012 - AgentMR";
	}

}
