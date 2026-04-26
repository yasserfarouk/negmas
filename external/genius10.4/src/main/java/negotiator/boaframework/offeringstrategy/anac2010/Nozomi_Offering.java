package negotiator.boaframework.offeringstrategy.anac2010;

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
import genius.core.issue.ISSUETYPE;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.misc.Range;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.Evaluator;
import genius.core.utility.EvaluatorDiscrete;
import negotiator.boaframework.opponentmodel.DefaultModel;
import negotiator.boaframework.sharedagentstate.anac2010.NozomiSAS;

/**
 * This is the decoupled Offering Strategy for Nozomi (ANAC2010). The code was
 * taken from the ANAC2010 Nozomi and adapted to work within the BOA framework.
 * 
 * DEFAULT OM: None
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class Nozomi_Offering extends OfferingStrategy {

	private double maxUtility = 0.0;
	private Bid prevBid = null;
	private Bid prevPartnerBid = null;
	private int continuousCompromiseBid = 0;
	private int continuousPartnerCompromiseBid = 0;
	private int continuousKeep = 0;
	private int measureT = 1;
	private static double COMPROMISE_PROBABILITY = 0.70;
	private static double KEEP_PROBABILITY = 0.15;

	private static enum BidType {
		COMPROMISE, KEEP, APPROACH, RESTORE, RANDOM
	};

	private final boolean TEST_EQUIVALENCE = false;
	private Random random100;
	private Random random200;

	/**
	 * Empty constructor called by BOA framework.
	 */
	public Nozomi_Offering() {
	}

	@Override
	public void init(NegotiationSession domainKnow, OpponentModel model, OMStrategy oms, Map<String, Double> parameters)
			throws Exception {
		initializeAgent(domainKnow, model, oms);
	}

	public void initializeAgent(NegotiationSession negoSession, OpponentModel om, OMStrategy oms) {
		if (om instanceof DefaultModel) {
			om = new NoModel();
		}
		this.negotiationSession = negoSession;
		this.opponentModel = om;
		this.omStrategy = oms;
		helper = new NozomiSAS(negoSession, null);

		if (TEST_EQUIVALENCE) {
			random100 = new Random(100);
			random200 = new Random(200);
		} else {
			random100 = new Random();
			random200 = new Random();
		}
		if (!(opponentModel instanceof NoModel)) {
			SortedOutcomeSpace space = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
			negotiationSession.setOutcomeSpace(space);
		}
		maxUtility = negotiationSession.getMaxBidinDomain().getMyUndiscountedUtil();
		((NozomiSAS) helper).setPrevAverageUtility(maxUtility);
		((NozomiSAS) helper).setMaxCompromiseUtility(maxUtility * 0.95);
	}

	@Override
	public BidDetails determineOpeningBid() {
		BidDetails bid;
		if (negotiationSession.getOpponentBidHistory().getHistory().isEmpty()) {
			bid = negotiationSession.getMaxBidinDomain();
			prevBid = bid.getBid();

		} else {
			bid = determineNextBid();
		}

		try {
		} catch (Exception e) {
			e.printStackTrace();
		}

		return bid;
	}

	@Override
	public BidDetails determineNextBid() {
		nextBid = chooseBidAction();
		prevPartnerBid = negotiationSession.getOpponentBidHistory().getLastBidDetails().getBid();
		try {
		} catch (Exception e) {
			e.printStackTrace();
		}
		return nextBid;
	}

	private BidDetails chooseBidAction() {
		BidDetails partnerBid = negotiationSession.getOpponentBidHistory().getLastBidDetails();

		double partnerUtility = partnerBid.getMyUndiscountedUtil();

		// This if is located in the isAccepted (AC)
		if (partnerUtility > ((NozomiSAS) helper).getMaxUtilityOfPartnerBid()) {
			((NozomiSAS) helper).setMaxUtilityPartnerBidDetails(partnerBid);
			((NozomiSAS) helper).setUpdateMaxPartnerUtility(true);
		}
		try {
			if (prevPartnerBid == null) {
				((NozomiSAS) helper).setMaxUtilityPartnerBidDetails(partnerBid);
				nextBid = negotiationSession.getMaxBidinDomain();
				prevBid = nextBid.getBid();
				return nextBid;
			} else {
				((NozomiSAS) helper).setBidNumber(((NozomiSAS) helper).getBidNumber() + 1);
				double prevPartnerUtility = negotiationSession.getUtilitySpace().getUtility(prevPartnerBid);

				double averageUtil = ((NozomiSAS) helper).getAveragePartnerUtility();
				averageUtil += partnerUtility;
				((NozomiSAS) helper).setAveragePartnerUtility(averageUtil);

				if (partnerUtility < prevPartnerUtility) {
					if (continuousPartnerCompromiseBid < 0)
						continuousPartnerCompromiseBid = 0;
					continuousPartnerCompromiseBid++;
				} else if (partnerUtility > prevPartnerUtility) {
					if (continuousPartnerCompromiseBid > 0)
						continuousPartnerCompromiseBid = 0;
					continuousPartnerCompromiseBid--;
				} else {
					continuousPartnerCompromiseBid = 0;
				}

				double time = negotiationSession.getTime() * 100;

				if (((NozomiSAS) helper).getMaxUtilityOfPartnerBid() > ((NozomiSAS) helper).getMaxCompromiseUtility()) {
					((NozomiSAS) helper).setMaxCompromiseUtility(((NozomiSAS) helper).getMaxUtilityOfPartnerBid());
				}

				if (time > 90 && ((NozomiSAS) helper).getMaxCompromiseUtility()
						- ((NozomiSAS) helper).getMaxUtilityOfPartnerBid() < 0.1) {
					((NozomiSAS) helper).setMaxCompromiseUtility((((NozomiSAS) helper).getMaxCompromiseUtility() * 7.0
							+ ((NozomiSAS) helper).getMaxUtilityOfPartnerBid() * 3.0) / 10.0);
				}
				if (((NozomiSAS) helper).getMaxCompromiseUtility() > maxUtility * 0.95) {
					((NozomiSAS) helper).setMaxCompromiseUtility(maxUtility * 0.95);
				}
				Bid bid2Offer = prevBid;
				BidType bidType = chooseBidType(partnerBid.getBid());

				switch (bidType) {
				case RESTORE:
					bid2Offer = ((NozomiSAS) helper).getRestoreBid();
					break;
				case APPROACH:

					bid2Offer = getApproachBid(partnerBid.getBid());

					if (negotiationSession.getUtilitySpace().getUtility(bid2Offer) < ((NozomiSAS) helper)
							.getMaxCompromiseUtility()
							|| negotiationSession.getUtilitySpace().getUtility(
									bid2Offer) > ((NozomiSAS) helper).getMaxCompromiseUtility() * 105 / 95) {
						bid2Offer = getBetweenBid(bid2Offer, partnerBid.getBid(),
								((NozomiSAS) helper).getMaxCompromiseUtility(),
								((NozomiSAS) helper).getMaxCompromiseUtility() * 105 / 95);
					}
					break;
				case COMPROMISE:
					bid2Offer = getCompromiseBid(partnerBid.getBid());

					if (negotiationSession.getUtilitySpace().getUtility(bid2Offer) < ((NozomiSAS) helper)
							.getMaxCompromiseUtility()) {
						bid2Offer = getBetweenBid(bid2Offer, partnerBid.getBid(),
								((NozomiSAS) helper).getMaxCompromiseUtility(),
								((NozomiSAS) helper).getMaxCompromiseUtility() * 105 / 95);
					}

					if (continuousCompromiseBid < 0)
						continuousCompromiseBid = 0;
					continuousCompromiseBid++;
					break;
				case KEEP:
					bid2Offer = prevBid;
					continuousCompromiseBid /= 2;
					break;
				}

				if (negotiationSession.getUtilitySpace()
						.getUtility(bid2Offer) <= negotiationSession.getUtilitySpace().getUtility(prevBid) / 2.0) {
					bid2Offer = getApproachBid(partnerBid.getBid());

					if (negotiationSession.getUtilitySpace().getUtility(bid2Offer) < ((NozomiSAS) helper)
							.getMaxCompromiseUtility()
							|| negotiationSession.getUtilitySpace().getUtility(
									bid2Offer) > ((NozomiSAS) helper).getMaxCompromiseUtility() * 105 / 95) {
						bid2Offer = getBetweenBid(bid2Offer, partnerBid.getBid(),
								((NozomiSAS) helper).getMaxCompromiseUtility(),
								((NozomiSAS) helper).getMaxCompromiseUtility() * 105 / 95);
					}
				}

				if (((negotiationSession.getUtilitySpace().getUtility(bid2Offer) < ((NozomiSAS) helper)
						.getMaxCompromiseUtility()
						|| negotiationSession.getUtilitySpace()
								.getUtility(bid2Offer) > ((NozomiSAS) helper).getMaxCompromiseUtility() * 105 / 95)
						&& continuousKeep < 20)
						|| negotiationSession.getUtilitySpace()
								.getUtility(bid2Offer) <= negotiationSession.getUtilitySpace().getUtility(prevBid) / 2.0
						|| negotiationSession.getUtilitySpace().getUtility(bid2Offer) < ((NozomiSAS) helper)
								.getMaxUtilityOfPartnerBid()) {
					bid2Offer = ((NozomiSAS) helper).getRestoreBid();
				}

				double util = ((NozomiSAS) helper).getAverageUtility();
				util += negotiationSession.getUtilitySpace().getUtility(bid2Offer);
				((NozomiSAS) helper).setAverageUtility(util);

				if (continuousKeep <= 0) {
					((NozomiSAS) helper).updateRestoreBid(bid2Offer);
				}

				if (time > (double) measureT * 3.0) {
					((NozomiSAS) helper).checkCompromise(time);
					((NozomiSAS) helper).setAverageUtility(0.0);
					((NozomiSAS) helper).setAveragePartnerUtility(0.0);
					((NozomiSAS) helper).setBidNumber(1);
					measureT++;
				}

				if (negotiationSession.getUtilitySpace().getUtility(bid2Offer) == negotiationSession.getUtilitySpace()
						.getUtility(prevBid)) {
					continuousKeep++;
				} else {
					continuousKeep = 0;
				}

				if (continuousKeep > 30 && time > 90.0) {
					double utility = ((NozomiSAS) helper).getMaxCompromiseUtility();
					utility *= 0.99;

					((NozomiSAS) helper).setMaxCompromiseUtility(utility);

					continuousKeep = 0;
				}
				nextBid = new BidDetails(bid2Offer, negotiationSession.getUtilitySpace().getUtility(bid2Offer),
						negotiationSession.getTime());
				prevBid = bid2Offer;

				return nextBid;
			}
		} catch (Exception e) {
			// should throw an accept when fail
			e.printStackTrace();
		}
		return null;
	}

	private BidType chooseBidType(Bid partnerBid) {
		BidType bidType = null;

		try {
			double time = negotiationSession.getTime();
			double prevUtility = negotiationSession.getUtilitySpace().getUtility(prevBid);
			double partnerUtility = negotiationSession.getUtilitySpace().getUtility(partnerBid);
			double gap = Math.abs(prevUtility - partnerUtility);

			if (gap < 0.05)
				return BidType.APPROACH;

			// this is alway 0 thus I made it so
			// gap = Math.abs(prevAverageUtility - prevAverageUtility);
			gap = 0;
			if (gap < 0.10 && time > 0.80)
				return BidType.APPROACH;

			if (random100.nextInt(20) <= 0)
				return BidType.RESTORE;

			int approachBorder = 5;
			if (time > 0.9) {
				approachBorder = (int) Math.round((time * 100) / 10);
			}
			if (random100.nextInt(20) <= approachBorder)
				return BidType.RESTORE;

			if (prevUtility > ((NozomiSAS) helper).getMaxCompromiseUtility() * 105 / 95) {
				return BidType.COMPROMISE;
			}
			if (continuousKeep > 20 && time > 0.60 && random100.nextDouble() > 0.60)
				return BidType.APPROACH;

		} catch (Exception e) {
			bidType = BidType.RESTORE; // bidType;
		}

		double rnd = random100.nextDouble();
		double compromiseProbability = getCompromiseProbability();

		if (rnd < compromiseProbability) {
			bidType = BidType.COMPROMISE;

		} else if (rnd < getKeepProbability(compromiseProbability) + compromiseProbability) {
			bidType = BidType.KEEP;
		}
		if (bidType == null)
			bidType = BidType.RESTORE;
		return bidType;
	}

	private double getCompromiseProbability() {
		double compromiseProbabilty = 0.0;

		try {
			double prevUtility = negotiationSession.getUtilitySpace().getUtility(prevBid);
			double prevUtilityOfPartnerBid = negotiationSession.getUtilitySpace().getUtility(prevPartnerBid);
			// Below Values are less and less, more and more compromised.
			double compromiseDegree = prevUtility / maxUtility;

			double compromisePartnerDegree = 1.0 - prevUtilityOfPartnerBid / maxUtility;

			double continuous = ((double) continuousCompromiseBid + (double) Math.abs(continuousPartnerCompromiseBid)
					+ 0.001) / 2.0;
			double ratioUtilityToMaxUtility = prevUtility / (maxUtility * 0.9);

			compromiseProbabilty = (compromiseDegree + compromisePartnerDegree) / 2.0;

			compromiseProbabilty /= 1.0 + continuous / 8.0;
			compromiseProbabilty *= ratioUtilityToMaxUtility;
		} catch (Exception e) {
			compromiseProbabilty = COMPROMISE_PROBABILITY;
		}

		return compromiseProbabilty;
	}

	private double getKeepProbability(double compromiseProbability) {
		double keepProbability = 1.0 - compromiseProbability;

		try {
			double time = negotiationSession.getTime();
			double prevUtility = negotiationSession.getUtilitySpace().getUtility(prevBid);
			double prevUtilityOfPartnerBid = negotiationSession.getUtilitySpace().getUtility(prevPartnerBid);
			double ratioUtilityToMaxUtility = prevUtility / (maxUtility * 0.8);

			if (prevUtility > prevUtilityOfPartnerBid) {
				keepProbability *= (prevUtility - prevUtilityOfPartnerBid) / maxUtility;
				keepProbability *= time;
				keepProbability *= 1.0 + (double) Math.abs(continuousCompromiseBid) / 4.0;
				keepProbability *= ratioUtilityToMaxUtility;
			} else {
				keepProbability = 0.0;
			}
		} catch (Exception e) {
			keepProbability = KEEP_PROBABILITY;
		}

		return keepProbability;
	}

	private Bid getCompromiseBid(Bid partnerBid) {
		Bid compromiseBid = prevBid;
		try {
			double compromiseUtility = 0.0;
			double prevUtility = negotiationSession.getUtilitySpace().getUtility(prevBid);
			double basicUtility = prevUtility;
			double gap = Math.abs(prevUtility - negotiationSession.getUtilitySpace().getUtility(partnerBid));
			HashMap<Integer, Value> values = new HashMap<Integer, Value>();
			List<Issue> issues = negotiationSession.getUtilitySpace().getDomain().getIssues();

			for (Issue issue : issues) {
				values.put(issue.getNumber(), prevBid.getValue(issue.getNumber()));
			}

			Integer changeIssueID = -1;
			for (Issue compromiseIssue : issues) {
				Integer compromiseIssueID = compromiseIssue.getNumber();

				if ((!prevPartnerBid.getValue(compromiseIssueID).equals(partnerBid.getValue(compromiseIssueID))
						&& gap > 0.05 && continuousKeep < 20)
						|| prevBid.getValue(compromiseIssueID).equals(partnerBid.getValue(compromiseIssueID))
						|| compromiseIssueID == changeIssueID) {
					continue;
				}

				HashMap<Integer, Value> candidateValues = new HashMap<Integer, Value>(values);
				candidateValues.remove(compromiseIssueID);
				candidateValues.put(compromiseIssueID, getCompromiseValue(compromiseIssue,
						prevBid.getValue(compromiseIssueID), partnerBid.getValue(compromiseIssueID)));
				Bid candidateBid = new Bid(negotiationSession.getUtilitySpace().getDomain(), candidateValues);
				double candidateUtility = negotiationSession.getUtilitySpace().getUtility(candidateBid);

				if (candidateUtility > compromiseUtility && candidateUtility < basicUtility) {

					compromiseUtility = candidateUtility;
					compromiseBid = candidateBid;
					changeIssueID = compromiseIssueID;
				}
			}

		} catch (Exception e) {
			e.printStackTrace();
			compromiseBid = ((NozomiSAS) helper).getRestoreBid(); // best guess
																	// if things
																	// go wrong.
		}
		try {
		} catch (Exception e) {
			e.printStackTrace();
		}
		return compromiseBid;
	}

	private Value getCompromiseValue(Issue issue, Value val, Value partnerVal) {
		Value compromiseVal = null;
		Evaluator eval = ((AdditiveUtilitySpace) negotiationSession.getUtilitySpace()).getEvaluator(issue.getNumber());

		switch (issue.getType()) {
		case DISCRETE:
			EvaluatorDiscrete evalDiscrete = (EvaluatorDiscrete) eval;
			compromiseVal = (ValueDiscrete) val;
			Integer evaluation = evalDiscrete.getValue((ValueDiscrete) val);
			IssueDiscrete issueDiscrete = (IssueDiscrete) issue;
			Integer compromiseEvaluation = 0;
			for (int i = 0; i < issueDiscrete.getNumberOfValues(); i++) {
				ValueDiscrete candidateVal = issueDiscrete.getValue(i);
				Integer candidateEvaluation = evalDiscrete.getValue(candidateVal);
				if (candidateEvaluation >= compromiseEvaluation && candidateEvaluation < evaluation) {
					compromiseVal = candidateVal;
					compromiseEvaluation = candidateEvaluation;
				}
			}
			break;
		case INTEGER:
			ValueInteger valInt = (ValueInteger) val;
			int compromiseInt = valInt.getValue();
			ValueInteger partnerValInt = (ValueInteger) partnerVal;
			if (compromiseInt > partnerValInt.getValue()) {
				compromiseInt--;
			} else {
				compromiseInt++;
			}
			compromiseVal = new ValueInteger(compromiseInt);
			break;
		case REAL:
			ValueReal valReal = (ValueReal) val;
			double compromiseReal = valReal.getValue();
			ValueReal partnerValReal = (ValueReal) partnerVal;
			compromiseReal += new Random().nextDouble() * (partnerValReal.getValue() - compromiseReal) / 10;
			compromiseVal = new ValueReal(compromiseReal);
			break;
		default:
			compromiseVal = val;
			break;
		}
		return compromiseVal;
	}

	private Bid getBetweenBid(Bid basicBid, Bid partnerBid, double minUtility, double maxUtility) {
		Bid betweenBid = basicBid;
		if (opponentModel instanceof NoModel) {
			try {
				List<Issue> issues = negotiationSession.getUtilitySpace().getDomain().getIssues();
				HashMap<Integer, Value> values = new HashMap<Integer, Value>();

				for (Issue issue : issues) {
					values.put(issue.getNumber(), basicBid.getValue(issue.getNumber()));
				}

				double utility = negotiationSession.getUtilitySpace().getUtility(basicBid);
				for (int i = 0; i < 1000; i++) {
					Issue issue = issues.get(random200.nextInt(issues.size()));

					int issueID = issue.getNumber();

					if (prevBid.getValue(issueID).equals(partnerBid.getValue(issueID))) {
						continue;
					}

					Value value = values.get(issueID);
					values.remove(issueID);
					if (utility > maxUtility) {
						values.put(issueID, getLowerValue(issue, value, values));
					} else {
						values.put(issueID, getHigherValue(issue, value, values));
					}

					Bid candidateBid = new Bid(negotiationSession.getUtilitySpace().getDomain(), values);
					utility = negotiationSession.getUtilitySpace().getUtility(candidateBid);
					if (utility > minUtility && utility < maxUtility) {
						betweenBid = candidateBid;
						break;
					}
				}
			} catch (Exception e) {
				betweenBid = ((NozomiSAS) helper).getRestoreBid();
			}
		} else {
			betweenBid = omStrategy.getBid(negotiationSession.getOutcomeSpace(), new Range(minUtility, maxUtility))
					.getBid();
		}
		return betweenBid;
	}

	private Value getHigherValue(Issue issue, Value value, HashMap<Integer, Value> values) {
		Value higher = null;
		Evaluator eval = ((AdditiveUtilitySpace) negotiationSession.getUtilitySpace()).getEvaluator(issue.getNumber());

		switch (issue.getType()) {
		case DISCRETE:
			EvaluatorDiscrete evalDiscrete = (EvaluatorDiscrete) eval;

			Integer evaluation = evalDiscrete.getValue((ValueDiscrete) value);
			IssueDiscrete issueDiscrete = (IssueDiscrete) issue;
			higher = evalDiscrete.getMaxValue();
			Integer highEvaluation = evalDiscrete.getValue((ValueDiscrete) higher);
			for (int i = 0; i < issueDiscrete.getNumberOfValues(); i++) {
				ValueDiscrete candidateVal = issueDiscrete.getValue(i);
				Integer candidateEvaluation = evalDiscrete.getValue(candidateVal);
				if (candidateEvaluation < highEvaluation && candidateEvaluation > evaluation) {
					higher = candidateVal;
					highEvaluation = candidateEvaluation;
				}
			}
			break;
		case INTEGER:
			IssueInteger issueInt = (IssueInteger) issue;
			ValueInteger valInt = (ValueInteger) value;
			try {
				Bid bid = new Bid(negotiationSession.getUtilitySpace().getDomain(), values);
				values.remove(issueInt.getNumber());
				values.put(issueInt.getNumber(), new ValueInteger(valInt.getValue() - 1));
				Bid candidateBid = new Bid(negotiationSession.getUtilitySpace().getDomain(), values);
				if (negotiationSession.getUtilitySpace().getUtility(candidateBid) > negotiationSession.getUtilitySpace()
						.getUtility(bid)) {
					higher = new ValueInteger(valInt.getValue() - 1);
				} else {
					higher = new ValueInteger(valInt.getValue() + 1);
				}
			} catch (Exception e) {
				higher = valInt;
			}
			break;
		case REAL:
			IssueReal issueReal = (IssueReal) issue;
			ValueReal valReal = (ValueReal) value;
			try {
				Bid bid = new Bid(negotiationSession.getUtilitySpace().getDomain(), values);
				values.remove(issueReal.getNumber());
				values.put(issueReal.getNumber(),
						new ValueReal(valReal.getValue() - (valReal.getValue() - issueReal.getLowerBound()) / 10));
				Bid candidateBid = new Bid(negotiationSession.getUtilitySpace().getDomain(), values);
				if (negotiationSession.getUtilitySpace().getUtility(candidateBid) > negotiationSession.getUtilitySpace()
						.getUtility(bid)) {
					higher = new ValueReal(valReal.getValue()
							- new Random().nextDouble() * (valReal.getValue() - issueReal.getLowerBound()) / 10);
				} else {
					higher = new ValueReal(valReal.getValue()
							+ new Random().nextDouble() * (issueReal.getUpperBound() - valReal.getValue()) / 10);
				}
			} catch (Exception e) {
				higher = valReal;
			}
			break;
		default:
			higher = value;
			break;
		}

		return higher;
	}

	private Value getLowerValue(Issue issue, Value value, HashMap<Integer, Value> values) {
		Value lower = null;
		Evaluator eval = ((AdditiveUtilitySpace) negotiationSession.getUtilitySpace()).getEvaluator(issue.getNumber());

		switch (issue.getType()) {
		case DISCRETE:
			EvaluatorDiscrete evalDiscrete = (EvaluatorDiscrete) eval;

			Integer evaluation = evalDiscrete.getValue((ValueDiscrete) value);
			IssueDiscrete issueDiscrete = (IssueDiscrete) issue;
			lower = evalDiscrete.getMinValue();
			Integer lowEvaluation = evalDiscrete.getValue((ValueDiscrete) lower);
			for (int i = 0; i < issueDiscrete.getNumberOfValues(); i++) {
				ValueDiscrete candidateVal = issueDiscrete.getValue(i);
				Integer candidateEvaluation = evalDiscrete.getValue(candidateVal);
				if (candidateEvaluation > lowEvaluation && candidateEvaluation < evaluation) {
					lower = candidateVal;
					lowEvaluation = candidateEvaluation;
				}
			}
			break;
		case INTEGER:
			IssueInteger issueInt = (IssueInteger) issue;
			ValueInteger valInt = (ValueInteger) value;
			try {
				Bid bid = new Bid(negotiationSession.getUtilitySpace().getDomain(), values);
				values.remove(issueInt.getNumber());
				values.put(issueInt.getNumber(), new ValueInteger(valInt.getValue() - 1));
				Bid candidateBid = new Bid(negotiationSession.getUtilitySpace().getDomain(), values);
				if (negotiationSession.getUtilitySpace().getUtility(candidateBid) < negotiationSession.getUtilitySpace()
						.getUtility(bid)) {
					lower = new ValueInteger(valInt.getValue() - 1);
				} else {
					lower = new ValueInteger(valInt.getValue() + 1);
				}
			} catch (Exception e) {
				lower = valInt;
			}
			break;
		case REAL:
			IssueReal issueReal = (IssueReal) issue;
			ValueReal valReal = (ValueReal) value;
			try {
				Bid bid = new Bid(negotiationSession.getUtilitySpace().getDomain(), values);
				values.remove(issueReal.getNumber());
				values.put(issueReal.getNumber(),
						new ValueReal(valReal.getValue() - (valReal.getValue() - issueReal.getLowerBound()) / 10));
				Bid candidateBid = new Bid(negotiationSession.getUtilitySpace().getDomain(), values);
				if (negotiationSession.getUtilitySpace().getUtility(candidateBid) < negotiationSession.getUtilitySpace()
						.getUtility(bid)) {
					lower = new ValueReal(valReal.getValue()
							- new Random().nextDouble() * (valReal.getValue() - issueReal.getLowerBound()) / 10);
				} else {
					lower = new ValueReal(valReal.getValue()
							+ new Random().nextDouble() * (issueReal.getUpperBound() - valReal.getValue()) / 10);
				}
			} catch (Exception e) {
				lower = valReal;
			}
			break;
		default:
			lower = value;
			break;
		}
		return lower;
	}

	private Bid getApproachBid(Bid partnerBid) {
		Bid approachBid = prevBid;
		try {
			HashMap<Integer, Value> values = new HashMap<Integer, Value>();
			HashMap<Integer, Value> approachValues = new HashMap<Integer, Value>();
			List<Issue> issues = negotiationSession.getUtilitySpace().getDomain().getIssues();

			for (Issue issue : issues) {
				values.put(issue.getNumber(), prevBid.getValue(issue.getNumber()));
			}
			approachValues = values;

			double candidateUtility = 0.0;
			for (Issue issue : issues) {
				int issueID = issue.getNumber();

				if (issue.getType() == ISSUETYPE.REAL) {
					IssueReal issueReal = (IssueReal) issue;
					ValueReal valueReal = (ValueReal) prevBid.getValue(issueID);
					ValueReal partnerValueReal = (ValueReal) partnerBid.getValue(issueID);
					ValueReal prevPartnerValueReal = (ValueReal) prevPartnerBid.getValue(issueID);
					double gap = Math.abs(valueReal.getValue() - partnerValueReal.getValue());
					double gapPartnerValue = Math.abs(partnerValueReal.getValue() - prevPartnerValueReal.getValue());
					if (gap < (issueReal.getUpperBound() - issueReal.getLowerBound()) / 100
							|| !(gapPartnerValue < (issueReal.getUpperBound() - issueReal.getLowerBound()) / 100)) {
						continue;
					}
				}

				if (prevBid.getValue(issueID).equals(partnerBid.getValue(issueID))) {
					continue;
				}

				HashMap<Integer, Value> candidateValues = new HashMap<Integer, Value>(values);

				candidateValues.remove(issueID);
				candidateValues.put(issueID,
						getApproachValue(issue, prevBid.getValue(issueID), partnerBid.getValue(issueID)));
				Bid candidateBid = new Bid(negotiationSession.getUtilitySpace().getDomain(), candidateValues);
				if (negotiationSession.getUtilitySpace().getUtility(candidateBid) > candidateUtility) {
					candidateUtility = negotiationSession.getUtilitySpace().getUtility(candidateBid);
					approachBid = candidateBid;
					approachValues = candidateValues;
				}
			}

			while (true) {
				candidateUtility = 0.0;
				HashMap<Integer, Value> candidateValues = new HashMap<Integer, Value>(approachValues);
				for (Issue issue : issues) {
					HashMap<Integer, Value> tempValues = new HashMap<Integer, Value>(approachValues);

					int issueID = issue.getNumber();

					if (issue.getType() == ISSUETYPE.REAL) {
						IssueReal issueReal = (IssueReal) issue;
						ValueReal valueReal = (ValueReal) prevBid.getValue(issueID);
						ValueReal partnerValueReal = (ValueReal) partnerBid.getValue(issueID);
						double gap = Math.abs(valueReal.getValue() - partnerValueReal.getValue());
						if (gap < (issueReal.getUpperBound() - issueReal.getLowerBound()) / 100) {
							continue;
						}
					}

					if (prevBid.getValue(issueID).equals(partnerBid.getValue(issueID))) {
						continue;
					}

					tempValues.remove(issueID);
					tempValues.put(issueID,
							getApproachValue(issue, prevBid.getValue(issueID), partnerBid.getValue(issueID)));

					Bid tempBid = new Bid(negotiationSession.getUtilitySpace().getDomain(), tempValues);
					if (negotiationSession.getUtilitySpace().getUtility(tempBid) > candidateUtility) {
						candidateValues = tempValues;
						candidateUtility = negotiationSession.getUtilitySpace().getUtility(tempBid);
					}
				}
				if (candidateUtility >= negotiationSession.getUtilitySpace().getUtility(approachBid)
						&& !candidateValues.equals(approachValues)) {
					Bid candidateBid = new Bid(negotiationSession.getUtilitySpace().getDomain(), candidateValues);
					approachBid = candidateBid;
					approachValues = candidateValues;
				} else {
					break;
				}
			}
		} catch (Exception e) {
			approachBid = ((NozomiSAS) helper).getRestoreBid(); // best guess if
																// things go
																// wrong.
		}
		return approachBid;
	}

	private Value getApproachValue(Issue issue, Value myVal, Value partnerVal) {
		Value approachVal = null;

		switch (issue.getType()) {
		case DISCRETE:
			IssueDiscrete issueDiscrete = (IssueDiscrete) issue;

			boolean checkMyVal = false;
			boolean checkPartnerVal = false;
			for (Value value : issueDiscrete.getValues()) {
				if (checkMyVal) {
					approachVal = value;
					break;
				}

				if (myVal.equals(value)) {
					if (checkPartnerVal) {
						break;
					} else {
						checkMyVal = true;
					}
				}

				if (partnerVal.equals(value)) {
					if (checkMyVal) {
						approachVal = value;
						break;
					} else {
						checkPartnerVal = true;
					}
				}

				approachVal = value;
			}
			break;
		case INTEGER:
			ValueInteger valInt = (ValueInteger) myVal;
			int approachInt = valInt.getValue();
			ValueInteger partnerValInt = (ValueInteger) partnerVal;
			if (approachInt > partnerValInt.getValue()) {
				approachInt--;
			} else {
				approachInt++;
			}
			approachVal = new ValueInteger(approachInt);
			break;
		case REAL:
			ValueReal valReal = (ValueReal) myVal;
			double approachReal = valReal.getValue();
			ValueReal partnerValReal = (ValueReal) partnerVal;
			approachReal += new Random().nextDouble() * (partnerValReal.getValue() - approachReal) / 10;
			approachVal = new ValueReal(approachReal);
			break;
		default:
			approachVal = myVal;
			break;
		}
		return approachVal;
	}

	@Override
	public String getName() {
		return "2010 - Nozomi";
	}
}
