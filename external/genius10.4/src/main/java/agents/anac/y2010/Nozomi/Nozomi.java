package agents.anac.y2010.Nozomi;

import java.util.HashMap;
import java.util.List;
import java.util.Random;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.SupportedNegotiationSetting;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Offer;
import genius.core.issue.ISSUETYPE;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.Evaluator;
import genius.core.utility.EvaluatorDiscrete;

/**
 * ANAC2010 competitor Nozomi.
 */
public class Nozomi extends Agent {
	private Bid maxUtilityBid = null;
	private double maxUtility = 0.0;
	private Bid prevBid = null;
	private Bid restoreBid = null;
	private double minGap = 0.0;
	private Action actionOfPartner = null;
	private Bid prevPartnerBid = null;
	private Bid maxUtilityPartnerBid = null;
	private double maxUtilityOfPartnerBid = 0.0;
	private double maxCompromiseUtility = 1.0;
	private boolean compromise = false;
	private boolean updateMaxPartnerUtility = false;
	private int continuousCompromiseBid = 0;
	private int continuousPartnerCompromiseBid = 0;
	private int continuousKeep = 0;
	private int measureT = 1;
	private int bidNumber = 0;
	private double prevAverageUtility = 0.0;
	private double averageUtility = 0.0;
	private double averagePartnerUtility = 0.0;
	private double prevAveragePartnerUtility = 0.0;
	private static double COMPROMISE_PROBABILITY = 0.70;
	private static double KEEP_PROBABILITY = 0.15;
	private final boolean TEST_EQUIVALENCE = false;
	private Random random100;
	private Random random200;

	private static enum BidType {
		COMPROMISE, KEEP, APPROACH, RESTORE, RANDOM
	};

	/**
	 * init is called when a next session starts with the same opponent.
	 */

	@Override
	public void init() {
		try {
			maxUtilityBid = utilitySpace.getMaxUtilityBid();
			maxUtility = utilitySpace.getUtility(maxUtilityBid);
			prevAverageUtility = maxUtility;
			maxCompromiseUtility = maxUtility * 0.95;
			restoreBid = maxUtilityBid;
			minGap = utilitySpace.getDomain().getIssues().size();

			if (TEST_EQUIVALENCE) {
				random100 = new Random(100);
				random200 = new Random(200);
			} else {
				random100 = new Random();
				random200 = new Random();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public String getVersion() {
		return "1.0";
	}

	@Override
	public String getName() {
		return "Nozomi";
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
	}

	@Override
	public Action chooseAction() {
		Action action = null;

		try {
			if (actionOfPartner == null) {
				action = new Offer(getAgentID(), maxUtilityBid);
				prevBid = maxUtilityBid;
			} else if (actionOfPartner instanceof Offer) {
				Bid partnerBid = ((Offer) actionOfPartner).getBid();

				if (isAccept(partnerBid)) {
					action = new Accept(getAgentID(), partnerBid);
				} else {
					action = chooseBidAction(partnerBid);
				}
				prevPartnerBid = partnerBid;
			}
		} catch (Exception e) {
			action = new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid());
		}
		return action;
	}

	private Action chooseBidAction(Bid partnerBid) {
		Action action = null;

		try {
			if (prevPartnerBid == null) {
				maxUtilityPartnerBid = partnerBid;
				maxUtilityOfPartnerBid = utilitySpace.getUtility(partnerBid);
				action = new Offer(getAgentID(), maxUtilityBid);

				prevBid = maxUtilityBid;
			} else {
				bidNumber++;
				double prevPartnerUtility = utilitySpace
						.getUtility(prevPartnerBid);
				double partnerUtility = utilitySpace.getUtility(partnerBid);

				averagePartnerUtility += partnerUtility;
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

				double time = timeline.getTime() * 100;

				if (maxUtilityOfPartnerBid > maxCompromiseUtility) {
					maxCompromiseUtility = maxUtilityOfPartnerBid;
				}

				if (time > 90 && maxCompromiseUtility
						- maxUtilityOfPartnerBid < 0.1) {
					maxCompromiseUtility = (maxCompromiseUtility * 7.0
							+ maxUtilityOfPartnerBid * 3.0) / 10.0;
				}

				if (maxCompromiseUtility > maxUtility * 0.95) {
					maxCompromiseUtility = maxUtility * 0.95;
				}

				Bid nextBid = prevBid;
				BidType bidType = chooseBidType(partnerBid);

				switch (bidType) {
				case RESTORE:
					nextBid = restoreBid;
					break;
				case APPROACH:
					nextBid = getApproachBid(partnerBid);

					if (utilitySpace.getUtility(nextBid) < maxCompromiseUtility
							|| utilitySpace
									.getUtility(nextBid) > maxCompromiseUtility
											* 105 / 95) {
						nextBid = getBetweenBid(nextBid, partnerBid,
								maxCompromiseUtility,
								maxCompromiseUtility * 105 / 95);
					}
					break;
				case COMPROMISE:
					nextBid = getCompromiseBid(partnerBid);

					if (utilitySpace
							.getUtility(nextBid) < maxCompromiseUtility) {
						nextBid = getBetweenBid(nextBid, partnerBid,
								maxCompromiseUtility,
								maxCompromiseUtility * 105 / 95);
					}

					if (continuousCompromiseBid < 0)
						continuousCompromiseBid = 0;
					continuousCompromiseBid++;
					break;
				case KEEP:
					nextBid = prevBid;
					continuousCompromiseBid /= 2;
					break;
				}

				if (utilitySpace.getUtility(
						nextBid) <= utilitySpace.getUtility(prevBid) / 2.0) {
					nextBid = getApproachBid(partnerBid);

					if (utilitySpace.getUtility(nextBid) < maxCompromiseUtility
							|| utilitySpace
									.getUtility(nextBid) > maxCompromiseUtility
											* 105 / 95) {
						nextBid = getBetweenBid(nextBid, partnerBid,
								maxCompromiseUtility,
								maxCompromiseUtility * 105 / 95);
					}
				}

				if (((utilitySpace.getUtility(nextBid) < maxCompromiseUtility
						|| utilitySpace.getUtility(
								nextBid) > maxCompromiseUtility * 105 / 95)
						&& continuousKeep < 20)
						|| utilitySpace.getUtility(
								nextBid) <= utilitySpace.getUtility(prevBid)
										/ 2.0
						|| utilitySpace
								.getUtility(nextBid) < maxUtilityOfPartnerBid) {

					nextBid = restoreBid;
				}

				averageUtility += utilitySpace.getUtility(nextBid);

				if (continuousKeep <= 0) {
					updateRestoreBid(nextBid);
				}

				if (time > measureT * 3.0) {
					checkCompromise(time);
					averageUtility = 0.0;
					averagePartnerUtility = 0.0;
					bidNumber = 1;
					measureT++;
				}

				if (utilitySpace.getUtility(nextBid) == utilitySpace
						.getUtility(prevBid)) {
					continuousKeep++;
				} else {
					continuousKeep = 0;
				}

				if (continuousKeep > 30 && time > 90.0) {
					maxCompromiseUtility *= 0.99;
					continuousKeep = 0;
				}

				prevBid = nextBid;
				action = new Offer(getAgentID(), nextBid);
			}
		} catch (Exception e) {
			action = new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid());
		}
		return action;
	}

	private void checkCompromise(double time) {
		averageUtility /= bidNumber;
		averagePartnerUtility /= bidNumber;

		double diff = prevAverageUtility - averageUtility;
		double partnerDiff = averagePartnerUtility - prevAveragePartnerUtility;

		if (compromise) {
			double gap = Math.abs(averageUtility - averagePartnerUtility);
			if (partnerDiff < diff * 0.90
					|| gap > Math.pow(1.0 - time / 100, 2.0) * 0.90) {
				double p1 = maxCompromiseUtility, p2 = maxCompromiseUtility;
				for (int attenuation = 95; attenuation <= 100; attenuation++) {
					if (maxCompromiseUtility * attenuation
							/ 100 > maxUtilityOfPartnerBid) {
						p1 = maxCompromiseUtility * attenuation / 100;
						break;
					}
				}

				for (int attenuation = 10; attenuation < 1000; attenuation++) {
					if (maxCompromiseUtility
							- gap / attenuation > maxUtilityOfPartnerBid) {
						p2 = maxCompromiseUtility - gap / attenuation;
						break;
					}
				}

				maxCompromiseUtility = (p1 + p2) / 2;

				prevAverageUtility = averageUtility;
				compromise = false;
			}
		} else {
			if (partnerDiff > diff * 0.90
					|| (time > 50 && updateMaxPartnerUtility)) {
				prevAveragePartnerUtility = averagePartnerUtility;
				compromise = true;
			}
		}
		updateMaxPartnerUtility = false;
	}

	private BidType chooseBidType(Bid partnerBid) {
		BidType bidType = null;

		try {
			double time = timeline.getTime();

			double prevUtility = utilitySpace.getUtility(prevBid);
			double partnerUtility = utilitySpace.getUtility(partnerBid);
			double gap = Math.abs(prevUtility - partnerUtility);

			if (gap < 0.05)
				return BidType.APPROACH;

			gap = Math.abs(prevAverageUtility - prevAverageUtility);
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

			if (prevUtility > maxCompromiseUtility * 105 / 95) {
				return BidType.COMPROMISE;
			}
			if (continuousKeep > 20 && time > 0.60
					&& random100.nextDouble() > 0.60)
				return BidType.APPROACH;

		} catch (Exception e) {
			bidType = BidType.RESTORE;
		}

		double rnd = random100.nextDouble();
		double compromiseProbability = getCompromiseProbability();

		if (rnd < compromiseProbability) {
			bidType = BidType.COMPROMISE;
		} else if (rnd < getKeepProbability(compromiseProbability)
				+ compromiseProbability) {
			bidType = BidType.KEEP;
		}

		if (bidType == null)
			bidType = bidType.RESTORE;
		return bidType;
	}

	private double getCompromiseProbability() {
		double compromiseProbabilty = 0.0;

		try {
			double prevUtility = utilitySpace.getUtility(prevBid);
			double prevUtilityOfPartnerBid = utilitySpace
					.getUtility(prevPartnerBid);
			// Below Values are less and less, more and more compromised.
			double compromiseDegree = prevUtility / maxUtility;

			double compromisePartnerDegree = 1.0
					- prevUtilityOfPartnerBid / maxUtility;

			double continuous = ((double) continuousCompromiseBid
					+ (double) Math.abs(continuousPartnerCompromiseBid) + 0.001)
					/ 2.0;
			double ratioUtilityToMaxUtility = prevUtility / (maxUtility * 0.9);

			compromiseProbabilty = (compromiseDegree + compromisePartnerDegree)
					/ 2.0;

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
			double time = timeline.getTime();
			double prevUtility = utilitySpace.getUtility(prevBid);
			double prevUtilityOfPartnerBid = utilitySpace
					.getUtility(prevPartnerBid);
			double ratioUtilityToMaxUtility = prevUtility / (maxUtility * 0.8);

			if (prevUtility > prevUtilityOfPartnerBid) {
				keepProbability *= (prevUtility - prevUtilityOfPartnerBid)
						/ maxUtility;
				keepProbability *= time;
				keepProbability *= 1.0
						+ Math.abs(continuousCompromiseBid) / 4.0;
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
			double prevUtility = utilitySpace.getUtility(prevBid);
			double basicUtility = prevUtility;
			double gap = Math
					.abs(prevUtility - utilitySpace.getUtility(partnerBid));
			HashMap<Integer, Value> values = new HashMap<Integer, Value>();
			List<Issue> issues = utilitySpace.getDomain().getIssues();

			for (Issue issue : issues) {
				values.put(issue.getNumber(),
						prevBid.getValue(issue.getNumber()));
			}

			Integer changeIssueID = -1;
			for (Issue compromiseIssue : issues) {
				Integer compromiseIssueID = compromiseIssue.getNumber();

				if ((!prevPartnerBid.getValue(compromiseIssueID)
						.equals(partnerBid.getValue(compromiseIssueID))
						&& gap > 0.05 && continuousKeep < 20)
						|| prevBid.getValue(compromiseIssueID)
								.equals(partnerBid.getValue(compromiseIssueID))
						|| compromiseIssueID == changeIssueID) {
					continue;
				}

				HashMap<Integer, Value> candidateValues = new HashMap<Integer, Value>(
						values);
				candidateValues.remove(compromiseIssueID);
				candidateValues.put(compromiseIssueID,
						getCompromiseValue(compromiseIssue,
								prevBid.getValue(compromiseIssueID),
								partnerBid.getValue(compromiseIssueID)));
				Bid candidateBid = new Bid(utilitySpace.getDomain(),
						candidateValues);
				double candidateUtility = utilitySpace.getUtility(candidateBid);

				if (candidateUtility > compromiseUtility
						&& candidateUtility < basicUtility) {

					compromiseUtility = candidateUtility;
					compromiseBid = candidateBid;
					changeIssueID = compromiseIssueID;
				}
			}

		} catch (Exception e) {
			e.printStackTrace();
			compromiseBid = restoreBid; // best guess if things go wrong.
		}
		try {
		} catch (Exception e) {
			e.printStackTrace();
		}
		return compromiseBid;
	}

	private Value getCompromiseValue(Issue issue, Value val, Value partnerVal) {
		Value compromiseVal = null;
		Evaluator eval = ((AdditiveUtilitySpace) utilitySpace)
				.getEvaluator(issue.getNumber());

		switch (issue.getType()) {
		case DISCRETE:
			EvaluatorDiscrete evalDiscrete = (EvaluatorDiscrete) eval;
			compromiseVal = val;
			Integer evaluation = evalDiscrete.getValue((ValueDiscrete) val);
			IssueDiscrete issueDiscrete = (IssueDiscrete) issue;
			Integer compromiseEvaluation = 0;
			for (int i = 0; i < issueDiscrete.getNumberOfValues(); i++) {
				ValueDiscrete candidateVal = issueDiscrete.getValue(i);
				Integer candidateEvaluation = evalDiscrete
						.getValue(candidateVal);
				if (candidateEvaluation >= compromiseEvaluation
						&& candidateEvaluation < evaluation) {
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
			compromiseReal += new Random().nextDouble()
					* (partnerValReal.getValue() - compromiseReal) / 10;
			compromiseVal = new ValueReal(compromiseReal);
			break;
		default:
			compromiseVal = val;
			break;
		}
		return compromiseVal;
	}

	private Bid getBetweenBid(Bid basicBid, Bid partnerBid, double minUtility,
			double maxUtility) {
		Bid betweenBid = basicBid;
		try {
			List<Issue> issues = utilitySpace.getDomain().getIssues();
			HashMap<Integer, Value> values = new HashMap<Integer, Value>();

			for (Issue issue : issues) {
				values.put(issue.getNumber(),
						basicBid.getValue(issue.getNumber()));
			}

			double utility = utilitySpace.getUtility(basicBid);
			for (int i = 0; i < 1000; i++) {
				Issue issue = issues.get(random200.nextInt(issues.size()));
				int issueID = issue.getNumber();

				if (prevBid.getValue(issueID)
						.equals(partnerBid.getValue(issueID))) {
					continue;
				}

				Value value = values.get(issueID);
				values.remove(issueID);
				if (utility > maxUtility) {
					values.put(issueID, getLowerValue(issue, value, values));
				} else {
					values.put(issueID, getHigherValue(issue, value, values));
				}

				Bid candidateBid = new Bid(utilitySpace.getDomain(), values);
				utility = utilitySpace.getUtility(candidateBid);
				if (utility > minUtility && utility < maxUtility) {
					betweenBid = candidateBid;
					break;
				}
			}
		} catch (Exception e) {
			betweenBid = restoreBid;
		}
		return betweenBid;
	}

	private Value getHigherValue(Issue issue, Value value,
			HashMap<Integer, Value> values) {
		Value higher = null;
		Evaluator eval = ((AdditiveUtilitySpace) utilitySpace)
				.getEvaluator(issue.getNumber());

		switch (issue.getType()) {
		case DISCRETE:
			EvaluatorDiscrete evalDiscrete = (EvaluatorDiscrete) eval;

			Integer evaluation = evalDiscrete.getValue((ValueDiscrete) value);
			IssueDiscrete issueDiscrete = (IssueDiscrete) issue;
			higher = evalDiscrete.getMaxValue();
			Integer highEvaluation = evalDiscrete
					.getValue((ValueDiscrete) higher);
			for (int i = 0; i < issueDiscrete.getNumberOfValues(); i++) {
				ValueDiscrete candidateVal = issueDiscrete.getValue(i);
				Integer candidateEvaluation = evalDiscrete
						.getValue(candidateVal);
				if (candidateEvaluation < highEvaluation
						&& candidateEvaluation > evaluation) {
					higher = candidateVal;
					highEvaluation = candidateEvaluation;
				}
			}
			break;
		case INTEGER:
			IssueInteger issueInt = (IssueInteger) issue;
			ValueInteger valInt = (ValueInteger) value;
			try {
				Bid bid = new Bid(utilitySpace.getDomain(), values);
				values.remove(issueInt.getNumber());
				values.put(issueInt.getNumber(),
						new ValueInteger(valInt.getValue() - 1));
				Bid candidateBid = new Bid(utilitySpace.getDomain(), values);
				if (utilitySpace.getUtility(candidateBid) > utilitySpace
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
				Bid bid = new Bid(utilitySpace.getDomain(), values);
				values.remove(issueReal.getNumber());
				values.put(issueReal.getNumber(), new ValueReal(valReal
						.getValue()
						- (valReal.getValue() - issueReal.getLowerBound())
								/ 10));
				Bid candidateBid = new Bid(utilitySpace.getDomain(), values);
				if (utilitySpace.getUtility(candidateBid) > utilitySpace
						.getUtility(bid)) {
					higher = new ValueReal(
							valReal.getValue()
									- new Random().nextDouble()
											* (valReal.getValue()
													- issueReal.getLowerBound())
											/ 10);
				} else {
					higher = new ValueReal(valReal.getValue() + new Random()
							.nextDouble()
							* (issueReal.getUpperBound() - valReal.getValue())
							/ 10);
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

	private Value getLowerValue(Issue issue, Value value,
			HashMap<Integer, Value> values) {
		Value lower = null;
		Evaluator eval = ((AdditiveUtilitySpace) utilitySpace)
				.getEvaluator(issue.getNumber());

		switch (issue.getType()) {
		case DISCRETE:
			EvaluatorDiscrete evalDiscrete = (EvaluatorDiscrete) eval;

			Integer evaluation = evalDiscrete.getValue((ValueDiscrete) value);
			IssueDiscrete issueDiscrete = (IssueDiscrete) issue;
			lower = evalDiscrete.getMinValue();
			Integer lowEvaluation = evalDiscrete
					.getValue((ValueDiscrete) lower);
			for (int i = 0; i < issueDiscrete.getNumberOfValues(); i++) {
				ValueDiscrete candidateVal = issueDiscrete.getValue(i);
				Integer candidateEvaluation = evalDiscrete
						.getValue(candidateVal);
				if (candidateEvaluation > lowEvaluation
						&& candidateEvaluation < evaluation) {
					lower = candidateVal;
					lowEvaluation = candidateEvaluation;
				}
			}
			break;
		case INTEGER:
			IssueInteger issueInt = (IssueInteger) issue;
			ValueInteger valInt = (ValueInteger) value;
			try {
				Bid bid = new Bid(utilitySpace.getDomain(), values);
				values.remove(issueInt.getNumber());
				values.put(issueInt.getNumber(),
						new ValueInteger(valInt.getValue() - 1));
				Bid candidateBid = new Bid(utilitySpace.getDomain(), values);
				if (utilitySpace.getUtility(candidateBid) < utilitySpace
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
				Bid bid = new Bid(utilitySpace.getDomain(), values);
				values.remove(issueReal.getNumber());
				values.put(issueReal.getNumber(), new ValueReal(valReal
						.getValue()
						- (valReal.getValue() - issueReal.getLowerBound())
								/ 10));
				Bid candidateBid = new Bid(utilitySpace.getDomain(), values);
				if (utilitySpace.getUtility(candidateBid) < utilitySpace
						.getUtility(bid)) {
					lower = new ValueReal(
							valReal.getValue()
									- new Random().nextDouble()
											* (valReal.getValue()
													- issueReal.getLowerBound())
											/ 10);
				} else {
					lower = new ValueReal(valReal.getValue() + new Random()
							.nextDouble()
							* (issueReal.getUpperBound() - valReal.getValue())
							/ 10);
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
			List<Issue> issues = utilitySpace.getDomain().getIssues();

			for (Issue issue : issues) {
				values.put(issue.getNumber(),
						prevBid.getValue(issue.getNumber()));
			}
			approachValues = values;

			double candidateUtility = 0.0;
			for (Issue issue : issues) {
				int issueID = issue.getNumber();

				if (issue.getType() == ISSUETYPE.REAL) {
					IssueReal issueReal = (IssueReal) issue;
					ValueReal valueReal = (ValueReal) prevBid.getValue(issueID);
					ValueReal partnerValueReal = (ValueReal) partnerBid
							.getValue(issueID);
					ValueReal prevPartnerValueReal = (ValueReal) prevPartnerBid
							.getValue(issueID);
					double gap = Math.abs(
							valueReal.getValue() - partnerValueReal.getValue());
					double gapPartnerValue = Math
							.abs(partnerValueReal.getValue()
									- prevPartnerValueReal.getValue());
					if (gap < (issueReal.getUpperBound()
							- issueReal.getLowerBound()) / 100
							|| !(gapPartnerValue < (issueReal.getUpperBound()
									- issueReal.getLowerBound()) / 100)) {
						continue;
					}
				}

				if (prevBid.getValue(issueID)
						.equals(partnerBid.getValue(issueID))) {
					continue;
				}

				HashMap<Integer, Value> candidateValues = new HashMap<Integer, Value>(
						values);

				candidateValues.remove(issueID);
				candidateValues.put(issueID,
						getApproachValue(issue, prevBid.getValue(issueID),
								partnerBid.getValue(issueID)));
				Bid candidateBid = new Bid(utilitySpace.getDomain(),
						candidateValues);
				if (utilitySpace.getUtility(candidateBid) > candidateUtility) {
					candidateUtility = utilitySpace.getUtility(candidateBid);
					approachBid = candidateBid;
					approachValues = candidateValues;
				}
			}

			while (true) {
				candidateUtility = 0.0;
				HashMap<Integer, Value> candidateValues = new HashMap<Integer, Value>(
						approachValues);
				for (Issue issue : issues) {
					HashMap<Integer, Value> tempValues = new HashMap<Integer, Value>(
							approachValues);

					int issueID = issue.getNumber();

					if (issue.getType() == ISSUETYPE.REAL) {
						IssueReal issueReal = (IssueReal) issue;
						ValueReal valueReal = (ValueReal) prevBid
								.getValue(issueID);
						ValueReal partnerValueReal = (ValueReal) partnerBid
								.getValue(issueID);
						double gap = Math.abs(valueReal.getValue()
								- partnerValueReal.getValue());
						if (gap < (issueReal.getUpperBound()
								- issueReal.getLowerBound()) / 100) {
							continue;
						}
					}

					if (prevBid.getValue(issueID)
							.equals(partnerBid.getValue(issueID))) {
						continue;
					}

					tempValues.remove(issueID);
					tempValues.put(issueID,
							getApproachValue(issue, prevBid.getValue(issueID),
									partnerBid.getValue(issueID)));

					Bid tempBid = new Bid(utilitySpace.getDomain(), tempValues);
					if (utilitySpace.getUtility(tempBid) > candidateUtility) {
						candidateValues = tempValues;
						candidateUtility = utilitySpace.getUtility(tempBid);
					}
				}
				if (candidateUtility >= utilitySpace.getUtility(approachBid)
						&& !candidateValues.equals(approachValues)) {
					Bid candidateBid = new Bid(utilitySpace.getDomain(),
							candidateValues);
					approachBid = candidateBid;
					approachValues = candidateValues;
				} else {
					break;
				}
			}
		} catch (Exception e) {
			approachBid = restoreBid; // best guess if things go wrong.
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
			approachReal += new Random().nextDouble()
					* (partnerValReal.getValue() - approachReal) / 10;
			approachVal = new ValueReal(approachReal);
			break;
		default:
			approachVal = myVal;
			break;
		}
		return approachVal;
	}

	private void updateRestoreBid(Bid nextBid) {
		List<Issue> issues = utilitySpace.getDomain().getIssues();

		try {
		} catch (Exception e) {
			e.printStackTrace();
		}
		double evalGap = 0.0;
		for (Issue issue : issues) {
			int issueID = issue.getNumber();
			Evaluator eval = ((AdditiveUtilitySpace) utilitySpace)
					.getEvaluator(issueID);

			try {
				evalGap += Math.abs(eval.getEvaluation(
						((AdditiveUtilitySpace) utilitySpace), nextBid, issueID)
						- eval.getEvaluation(
								((AdditiveUtilitySpace) utilitySpace),
								maxUtilityPartnerBid, issueID));

			} catch (Exception e) {
				evalGap += 1.0;
			}
		}
		if (evalGap < minGap) {
			restoreBid = nextBid;
			minGap = evalGap;
		} else if (evalGap == minGap) {
			try {
				if (utilitySpace.getUtility(nextBid) > utilitySpace
						.getUtility(restoreBid)) {
					restoreBid = nextBid;
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	private boolean isAccept(Bid partnerBid) {
		boolean accept = false;
		try {
			double prevUtility = utilitySpace.getUtility(prevBid);
			double offeredUtil = utilitySpace.getUtility(partnerBid);
			double time = timeline.getTime();

			if (offeredUtil > maxUtilityOfPartnerBid) {
				maxUtilityPartnerBid = partnerBid;
				maxUtilityOfPartnerBid = offeredUtil;
				updateMaxPartnerUtility = true;
			}

			if (offeredUtil > maxUtility * 0.95 || offeredUtil >= prevUtility) {
				accept = true;
			} else {
				List<Issue> issues = utilitySpace.getDomain().getIssues();

				double evalGap = 0.0;
				for (Issue issue : issues) {

					int issueID = issue.getNumber();

					switch (issue.getType()) {
					case REAL:
						IssueReal issueReal = (IssueReal) issue;
						ValueReal valueReal = (ValueReal) prevBid
								.getValue(issueID);
						ValueReal partnerValueReal = (ValueReal) partnerBid
								.getValue(issueID);
						double gap = Math.abs(valueReal.getValue()
								- partnerValueReal.getValue());
						if (gap > (issueReal.getUpperBound()
								- issueReal.getLowerBound()) / 100) {
							evalGap += 0.5;
						}
						break;
					default:
						if (!prevBid.getValue(issueID)
								.equals(partnerBid.getValue(issueID))) {
							evalGap += 0.5;
						}
						break;
					}

					Evaluator eval = ((AdditiveUtilitySpace) utilitySpace)
							.getEvaluator(issueID);

					try {
						evalGap += Math.abs(eval.getEvaluation(
								((AdditiveUtilitySpace) utilitySpace), prevBid,
								issueID)
								- eval.getEvaluation(
										((AdditiveUtilitySpace) utilitySpace),
										partnerBid, issueID));
					} catch (Exception e) {
						evalGap += 1.0;
					}

					evalGap += 0.5;
				}

				evalGap /= 2.0 * issues.size();

				if (time < 0.50) {
					if (prevPartnerBid != null
							&& offeredUtil > maxCompromiseUtility
							&& offeredUtil >= maxUtilityOfPartnerBid) {
						double acceptCoefficient = -0.1 * time + 1.0;

						if (evalGap < 0.30 && offeredUtil > prevUtility
								* acceptCoefficient) {
							accept = true;
						}
					}
				} else if (time < 0.80) {
					if (prevPartnerBid != null
							&& offeredUtil > maxCompromiseUtility * 0.95
							&& offeredUtil > maxUtilityOfPartnerBid * 0.95) {
						double diffMyBidAndOffered = prevUtility - offeredUtil;

						if (diffMyBidAndOffered < 0.0)
							diffMyBidAndOffered = 0.0;

						double acceptCoefficient = -0.16 * (time - 0.50) + 0.95;

						if (evalGap < 0.35 && offeredUtil > prevUtility
								* acceptCoefficient) {
							accept = true;
						}
					}
				} else {

					if (prevPartnerBid != null
							&& offeredUtil > maxCompromiseUtility * 0.90
							&& offeredUtil > maxUtilityOfPartnerBid * 0.95) {

						double restoreEvalGap = 0.0;
						for (Issue issue : issues) {
							int issueID = issue.getNumber();

							switch (issue.getType()) {
							case REAL:
								IssueReal issueReal = (IssueReal) issue;
								ValueReal valueReal = (ValueReal) restoreBid
										.getValue(issueID);
								ValueReal partnerValueReal = (ValueReal) partnerBid
										.getValue(issueID);
								double gap = Math.abs(valueReal.getValue()
										- partnerValueReal.getValue());
								if (gap > (issueReal.getUpperBound()
										- issueReal.getLowerBound()) / 100) {
									restoreEvalGap += 0.5;
								}
								break;
							default:
								if (!restoreBid.getValue(issueID)
										.equals(partnerBid.getValue(issueID))) {
									restoreEvalGap += 0.5;
								}
								break;
							}

							Evaluator eval = ((AdditiveUtilitySpace) utilitySpace)
									.getEvaluator(issueID);

							try {
								restoreEvalGap += Math.abs(eval.getEvaluation(
										((AdditiveUtilitySpace) utilitySpace),
										prevBid, issueID)
										- eval.getEvaluation(
												((AdditiveUtilitySpace) utilitySpace),
												partnerBid, issueID));
							} catch (Exception e) {
								restoreEvalGap += 1.0;
							}

							restoreEvalGap += 0.5;
						}

						restoreEvalGap /= 2.0 * issues.size();

						double threshold = 0.40;

						if (time > (0.90)) {
							threshold = 0.50;

						}
						if (restoreEvalGap <= threshold
								|| evalGap <= threshold) {
							accept = true;
						}
					}
				}
			}
		} catch (Exception e) {
			accept = false;
		}
		return accept;
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
		return (u - 2. * u * t
				+ 2. * (-1. + t + Math.sqrt(sq(-1. + t) + u * (-1. + 2 * t))))
				/ (-1. + 2 * t);
	}

	double sq(double x) {
		return x * x;
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