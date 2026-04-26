package agents.anac.y2012.MetaAgent.agents.ShAgent;

//package agents;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
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
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AdditiveUtilitySpace;

public class ShAgent extends Agent {

	private Action actionOfPartner = null;
	private Bid partnerFirstBid = null;
	private Bid partnerBestBid = null;
	private double threshold;

	private Bid maxUtilityBid = null;

	private OpponentModel opponentModel;
	private UtilityAnalyzer utilAnalyzer;

	/**
	 * init is called when a next session starts with the same opponent.
	 */
	public void init() {
		opponentModel = new OpponentModel((AdditiveUtilitySpace) utilitySpace, timeline);

		try {
			maxUtilityBid = utilitySpace.getMaxUtilityBid();
		} catch (Exception e) {
		}

		utilAnalyzer = new UtilityAnalyzer((AdditiveUtilitySpace) utilitySpace, maxUtilityBid);
	}

	@Override
	public String getVersion() {
		return "1.1.1";
	}

	@Override
	public String getName() {
		return "ShAgent";
	}

	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
	}

	public Action chooseAction() {
		Action action = null;
		Bid partnerBid = null;
		try {
			if (actionOfPartner == null) {
				action = new Offer(getAgentID(), maxUtilityBid);
			} else if (actionOfPartner instanceof Offer) {
				partnerBid = ((Offer) actionOfPartner).getBid();

				opponentModel.receiveNewOpponentBid(partnerBid);

				if (partnerFirstBid == null) {
					partnerFirstBid = partnerBid;
				}

				this.setBestBid(partnerBid);

				this.threshold = calculateThreshold();

				// System.out.println("* " + getName() + ": Time: + " +
				// this.timeline.getTime() + ", New threshold: " +
				// this.threshold);

				// accept under certain circumstances
				if (isAcceptable(partnerBid)) {
					action = new Accept(getAgentID(), partnerBid);
				}
				// else if (opponentModel.getNumberOfRemainingTurns() <= 2 ||
				// this.threshold <= this.partnerBestBidUtility)
				// {
				// //if there is no time left, or the threshold is now lower
				// than his best bid so far
				// Bid ourOffer = this.partnerBestBid;
				// action = new Offer(this.getAgentID(), ourOffer);
				// }
				else {
					Bid ourOffer = utilAnalyzer.GetClosestOffer(partnerBid, threshold);
					action = new Offer(this.getAgentID(), ourOffer);
				}
			}
		} catch (Exception e) {
			// System.out.println("Exception in ChooseAction:"+e.getMessage());
			action = new Accept(getAgentID(), partnerBid); // best guess if
															// things go wrong.
		}
		return action;
	}

	/**
	 * Save the best bid (max utility for us) that the partner had suggested.
	 * 
	 * @param bid
	 */
	private void setBestBid(Bid bid) {
		if (this.partnerBestBid == null) {
			this.partnerBestBid = bid;
		} else {
			double newUtil = this.getUtilityWithoutDiscount(bid);
			double oldUtil = this.getUtilityWithoutDiscount(this.partnerBestBid);

			if (newUtil > oldUtil) {
				this.partnerBestBid = bid;
			}
		}

	}

	/**
	 * Returns true if we should accept the offer.
	 * 
	 * @param partnerBid
	 * @return
	 */
	private boolean isAcceptable(Bid partnerBid) {
		return this.getUtility(partnerBid) >= this.threshold;
	}

	/**
	 * Calculate threshold, all threshold suggested must exceed this threshold.
	 * We also accept all offers above this threshold.
	 * 
	 * @return The threshold.
	 * @throws Exception
	 */
	private double calculateThreshold() throws Exception {
		// top - The offer with max (top) utility.
		double top = this.getUtility(this.maxUtilityBid);

		// bottom - The offer with max utility for the partner (we assume that
		// this is the first bid he suggests).
		// This offer is the min (bottom) utility.
		double bottom = this.getUtility(this.partnerFirstBid);

		// l - The range between top & bottom.
		double l = top - bottom;

		// timeFactor - defines where we are on the range l. The thumb rule is
		// that we start close to top and slowly
		// diverge to bottom.
		double timeFactor = getTimeFactor();

		// hardThreshold - The threshold which is somewhere between top and
		// bottom.
		double hardThreshold = bottom + (l * timeFactor);

		// We return the hardThreshold after applying the discount factor on it.
		return this.getUtilityWithDiscount(hardThreshold);
	}

	/**
	 * Return a factor between 0.0 to 1.0 which defines how our threshold will
	 * be influenced by time & discount.
	 * 
	 * @return
	 */
	protected double getTimeFactor() {
		double t = this.timeline.getTime();
		double d = this.utilitySpace.getDiscountFactor();
		d = ((d <= 0.0) || (d > 1.0)) ? 1.0 : d;

		// This is the time factor formula: ((1-t)^(t/4)) - ((1-d)*t)
		return Math.pow((1 - t), (t / 4)) - ((1 - d) * t);
	}

	/**
	 * Returns the utility of the bid without the discount factor.
	 * 
	 * @param bid
	 * @return
	 */
	private double getUtilityWithoutDiscount(Bid bid) {
		double d = this.utilitySpace.getDiscountFactor();
		d = ((d <= 0.0) || (d > 1.0)) ? 1.0 : d;
		return this.getUtility(bid) / Math.pow(d, this.timeline.getTime());
	}

	/**
	 * Applying the discount factor on a utility value.
	 * 
	 * @param util
	 * @return
	 */
	private double getUtilityWithDiscount(double util) {
		double t = this.timeline.getTime();
		double d = this.utilitySpace.getDiscountFactor();
		d = ((d <= 0.0) || (d > 1.0)) ? 1.0 : d;
		return util * Math.pow(d, t);
	}

	private class UtilityAnalyzer {
		private AdditiveUtilitySpace utilitySpace;
		private List<Issue> allIssues;
		private HashMap<Issue, ArrayList<Value>> preferredValuesPerIssue = new HashMap<Issue, ArrayList<Value>>();
		private HashMap<Issue, HashMap<Value, Double>> utilityPerValuePerIssue = new HashMap<Issue, HashMap<Value, Double>>();
		private Bid bid;

		public UtilityAnalyzer(AdditiveUtilitySpace space, Bid maxUtilityBid) {
			this.utilitySpace = space;
			allIssues = utilitySpace.getDomain().getIssues();
			bid = Clone(maxUtilityBid);
			try {
				BuildValuePreferences();
			} catch (Exception e) {
			}
		}

		private void BuildValuePreferences() throws Exception {

			for (final Issue issue : allIssues) {

				utilityPerValuePerIssue.put(issue, new HashMap<Value, Double>());

				int numOptions;
				ArrayList<Value> values = new ArrayList<Value>();
				switch (issue.getType()) {
				case DISCRETE:
					IssueDiscrete issueDiscrete = (IssueDiscrete) issue;
					numOptions = issueDiscrete.getNumberOfValues();
					for (int i = 0; i < numOptions; i++) {
						Value value = issueDiscrete.getValue(i);
						values.add(value);
						updateUtilityForSpecificValue(issue, value);
					}
					break;
				case REAL:
					IssueReal issueReal = (IssueReal) issue;
					numOptions = issueReal.getNumberOfDiscretizationSteps() - 1;

					double step = (issueReal.getUpperBound() - issueReal.getLowerBound())
							/ (double) (issueReal.getNumberOfDiscretizationSteps());
					for (double dVal = issueReal.getLowerBound(); dVal <= issueReal.getUpperBound(); dVal += step) {
						Value value = new ValueReal(dVal);
						values.add(value);
						updateUtilityForSpecificValue(issue, value);
					}
					break;
				case INTEGER:
					IssueInteger issueInteger = (IssueInteger) issue;

					// the values are inclusive of lower and upper bound
					for (int i = issueInteger.getLowerBound(); i <= issueInteger.getUpperBound(); i++) {
						Value value = new ValueInteger(i);
						values.add(value);
						updateUtilityForSpecificValue(issue, value);
					}
					break;
				}

				Collections.sort(values, new Comparator<Value>() {

					public int compare(Value o1, Value o2) {
						return Double.compare(utilityPerValuePerIssue.get(issue).get(o1),
								utilityPerValuePerIssue.get(issue).get(o2));
					}
				});

				// finished sorting the values for this issue

				preferredValuesPerIssue.put(issue, values);
			}
		}

		/**
		 * Update the map of utility per value per issue
		 * 
		 * @param issue
		 * @param value
		 * @throws Exception
		 */
		private void updateUtilityForSpecificValue(final Issue issue, Value value) throws Exception {
			bid = bid.putValue(issue.getNumber(), value);
			double util = utilitySpace.getWeight(issue.getNumber())
					* utilitySpace.getEvaluation(issue.getNumber(), bid);
			utilityPerValuePerIssue.get(issue).put(value, util);
		}

		private class OfferStep implements Comparable<OfferStep> {
			public double UtilOffset;
			public Issue Issue;
			public Value Value;

			public OfferStep(double utilOffset, Issue issue, Value value) {
				this.UtilOffset = utilOffset;
				this.Issue = issue;
				this.Value = value;
			}

			public int compareTo(OfferStep other) {
				return Double.compare(this.UtilOffset, other.UtilOffset);
			}
		}

		private Bid Clone(Bid source) {
			Bid clone = null;

			try {
				HashMap<Integer, Value> map = new HashMap<Integer, Value>();
				for (Issue issue : allIssues)
					map.put(issue.getNumber(), source.getValue(issue.getNumber()));

				clone = new Bid(this.utilitySpace.getDomain(), map);

			} catch (Exception e) {
				// e.printStackTrace();
			}

			return clone;
		}

		public Bid StepCloser(Bid lastOffer) throws Exception {
			Bid offer = this.Clone(lastOffer);

			int issueCount = allIssues.size();
			List<OfferStep> steps = new ArrayList<OfferStep>(issueCount);

			// FILL IN utilOffset
			for (Issue issue : allIssues) {
				OfferStep step = GetOfferStepInIssue(lastOffer, issue);
				steps.add(step);
			}

			// SORT
			Collections.sort(steps, Collections.reverseOrder());

			// random select
			double totalOffset = 0;
			for (int i = 0; i < issueCount; i++)
				totalOffset += steps.get(i).UtilOffset;

			// Select random best util offset
			Random random = new Random();
			double randomValue = random.nextDouble() * totalOffset;

			OfferStep bestStep = steps.get(0);
			for (int i = 0; i < issueCount; i++) {
				randomValue -= steps.get(i).UtilOffset;
				if (randomValue < 0) {
					bestStep = steps.get(i);
					break;
				}
			}

			offer = offer.putValue(bestStep.Issue.getNumber(), bestStep.Value);

			return offer;
		}

		private OfferStep GetOfferStepInIssue(Bid lastOffer, Issue issue) throws Exception {
			int i = issue.getNumber();

			ArrayList<Value> issueValues = this.preferredValuesPerIssue.get(issue);

			Value oldValue;
			int oldIndex;

			int newIndex;
			Value newValue;

			double oldUtil = 0;
			double newUtil = 0;

			switch (issue.getType()) {
			case INTEGER:
			case DISCRETE:
				oldValue = lastOffer.getValue(i);
				oldIndex = issueValues.indexOf(oldValue);

				newIndex = Math.min(issueValues.size() - 1, oldIndex + 1);
				newValue = issueValues.get(newIndex);

				oldUtil = this.utilityPerValuePerIssue.get(issue).get(oldValue);
				newUtil = this.utilityPerValuePerIssue.get(issue).get(newValue);
				break;
			case REAL:
				IssueReal issueReal = (IssueReal) issue;

				oldValue = lastOffer.getValue(i);
				oldIndex = issueValues.indexOf(oldValue);

				bid = bid.putValue(issue.getNumber(), oldValue);
				oldUtil = this.utilitySpace.getWeight(issue.getNumber())
						* this.utilitySpace.getEvaluation(issue.getNumber(), bid);

				double dOldValue = ((ValueReal) oldValue).getValue();
				double step = (issueReal.getUpperBound() - issueReal.getLowerBound())
						/ issueReal.getNumberOfDiscretizationSteps();
				newValue = new ValueReal(Math.min(dOldValue + step, issueReal.getUpperBound()));

				bid = bid.putValue(issue.getNumber(), newValue);
				newUtil = this.utilitySpace.getWeight(issue.getNumber())
						* this.utilitySpace.getEvaluation(issue.getNumber(), bid);

				Value newValueDown = new ValueReal(Math.max(dOldValue - step, issueReal.getLowerBound()));
				bid = bid.putValue(issue.getNumber(), newValueDown);
				double newUtilDown = this.utilitySpace.getWeight(issue.getNumber())
						* this.utilitySpace.getEvaluation(issue.getNumber(), bid);

				if (newUtilDown > newUtil) {
					newUtil = newUtilDown;
					newValue = newValueDown;
				}

				break;

			default:
				throw new Exception("Unknown issue type");

			}

			double utilOffset = newUtil - oldUtil;

			OfferStep step = new OfferStep(utilOffset, issue, newValue);
			return step;
		}

		public Bid GetClosestOffer(Bid lastBid, double threshold) throws Exception {
			Bid offer = null;
			Bid lastOffer = this.Clone(lastBid);

			do {
				offer = StepCloser(lastOffer);
				lastOffer = offer;
			} while (this.utilitySpace.getUtility(offer) < threshold);

			return offer;
		}
	}

	/**
	 * This class is responsible for maintaining the model of the opponent and
	 * for providing API functions to easily select bids offered to the opponent
	 * 
	 * @author Ron
	 * 
	 */
	private class OpponentModel {

		private List<Issue> allIssues;

		/**
		 * For each issue, a list of values in the order that they were offered
		 * by the opponent It is assumed that the opponent offers the values in
		 * the order of his preference
		 */
		private HashMap<Issue, ArrayList<Value>> opponentPrefferedValuePerIssue = new HashMap<Issue, ArrayList<Value>>();
		private TimeLineInfo timeline;

		private double[] previousTurnTimesArr = new double[100];
		private int turnNumber = 0;
		private double lastTurnStartTime = 0;

		public OpponentModel(AdditiveUtilitySpace utilitySpace, TimeLineInfo timeline) {
			this.timeline = timeline;

			allIssues = utilitySpace.getDomain().getIssues();

			for (Issue issue : allIssues) {
				opponentPrefferedValuePerIssue.put(issue, new ArrayList<Value>());
			}
		}

		/**
		 * receiveMessage this class with the last bid from the opponent
		 * 
		 * @param bid
		 */
		public void receiveNewOpponentBid(Bid bid) {

			manageTiming();

			for (Issue issue : allIssues) {
				try {
					Value value = bid.getValue(issue.getNumber());
					if (opponentPrefferedValuePerIssue.get(issue).contains(value)) {
						// the opponent has offered this value in the past
					} else {
						opponentPrefferedValuePerIssue.get(issue).add(value);
					}
				} catch (Exception e) {
					// do nothing
				}
			}
		}

		/**
		 * Update timing variables of the model
		 */
		private void manageTiming() {
			double currTime = this.timeline.getTime(); // between 0 and 1

			previousTurnTimesArr[turnNumber % previousTurnTimesArr.length] = currTime - lastTurnStartTime;

			turnNumber++;

			lastTurnStartTime = currTime;
		}
	}
}