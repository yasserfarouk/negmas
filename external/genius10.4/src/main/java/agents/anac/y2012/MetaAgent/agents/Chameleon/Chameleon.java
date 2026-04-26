package agents.anac.y2012.MetaAgent.agents.Chameleon;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
import genius.core.utility.AdditiveUtilitySpace;

/*
 * Agent Chameleon. Changes its behavior by adapting to the opponent.
 */
public class Chameleon extends Agent {
	/*
	 * A strategy is almost the same thing as an agent.
	 */
	public static interface Strategy {

		/*
		 * Initialize the given strategy
		 */
		public void init(AdditiveUtilitySpace utilitySpace);

		/*
		 * Receive an opponent bid
		 */
		public void receiveOpponentBid(Bid bid, double receiveTime);

		/*
		 * Respond to the last opponent bid
		 */
		public Bid respondToBid(double currentTime);
	}

	/*
	 * Base class for strategies of the Chameleon
	 */
	public static abstract class BaseStrategy implements Strategy {
		private static final int MAX_RANDOM_BID_SEARCH_TRIES = 10000;
		private Bid maxUtilityBid;
		private double maxUtility;
		private AdditiveUtilitySpace utilitySpace;
		private Map<Bid, Double> opponentBids;
		private Map<Bid, Double> opponentBidTimes;
		private Bid lastOpponentBid;
		private double lastOpponentBidTime;
		private Random random;

		/*
		 * Initialize the given strategy
		 */
		public void init(AdditiveUtilitySpace utilitySpace) {
			this.utilitySpace = utilitySpace;

			try {
				this.maxUtilityBid = utilitySpace.getMaxUtilityBid();
				this.maxUtility = utilitySpace.getUtility(maxUtilityBid);
			} catch (Exception ex) {
				ex.printStackTrace();
			}
			opponentBids = new HashMap<Bid, Double>();
			opponentBidTimes = new HashMap<Bid, Double>();
			random = new Random();
		}

		/*
		 * Get the maximum utility bid
		 */
		protected Bid getMaxUtilityBid() {
			return maxUtilityBid;
		}

		/*
		 * Get the maximum utility
		 */
		protected double getMaxUtility() {
			return maxUtility;
		}

		/*
		 * Get the utility space
		 */
		protected AdditiveUtilitySpace getUtilitySpace() {
			return utilitySpace;
		}

		/*
		 * Searches a random bid with minimal utility.
		 */
		protected Bid searchRandomBidWithMinimalUtility(double minimalUtility) {
			HashMap<Integer, Value> values = new HashMap<Integer, Value>();
			List<Issue> issues = this.utilitySpace.getDomain().getIssues();
			int numTries = 0;

			Bid bid = null;

			int combinations = 1;

			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {
				case DISCRETE:
					combinations *= ((IssueDiscrete) lIssue)
							.getNumberOfValues();
					break;
				case REAL:
					combinations *= ((IssueReal) lIssue)
							.getNumberOfDiscretizationSteps();
					break;
				case INTEGER:
					combinations *= (((IssueInteger) lIssue).getUpperBound() - ((IssueInteger) lIssue)
							.getLowerBound());
					break;
				}
			}
			/*
			 * System.out.println("Started searching for random bid with utility = "
			 * + minimalUtility + "(issues = " + issues.size() +
			 * ", combinations = " + combinations + ")");
			 */

			try {
				do {
					for (Issue lIssue : issues) {
						switch (lIssue.getType()) {
						case DISCRETE:
							IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
							int optionIndex = random.nextInt(lIssueDiscrete
									.getNumberOfValues());
							values.put(Integer.valueOf(lIssue.getNumber()),
									lIssueDiscrete.getValue(optionIndex));
							break;

						case REAL:
							IssueReal lIssueReal = (IssueReal) lIssue;
							int optionInd = random.nextInt(lIssueReal
									.getNumberOfDiscretizationSteps() - 1);
							values.put(
									Integer.valueOf(lIssueReal.getNumber()),
									new ValueReal(
											lIssueReal.getLowerBound()
													+ (lIssueReal
															.getUpperBound() - lIssueReal
															.getLowerBound())
													* optionInd
													/ lIssueReal
															.getNumberOfDiscretizationSteps()));
							break;

						case INTEGER:
							IssueInteger lIssueInteger = (IssueInteger) lIssue;
							int optionIndex2 = lIssueInteger.getLowerBound()
									+ random.nextInt(lIssueInteger
											.getUpperBound()
											- lIssueInteger.getLowerBound());
							values.put(
									Integer.valueOf(lIssueInteger.getNumber()),
									new ValueInteger(optionIndex2));
							break;

						default:
							throw new Exception("issue type "
									+ lIssue.getType() + " not supported");
						}
					}

					bid = new Bid(this.utilitySpace.getDomain(), values);
					numTries++;
				} while (this.utilitySpace.getUtility(bid) < minimalUtility
						&& numTries <= MAX_RANDOM_BID_SEARCH_TRIES);

				if (this.utilitySpace.getUtility(bid) < minimalUtility)
					return getMaxUtilityBid();
			} catch (Exception ex) {
				ex.printStackTrace();
			}

			/*
			 * System.out.println(
			 * "Finished searching for random bid with utility = " +
			 * minimalUtility);
			 */

			return bid;
		}

		/*
		 * Get the last opponent bid
		 */
		protected Bid getLastOpponentBid() {
			return lastOpponentBid;
		}

		/*
		 * Get the time when the opponent has issued the last bid
		 */
		protected double getLastOpponentBidTime() {
			return lastOpponentBidTime;
		}

		/*
		 * Receive the opponent's bid, compute its utility, and store it in a
		 * map.
		 */
		public void receiveOpponentBid(Bid bid, double receiveTime) {
			try {
				lastOpponentBid = bid;
				lastOpponentBidTime = receiveTime;
				opponentBids.put(bid, utilitySpace.getUtility(bid));
				opponentBidTimes.put(bid, receiveTime);
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		}

		/*
		 * Returns the previous opponent bids.
		 */
		protected Map<Bid, Double> getOpponentBids() {
			return opponentBids;
		}

		/*
		 * Returns the private number generator.
		 */
		protected Random getRandom() {
			return random;
		}
	}

	/*
	 * Mirroring strategy.
	 */
	public class MirrorStrategy extends BaseStrategy {
		private Bid secondToLastBid;
		private double secondToLastBidTime;
		private double targetUtility;

		/*
		 * Initialize the strategy
		 */
		public void init(AdditiveUtilitySpace utilitySpace) {
			super.init(utilitySpace);
			targetUtility = 1.0;
		}

		/*
		 * Store the second to last bid
		 */
		public void receiveOpponentBid(Bid bid, double receiveTime) {
			secondToLastBid = getLastOpponentBid();
			secondToLastBidTime = timeline.getTime();
			super.receiveOpponentBid(bid, receiveTime);
		}

		/*
		 * Respond to a bid
		 */
		public Bid respondToBid(double currentTime) {
			if (secondToLastBid == null || getLastOpponentBid() == null)
				return getMaxUtilityBid();

			try {
				double lastBidUtilityNow = getUtilitySpace()
						.getUtilityWithDiscount(getLastOpponentBid(),
								currentTime);
				double lastBidUtilityThen = getUtilitySpace()
						.getUtilityWithDiscount(getLastOpponentBid(),
								getLastOpponentBidTime());
				double secondToLastBidUtilityThen = getUtilitySpace()
						.getUtilityWithDiscount(secondToLastBid,
								secondToLastBidTime);
				double lastChameleonBidUtilityThen = getUtilitySpace()
						.getUtilityWithDiscount(lastChameleonBid,
								lastChameleonBidTime);
				double currentDiscount = getDiscount(currentTime);
				double willingToAcceptUtilityNotDiscounted = lastChameleonBidUtilityThen
						+ lastBidUtilityThen - secondToLastBidUtilityThen;

				targetUtility = Math.max(willingToAcceptUtilityNotDiscounted,
						lastBidUtilityNow / currentDiscount);
				targetUtility = Math.max(targetUtility, 1 - currentTime * 0.3);
				targetUtility = Math.min(targetUtility, 1);

				// Accept the last bid if it matches the target utility we
				// defined
				if (lastBidUtilityNow >= targetUtility)
					return null;

				return searchRandomBidWithMinimalUtility(targetUtility);
			} catch (Exception ex) {
				ex.printStackTrace();
				return null;
			}
		}

	}

	/*
	 * Implements a stubborn strategy.
	 */
	public static class StubbornStrategy extends BaseStrategy {
		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * ro.pub.cs.anac.strategies.Strategy#respondToBid(negotiator.actions
		 * .Action)
		 */
		public Bid respondToBid(double currentTime) {
			if (getLastOpponentBid() == null) {
				return getMaxUtilityBid();
			} else {
				double desiredUtility = 0.0;

				try {
					double lastOpponentUtility = getUtilitySpace().getUtility(
							getLastOpponentBid());

					if (currentTime >= 0 && currentTime < 0.333)
						return getMaxUtilityBid();

					// Determine the desired utility as a function of time
					if (currentTime >= 0.333 && currentTime < 0.666)
						desiredUtility = getMaxUtility() * 0.9;
					else if (currentTime >= 0.666 && currentTime <= 0.95)
						desiredUtility = (currentTime - 0.666) / 0.334 * 0.1
								* getMaxUtility() + 0.8 * getMaxUtility();
					else
						desiredUtility = 0.7 * currentTime + 0.3;

					// Accept the opponent offer if we don't afford to be
					// ambitious (time is becoming critical)
					if (lastOpponentUtility >= desiredUtility
							&& currentTime >= 0.666)
						return null;

					// If opponent offers something better than we ever thought
					// of, accept
					if (lastOpponentUtility >= 1.05 * desiredUtility)
						return null;

					// Make a concession to 80% of desired utility if time is
					// almost up
					if (currentTime >= 0.95
							&& lastOpponentUtility >= 0.8 * desiredUtility)
						return null;

					return searchRandomBidWithMinimalUtility(desiredUtility);
				} catch (Exception ex) {
					ex.printStackTrace();
					return searchRandomBidWithMinimalUtility(desiredUtility);
				}
			}
		}

	}

	/*
	 * Piggy back strategy.
	 */
	public static class PiggyBackStrategy extends BaseStrategy {
		private static final int REASONABLE_NUMBER_OF_ISSUES = 15;
		private static final int REASONABLE_NUMBER_OF_TRIALS = 10000;

		private Bid bestNeighbour;
		private double bestNeighbourUtility;

		/*
		 * (non-Javadoc)
		 * 
		 * @see ro.pub.cs.anac.strategies.BaseStrategy#init(negotiator.utility.
		 * UtilitySpace)
		 */
		public void init(AdditiveUtilitySpace utilitySpace) {
			super.init(utilitySpace);
			bestNeighbourUtility = 0.0;
			bestNeighbour = null;
		}

		/*
		 * Searches for a better bid for our agent in the neighborhood of
		 * another bid. Searches all possibilities, given that there is a
		 * reasonable number of issues.
		 */
		private Bid searchBidInNeighbourhoodOf(Bid bid) {
			boolean tooManyIssues, ready;
			List<Issue> issues = this.getUtilitySpace().getDomain().getIssues();
			HashMap<Integer, Value> values = new HashMap<Integer, Value>();
			int[] modifications = new int[issues.size()];
			double maxUtility, utility;
			Bid maxUtilityBid = null, currentBid;
			int sum = 0, i, j, numTrials = 0;

			tooManyIssues = (getUtilitySpace().getDomain().getIssues().size() > REASONABLE_NUMBER_OF_ISSUES);
			for (i = 0; i < issues.size(); i++)
				modifications[i] = 0;

			try {
				maxUtilityBid = bid;
				maxUtility = getUtilitySpace().getUtility(bid);

				do {
					// Re-initialize values for current search
					values = new HashMap<Integer, Value>();

					for (j = 0; j < issues.size(); j++) {
						Issue lIssue = issues.get(j);
						int no = lIssue.getNumber();
						int optionIndex = 0;
						int bidValue;

						switch (lIssue.getType()) {
						case DISCRETE:
							IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
							bidValue = lIssueDiscrete
									.getValueIndex((ValueDiscrete) bid
											.getValue(no));
							switch (modifications[j]) {
							case 0:
								optionIndex = bidValue;
								break;
							case 1:
								if (bidValue > 0)
									optionIndex = bidValue - 1;
								else
									optionIndex = bidValue;
								break;
							case 2:
								if (bidValue + 1 < lIssueDiscrete.getValues()
										.size())
									optionIndex = bidValue + 1;
								else
									optionIndex = bidValue;
								break;
							}
							values.put(no, lIssueDiscrete.getValue(optionIndex));
							break;
						case REAL:
							IssueReal lIssueReal = (IssueReal) lIssue;
							double step = (lIssueReal.getUpperBound() - lIssueReal
									.getLowerBound())
									/ lIssueReal
											.getNumberOfDiscretizationSteps();
							bidValue = (int) Math.round((((ValueReal) bid
									.getValue(no)).getValue() - lIssueReal
									.getLowerBound())
									/ step);
							switch (modifications[j]) {
							case 0:
								optionIndex = bidValue;
								break;
							case 1:
								if (bidValue > 0)
									optionIndex = bidValue - 1;
								else
									optionIndex = bidValue;
								break;
							case 2:
								if (bidValue + 1 < lIssueReal
										.getNumberOfDiscretizationSteps())
									optionIndex = bidValue + 1;
								else
									optionIndex = bidValue;
								break;
							}
							values.put(
									Integer.valueOf(lIssueReal.getNumber()),
									new ValueReal(
											lIssueReal.getLowerBound()
													+ (lIssueReal
															.getUpperBound() - lIssueReal
															.getLowerBound())
													* optionIndex
													/ lIssueReal
															.getNumberOfDiscretizationSteps()));
							break;

						case INTEGER:
							IssueInteger lIssueInteger = (IssueInteger) lIssue;
							bidValue = ((ValueInteger) bid.getValue(no))
									.getValue();
							switch (modifications[j]) {
							case 0:
								optionIndex = bidValue;
								break;
							case 1:
								if (bidValue > lIssueInteger.getLowerBound())
									optionIndex = bidValue - 1;
								else
									optionIndex = bidValue;
								break;
							case 2:
								if (bidValue + 1 < lIssueInteger
										.getUpperBound())
									optionIndex = bidValue + 1;
								else
									optionIndex = bidValue;
								break;
							}
							values.put(
									Integer.valueOf(lIssueInteger.getNumber()),
									new ValueInteger(optionIndex));
							break;

						default:
							throw new Exception("issue type "
									+ lIssue.getType() + " not supported");
						}
					}

					// Generate bid and check if it is better than current bid
					currentBid = new Bid(getUtilitySpace().getDomain(), values);
					utility = getUtilitySpace().getUtility(currentBid);
					if (utility > maxUtility) {
						maxUtilityBid = currentBid;
						maxUtility = utility;
					}

					ready = false;

					// Reasonable number of issues means that we try out all the
					// combinations
					if (!tooManyIssues) {
						// Compute sum of modifications
						for (i = 0, sum = 0; i < issues.size(); sum += modifications[i], i++)
							;

						// If we can further try another modification
						if (sum < 2 * issues.size()) {
							for (i = 0; i < issues.size(); i++) {
								modifications[i]++;
								if (modifications[i] == 3) {
									modifications[i] = 0;
									continue;
								} else {
									break;
								}
							}
						}

						ready = (sum < 2 * issues.size());

						// If there are too many issues, just try out randomly
						// another bid and increment the trial counter
					} else {
						numTrials++;

						for (i = 0; i < issues.size(); i++)
							modifications[i] = getRandom().nextInt(3);

						ready = numTrials < REASONABLE_NUMBER_OF_TRIALS;
					}
				} while (!ready);
			} catch (Exception ex) {
				ex.printStackTrace();
			}

			return maxUtilityBid;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see ro.pub.cs.anac.strategies.Strategy#respondToBid(double)
		 */
		public Bid respondToBid(double currentTime) {
			if (getLastOpponentBid() == null)
				return getMaxUtilityBid();
			else {
				try {
					double targetUtility = 1 - currentTime * 0.2;
					double lastOpponentBidUtility = getUtilitySpace()
							.getUtility(getLastOpponentBid());

					// Accept a bid with utility at least the target utility, or
					// make a concession
					// if time is almost over
					if (lastOpponentBidUtility >= targetUtility
							|| (currentTime >= 0.85 && lastOpponentBidUtility >= 0.85 * targetUtility))
						return null;
					// Otherwise, just search for a better bid in the
					// neighbourhood of the last bid
					else {
						Bid neighbour = searchBidInNeighbourhoodOf(getLastOpponentBid());
						double neighbourUtility = getUtilitySpace().getUtility(
								neighbour);

						if (neighbourUtility > bestNeighbourUtility) {
							bestNeighbourUtility = neighbourUtility;
							bestNeighbour = neighbour;
						}

						/*
						 * System.out.println(
						 * "Searching for neighboring bid with utility = " +
						 * targetUtility + ", found utility " +
						 * neighbourUtility);
						 */
						/*
						 * System.out.println(
						 * "Bidding with best neighbour utility: " +
						 * bestNeighbourUtility);
						 */

						if (bestNeighbourUtility >= targetUtility)
							return bestNeighbour;
						else
							return searchRandomBidWithMinimalUtility(targetUtility);
					}

				} catch (Exception ex) {
					ex.printStackTrace();
					return null;
				}
			}
		}

	}

	private static final double MAX_WEIGHT_CHANGE = 1.0;
	private static final double EPS = 0.00001;
	private List<Strategy> strategies;
	private List<Double> strategyWeights;
	private int currentStrategy;
	private Bid lastBid;
	private Bid lastChameleonBid;
	private double lastChameleonBidTime;

	/*
	 * The version of the agent
	 */
	@Override
	public String getVersion() {
		return "0.1";
	}

	/*
	 * The strategies used by the Chameleon agent
	 */
	private List<Strategy> getStrategies() {
		strategies = new ArrayList<Strategy>();
		strategies.add(new StubbornStrategy());
		strategies.add(new MirrorStrategy());
		strategies.add(new PiggyBackStrategy());
		return strategies;
	}

	/*
	 * Initialization sequence of the agent
	 */
	public void init() {
		strategies = getStrategies();
		for (Strategy strategy : strategies)
			strategy.init((AdditiveUtilitySpace) utilitySpace);

		strategyWeights = new ArrayList<Double>();
		for (int i = 0; i < strategies.size(); i++)
			strategyWeights.add(1.0 / strategies.size());

		currentStrategy = pickAStrategy();
	}

	/*
	 * Picks the next strategy. This is done probabilistically.
	 */
	private int pickAStrategy() {
		double[] probabilities = new double[strategies.size() + 1];

		// Compute the intervals in [0,1], one interval for each strategy
		probabilities[0] = 0.0;
		for (int i = 0; i < strategies.size(); i++) {
			probabilities[i + 1] = probabilities[i] + strategyWeights.get(i);
		}
		probabilities[strategies.size()] = 1.0;

		// Pick a number between 0 and 1 and see which strategy it represents
		double rand = Math.random();
		for (int i = 0; i < strategies.size(); i++)
			if (probabilities[i] <= rand && rand < probabilities[i + 1])
				return i;

		return -1;
	}

	/*
	 * Returns the discount for the given time.
	 */
	private double getDiscount(double time) {
		double discount = utilitySpace.getDiscountFactor();
		if (discount <= 0.0D || discount >= 1.0D)
			return 1;
		return Math.pow(discount, time);
	}

	/*
	 * Print the weights of the strategies
	 */
	private void printWeights() {
		/*
		 * System.out.print("weights: "); for (int i = 0; i <
		 * strategyWeights.size(); i++) System.out.print(strategyWeights.get(i)
		 * + " "); System.out.println();
		 */
	}

	/*
	 * Adjust the weights of the strategies.
	 */
	private void adjustCurrentStrategy(Bid bid, double time) {
		try {
			double bidUtility = utilitySpace.getUtility(bid);
			double lastBidUtility = utilitySpace.getUtility(lastBid);
			double changeInUtility = bidUtility / lastBidUtility - 1;

			// Positive change in utility should be encouraged
			if (changeInUtility > 0) {
				encourageStrategyForPositiveChangeInUtility(changeInUtility);
				// Negative change in utility should be penalized
			} else if (changeInUtility < 0) {
				penalizeStrategyForNegativeChangeInUtility(changeInUtility);
			}

			for (int i = 0; i < strategyWeights.size(); i++)
				if (Math.abs(strategyWeights.get(i) - 1) < EPS) {
					int saveCurrentStrategy = currentStrategy;
					currentStrategy = i;
					penalizeStrategyForNegativeChangeInUtility(-0.2);
					currentStrategy = saveCurrentStrategy;
					break;
				}

			// Change the current strategy
			currentStrategy = pickAStrategy();

		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	/*
	 * Encourages the current strategy if it has given good results.
	 */
	private void encourageStrategyForPositiveChangeInUtility(
			double changeInUtility) {
		double strategyWeight = strategyWeights.get(currentStrategy);

		// There's no point in encouraging an already very encouraged strategy
		if (Math.abs(1 - strategyWeight) <= EPS)
			return;

		// Compute maximum percent with which we can encourage
		double percent = changeInUtility;
		percent = Math.min(percent, MAX_WEIGHT_CHANGE);
		if (Math.abs(strategyWeight) > EPS) {
			percent = Math.min(percent, (1 - strategyWeight) / strategyWeight);
		}

		// System.out.println("Encouraging strategy " + currentStrategy +
		// " for an increase in utility of " + changeInUtility + "(increase = "
		// + percent + ")");
		printWeights();

		// Decrease the other weights
		double sumWeights = 0.0;
		for (int i = 0; i < strategies.size(); i++) {
			double w = strategyWeights.get(i);

			if (i == currentStrategy)
				continue;

			// We only decrease the significant weights.
			// The others are small enough already.
			if (Math.abs(w) <= EPS) {
				sumWeights += w;
				continue;
			}

			w -= w * strategyWeight / (1 - strategyWeight) * percent;
			strategyWeights.set(i, w);
			sumWeights += w;
		}

		// Increase the weight of the current strategy
		strategyWeights.set(currentStrategy, 1 - sumWeights);

		printWeights();
	}

	/*
	 * Discourages the current strategy if it has given bad results.
	 * Unfortunately, we don't allow a second chance because sometimes the
	 * results can be disastruous.
	 */
	private void penalizeStrategyForNegativeChangeInUtility(
			double changeInUtility) {
		double strategyWeight = strategyWeights.get(currentStrategy);
		boolean[] takenIntoAccount = new boolean[strategies.size()];
		double sumTakenIntoAccount = 0.0;
		int numTakenIntoAccount = 0;

		// Compute maximum percent with which we can penalize
		double percent = -changeInUtility;
		percent = Math.min(percent, MAX_WEIGHT_CHANGE);
		for (int i = 0; i < strategies.size(); i++) {
			double w = strategyWeights.get(i);

			if (i == currentStrategy) {
				takenIntoAccount[i] = false;
				continue;
			}

			// If weight of this strategy is almost 0, don't worry about
			// overflowing its weight.
			if (Math.abs(w) <= EPS) {
				takenIntoAccount[i] = true;
				continue;
			}

			// If weight of this strategy is almost 1, it won't be increased
			// because it is already big.
			if (Math.abs(w - 1) <= EPS) {
				takenIntoAccount[i] = false;
				continue;
			}

			// Compute a new value for the percent, which would not overflow the
			// weight of the current strategy.
			percent = Math.min(percent, (1 - w) * (1 - strategyWeight)
					/ (w * strategyWeight));
			takenIntoAccount[i] = true;
		}

		// Compute the sum of the strategies taken into account
		for (int i = 0; i < strategies.size(); i++)
			if (takenIntoAccount[i]) {
				sumTakenIntoAccount += strategyWeights.get(i);
				numTakenIntoAccount++;
			}

		// System.out.println("Discouraging strategy " + currentStrategy +
		// " for a decrease in utility of " + changeInUtility + "(decrease = " +
		// percent + ")");
		printWeights();

		// Increase other weights
		double sumWeights = 0.0;
		for (int i = 0; i < strategies.size(); i++) {
			double w = strategyWeights.get(i);

			if (i == currentStrategy)
				continue;

			if (!takenIntoAccount[i]) {
				sumWeights += w;
				continue;
			}

			// If the strategy we're discouraging doesn't dominate the others
			if (Math.abs(sumTakenIntoAccount) > EPS)
				w += w * strategyWeight / sumTakenIntoAccount * percent;
			// If the strategy we're discouraging dominates the others,
			// increase each of the others equally.
			else {
				w += strategyWeight * percent / numTakenIntoAccount;
			}

			strategyWeights.set(i, w);
			sumWeights += w;
		}

		// Decrease the weight of the current strategy
		strategyWeights.set(currentStrategy, 1 - sumWeights);

		printWeights();
	}

	/*
	 * Callback that is executed whenever the opponent has responded to our
	 * current bid (or initialized the bidding process).
	 */
	public void ReceiveMessage(Action opponentAction) {
		if (opponentAction instanceof Offer) {

			Offer offer = (Offer) opponentAction;
			Bid bid = offer.getBid();
			double time = timeline.getTime();

			// Notify the strategies of the newly-received offer
			for (Strategy strategy : strategies)
				strategy.receiveOpponentBid(bid, time);

			// Adjust the weights
			if (lastBid != null)
				adjustCurrentStrategy(bid, time);

			// Store the last bid
			lastBid = bid;
		}
	}

	/*
	 * Choose our response to a negociation.
	 */
	public Action chooseAction() {
		Action action = null;

		try {
			Strategy strategy = strategies.get(currentStrategy);
			Bid responseBid = strategy.respondToBid(timeline.getTime());
			if (responseBid == null)
				return new Accept(getAgentID(), lastBid);
			else {
				lastChameleonBid = responseBid;
				lastChameleonBidTime = timeline.getTime();
				return new Offer(getAgentID(), responseBid);
			}
		} catch (Exception ex) {
			ex.printStackTrace();
			action = new Accept(this.getAgentID(), lastBid);
		}

		return action;
	}
}