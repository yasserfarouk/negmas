package agents.anac.y2015.pokerface;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.DefaultAction;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.misc.Range;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.timeline.ContinuousTimeline;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.Evaluator;

/**
 * This is your negotiation party.
 */
public class PokerFace extends AbstractNegotiationParty {

	/**
	 * Please keep this constructor. This is called by genius.
	 *
	 * @param utilitySpace
	 *            Your utility space.
	 * @param deadlines
	 *            The deadlines set for this negotiation.
	 * @param timeline
	 *            Value counting from 0 (start) to 1 (end).
	 * @param randomSeed
	 *            If you use any randomization, use this seed for it.
	 */
	private ArrayList<Bid> high_utility_bids = null;
	private OpponentBidLists opponent_bid_list;

	private final double RANDOM_WALKER_TRESHOLD = 0.6;
	private final double MINIMAL_WALKER_UTILITY = 0.85;

	private final double CONCEDE_SPEED = (double) 1 / 4;
	private final double CONCEDE_MINIMUM = 0.5;
	private final double CONCEDE_TO = 0.5;
	private Bid last_received_bid = null;
	private Bid last_concede_bid = null;
	private Bid final_bid = null;
	private BigInteger current_step = BigInteger.valueOf(1);
	private int first_to_offer = -1;
	private double total_time = 0;
	private double current_time = 0;

	private int rounds = 0; // rounds seen so far
	private double[] last_rounds_time; // array to store time of last rounds,
										// used in moving average
	private final int MA_SIZE = 10;
	private double average_time = 0.001; // average time taken for a round (if
											// time is continuous) to avoid
											// infinity, initial non 0

	private Random random;

	@Override
	public void init(NegotiationInfo info) {
		// Make sure that this constructor calls it's parent.
		super.init(info);

		random = new Random(info.getRandomSeed());
		opponent_bid_list = new OpponentBidLists(
				(AdditiveUtilitySpace) utilitySpace, true);
		total_time = timeline.getTotalTime();
		last_rounds_time = new double[MA_SIZE];
	}

	/**
	 * Each round this method gets called and ask you to accept or offer. The
	 * first party in the first round is a bit different, it can only propose an
	 * offer.
	 *
	 * @param validActions
	 *            Either a list containing both accept and offer or only offer.
	 * @return The chosen action.
	 */
	@SuppressWarnings("rawtypes")
	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		// with 50% chance, counter offer
		// if we are the first party, also offer.
		double time_left, time_after_walker, temp;
		// ensure time_left is in steps we can still take
		if (timeline instanceof ContinuousTimeline) {
			time_left = Math.floor(
					(total_time - timeline.getCurrentTime()) / average_time);
			time_after_walker = (total_time / average_time)
					* (1.0 - RANDOM_WALKER_TRESHOLD);
			temp = time_left / time_after_walker;
		} else {
			time_left = total_time - rounds - 1;
			time_after_walker = ((total_time - 1)
					* (1.0 - RANDOM_WALKER_TRESHOLD));
			temp = (time_left - 1.0) / time_after_walker;
		}
		double threshold = CONCEDE_TO + Math.pow(temp, CONCEDE_SPEED) / 2;
		double last_utility;
		try {
			last_utility = utilitySpace.getUtility(last_received_bid);
		} catch (Exception e) {
			last_utility = 1.0;
		}
		if (first_to_offer == -1) {
			if (!validActions.contains(Accept.class)) {
				first_to_offer = 1;
			} else {
				first_to_offer = 0;
			}
		}
		if (!validActions.contains(Accept.class) || (last_utility < threshold
				&& (time_left > 1 || first_to_offer == 1))) {
			if (((timeline instanceof ContinuousTimeline ? current_time
					: rounds) / total_time) < RANDOM_WALKER_TRESHOLD) {
				final_bid = randomWalker();
			} else {
				if ((int) time_left < 3) {
					// Opponent is not conceding. Deadline has been reached.
					// Let them taste a bit of their own medicine. (They have to
					// accept!)
					try {
						final_bid = utilitySpace.getMaxUtilityBid();
					} catch (Exception e) {
						// fail safe
						final_bid = randomWalker();
					}
				} else {
					List<Object> senders = opponent_bid_list.getSenders();
					List<Entry<Pair<Integer, Value>, Integer>> pair_frequency = opponent_bid_list
							.getMostFrequentIssueValues(senders.get(0));
					List<Entry<Pair<Integer, Value>, Double>> weighted_list = opponent_bid_list
							.weightIssueValues(pair_frequency);
					final_bid = concederBid(weighted_list);
				}
			}
			updateTime();
			return new Offer(getPartyId(), final_bid);

		} else {
			updateTime();
			return new Accept(getPartyId(), last_received_bid);
		}
	}

	/**
	 * All offers proposed by the other parties will be received as a message.
	 * You can use this information to your advantage, for example to predict
	 * their utility.
	 *
	 * @param sender
	 *            The party that did the action.
	 * @param action
	 *            The action that party did.
	 */
	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);

		// Here you can listen to other parties' messages
		Bid bid = DefaultAction.getBidFromAction(action);
		if (bid != null) {
			opponent_bid_list.insertBid(sender, bid);
			last_received_bid = bid;
		}
	}

	private Bid concederBid(
			List<Entry<Pair<Integer, Value>, Double>> ordered_pair) {
		int actions_left;
		int actions_after_walker;
		if (timeline instanceof ContinuousTimeline) {
			actions_left = (int) Math
					.round((total_time - current_time) / average_time);
			actions_after_walker = (int) ((total_time / average_time)
					* (1.0 - RANDOM_WALKER_TRESHOLD));
		} else {
			actions_left = (int) total_time - rounds - 1;
			actions_after_walker = (int) (total_time
					* (1.0 - RANDOM_WALKER_TRESHOLD));
		}
		Bid average_bid = randomWalker();
		// calculate how far in the pairset the best bid occurred
		List<Issue> issues = average_bid.getIssues();
		Map<Integer, Value> values = average_bid.getValues();
		List<Pair<Integer, Value>> issue_pairs = new ArrayList<Pair<Integer, Value>>();
		for (int i = 0; i < values.size(); i++) {
			Integer issue_id = issues.get(i).getNumber();
			Value value_id = values.get(i + 1);
			issue_pairs.add(new Pair<Integer, Value>(issue_id, value_id));
		}
		int max_index = 0;
		for (int i = 0; i < issue_pairs.size(); i++) {
			int index = -1;
			for (int j = 0; j < ordered_pair.size(); j++) {
				Pair<Integer, Value> pair = ordered_pair.get(j).getKey();
				if (pair.equals(issue_pairs.get(i))) {
					index = j;
					break;
				}
			}
			if (index != -1 && index > max_index) {
				max_index = index;
			}
		}
		BigInteger total_steps = BigInteger.valueOf(2).pow(max_index);
		BigInteger steps_per_action = total_steps
				.divide(BigInteger.valueOf(actions_after_walker));
		BigInteger previous_step = current_step;
		current_step = steps_per_action.multiply(
				BigInteger.valueOf(actions_after_walker - actions_left));

		Bid concede_bid;

		try {
			concede_bid = utilitySpace.getMaxUtilityBid();
			if (last_concede_bid == null) {
				last_concede_bid = concede_bid;
			}
		} catch (Exception e) {
			// fail safe
			concede_bid = average_bid;
		}

		// binary concede bid
		concede_bid = generateConcedeBid(current_step, concede_bid,
				ordered_pair, max_index);
		for (BigInteger i = previous_step; i.compareTo(
				current_step) == -1; i = i.add(BigInteger.valueOf(1))) {
			Bid new_bid = new Bid(concede_bid);
			new_bid = generateConcedeBid(i, new_bid, ordered_pair, max_index);
			try {
				if (utilitySpace.getUtility(new_bid) > utilitySpace
						.getUtility(concede_bid)) {
					// System.out.println("===\n new_bid:
					// "+utilitySpace.getUtility(new_bid)+" concede_bid:
					// "+utilitySpace.getUtility(concede_bid));
					concede_bid = new_bid;
				}
			} catch (Exception e) {
			}
		}
		final_bid = concede_bid;
		try {

			if (utilitySpace.getUtility(final_bid) > CONCEDE_MINIMUM) {
				// System.out.println("Conceded,
				// Utility:"+utilitySpace.getUtility(concede_bid)+" and
				// "+actions_left+" steps to go.");
				return final_bid;
			} else {
				// System.out.println("Average Bid,
				// Utility:"+utilitySpace.getUtility(average_bid)+" and
				// "+actions_left+" steps to go.");
				return average_bid;
			}
		} catch (Exception e) {
			return average_bid;
		}
	}

	private Bid generateConcedeBid(BigInteger step, Bid bid,
			List<Entry<Pair<Integer, Value>, Double>> ordered_pair,
			int issue_count) {
		for (int i = 0; i < issue_count; i++) {
			// binary split index to steps to include or not (max_index-bits
			// number)
			BigInteger bit = BigInteger.valueOf(2).pow(i);
			if (step.and(bit).equals(bit)) {
				Pair<Integer, Value> pair = ordered_pair.get(i).getKey();
				int issue = pair.getInteger();
				Value value = pair.getValue();
				bid = bid.putValue(issue, value);
			}
		}
		return bid;
	}

	private Bid randomWalker() {
		List<Bid> bids = getHighUtilityBids(MINIMAL_WALKER_UTILITY);
		double rnd = random.nextDouble();
		double[] W = getWeights(bids);
		int index = 0;
		double sum = 0;
		for (int i = 0; i < W.length; i++) { // Sum W until bigger then rnd
			index = i;
			sum += W[i];
			if (sum > rnd) {
				break;
			}
		}
		// System.out.println("Random bid: "+bids.get(index).toString());
		return bids.get(index);
	}

	private List<Bid> getHighUtilityBids(double minimal_utility) {
		if (high_utility_bids == null) { // singleton
			high_utility_bids = new ArrayList<Bid>();
			SortedOutcomeSpace sorted_outcome = new SortedOutcomeSpace(
					utilitySpace);
			Range r = new Range(minimal_utility, 1.0);
			List<BidDetails> bid_list = sorted_outcome.getBidsinRange(r);
			for (int i = 0; i < bid_list.size(); i++) {
				high_utility_bids.add(bid_list.get(i).getBid());
			}
		}
		return high_utility_bids;
	}

	private double[] getWeights(List<Bid> high_bids_list) {
		int issue_length = high_bids_list.get(0).getIssues().size();
		int datalength = high_bids_list.size();
		// get mean values of the issues
		double[] mean = new double[issue_length];
		for (int i = 0; i < datalength; i++) {
			double[] v = getIssueValues(high_bids_list.get(i));
			for (int j = 0; j < issue_length; j++) {
				mean[j] += v[j];
			}
		}
		// get variance and distance
		double[][] dist = new double[datalength][issue_length];
		double[] sum = new double[issue_length];
		double[] sum_sq = new double[issue_length];
		for (int i = 0; i < datalength; i++) {
			double[] v = getIssueValues(high_bids_list.get(i));
			for (int j = 0; j < issue_length; j++) {
				dist[i][j] = Math.abs(v[j] - mean[j]);
				sum[j] += dist[i][j];
				sum_sq[j] += sum[j] * sum[j];
			}
		}
		double[] var = new double[issue_length];
		double sum_var = 0.0;
		for (int i = 0; i < issue_length; i++) {
			var[i] = (sum_sq[i] - (sum[i] * sum[i] / datalength))
					/ (datalength - 1);
			sum_var += var[i];
		}
		// normalize variance and distance
		for (int i = 0; i < datalength; i++) {
			for (int j = 0; j < issue_length; j++) {
				dist[i][j] = dist[i][j] / sum[j];
			}
		}
		for (int i = 0; i < issue_length; i++) {
			var[i] = var[i] / sum_var;
		}

		// calculate weights for every sample
		double[] W = new double[datalength];
		for (int i = 0; i < datalength; i++) {
			double sample_sum = 0.0;
			for (int j = 0; j < issue_length; j++) {
				sample_sum += dist[i][j] * var[j];
			}
			W[i] = sample_sum;
		}
		return W;
	}

	private double[] getIssueValues(Bid b) {
		int size = b.getIssues().size();
		double[] w = new double[size];
		for (int j = 0; j < size; j++) {
			Issue issue = b.getIssues().get(j);
			double eval;
			try {
				Evaluator evaluator = ((AdditiveUtilitySpace) utilitySpace)
						.getEvaluator(issue.getNumber());
				eval = evaluator.getWeight() * evaluator.getEvaluation(
						(AdditiveUtilitySpace) utilitySpace, b,
						issue.getNumber());
			} catch (Exception e) {
				eval = 0.0;
			}
			w[j] = eval;
		}
		return w;
	}

	// manage updating time variables
	private void updateTime() {
		rounds++;
		// with continuous time it is more involved;
		// we store the average time taken per round to estimate time remaining
		// using an moving average prevents averaging over the full timeline
		// where
		// different tactics might be used with different time complexity
		if (timeline instanceof ContinuousTimeline) {
			average_time -= last_rounds_time[rounds % MA_SIZE] / MA_SIZE;
			last_rounds_time[rounds
					% MA_SIZE] = (timeline.getCurrentTime() - current_time)
							/ rounds;
			average_time += last_rounds_time[rounds % MA_SIZE] / MA_SIZE;
			current_time = timeline.getCurrentTime();
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2015";
	}

}
