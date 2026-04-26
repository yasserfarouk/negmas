package agents.anac.y2014.Flinch;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.NegotiationResult;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;

public class Flinch extends Agent {
	private Action actionOfPartner = null;

	private ArrayList<BidDetails> history = null; // opponent's bid history
	private ArrayList<BidDetails> myHistory = null; // my bid history
	private Bid bestBidFromOpponent = null;

	private double estimatedMaxUtil = 1.0; // estimated best utility for me,
											// default at 1.0
	private boolean defaultEMU = true;

	private double acceptThreshold = 0.0;

	private int round = 0;

	// parameter for acceptThreshold calculation
	private double t_lambda;
	private final double t_lambda_min = 0.1;
	private final double t_lambda_max = 0.8; // @POINT
	// private final double t_lambda_max = 1.0;

	private final double beta = 2;
	private final double p_max = 0.95;
	private final double p_min = 0.84; // @POINT

	// parameter for the assess function
	private double Kop; // dynamic

	// parameter for kernel function
	private double h; // dynamic

	private final int MINIMUM_TRAIN_DATA = 20;
	// private final double MINIMUM_PANIC_CHECK_TIME = 0.9;

	// parameter for GA
	private int GA_MAX_ITERATIONS = 200;
	private int POP_SIZE = 20;
	private double Ps = 0.6;
	private double Pm = 0.4;

	/**
	 * init is called when a next session starts with the same opponent.
	 */
	@Override
	public void init() {
		history = new ArrayList<BidDetails>();
		myHistory = new ArrayList<BidDetails>();

		t_lambda = getLambda();
	}

	@Override
	public String getVersion() {
		return "1.0";
	}

	@Override
	public String getName() {
		return "Flinch";
	}

	@Override
	public void endSession(NegotiationResult dUtil) {

	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
	}

	@Override
	public Action chooseAction() // main function
	{
		// System.out.println("===============");

		Action action = null;

		try {
			double t = timeline.getTime();
			updateAcceptThreshold(timeline.getTime());
			// System.out.println("updated acceptThreshold = " +
			// acceptThreshold);
			// System.out.println("estimatedMaxUtil = " + estimatedMaxUtil);
			// System.out.println("t = " + t + "\tt_lambda = " + t_lambda);
			// if(bestBidFromOpponent != null)
			// System.out.println("U_bbo = " +
			// utilitySpace.getUtility(bestBidFromOpponent));

			round++;

			if (actionOfPartner == null) {
				Bid myBid = chooseNextBid();
				myHistory.add(
						new BidDetails(myBid, utilitySpace.getUtility(myBid)));

				action = new Offer(getAgentID(), myBid);
			} else {
				if (actionOfPartner instanceof Offer) {
					Bid partnerBid = ((Offer) actionOfPartner).getBid();
					double offeredUtilFromOpponent = utilitySpace
							.getUtility(partnerBid);

					history.add(new BidDetails(partnerBid,
							offeredUtilFromOpponent, t));
					if (bestBidFromOpponent == null
							|| offeredUtilFromOpponent > utilitySpace
									.getUtility(bestBidFromOpponent)) {
						bestBidFromOpponent = partnerBid;
					}

					if (isAcceptable(offeredUtilFromOpponent)) {
						action = new Accept(getAgentID(), partnerBid);
					} else if (shouldTerminate(offeredUtilFromOpponent)) {
						action = new EndNegotiation(getAgentID());
					} else {
						Bid myBid = chooseNextBid();
						myHistory.add(new BidDetails(myBid,
								utilitySpace.getUtility(myBid)));

						action = new Offer(getAgentID(), myBid);
					}
				} else {
					throw new Exception("partner action type should be Offer!");
				}
			}
		} catch (Exception e) {
			System.out.println("Exception in ChooseAction:" + e.getMessage());
			e.printStackTrace();

			action = new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid());
		}

		return action;
	}

	private int estimatedRoundsLeft() {
		int n = history.size();
		if (n <= 10) {
			return 1000; // infinity
		}

		double dur = history.get(n - 1).getTime()
				- history.get(n - 1 - 10).getTime();
		double timeForOneRound = dur / 10;
		double round = (1.0 - timeline.getTime()) / timeForOneRound;

		return (int) round;
	}

	private double getLambda() {
		double beta = 1.5;
		double delta = utilitySpace.getDiscountFactor();

		if (delta > 0.75) {
			beta = 2.5;
		} else if (delta > 0.5) {
			beta = 2;
		} else {
			beta = 1.5;
		}

		return t_lambda_min
				+ (t_lambda_max - t_lambda_min) * Math.pow(delta, beta);
	}

	private double getAcceptThreshold(double t) {
		double ret = 1.0;
		double r = utilitySpace.getReservationValue();
		double M = estimatedMaxUtil;

		double left = Math.max(r, M * p_max);
		double right = Math.max(r, M * p_min);

		double middle = 0.5 * left + 0.5 * right;

		assert left >= middle;
		assert middle >= right;

		if (t <= t_lambda) {
			ret = left + (middle - left) * Math.pow(t / t_lambda, 1 / beta);
		} else { // @POINT
			try {
				right = Math.min(right,
						utilitySpace.getUtility(bestBidFromOpponent));
			} catch (Exception e) {
			}

			ret = middle + (right - middle)
					* Math.pow((t - t_lambda) / (1 - t_lambda), 1 / beta);
		}

		if (Double.isNaN(ret) || Double.isInfinite(ret)) { // @TODO
			System.out.println(
					"Errors in calculating acceptThreshold, default to M*p = "
							+ left);
			ret = left;
		}

		return ret;
	}

	private void updateAcceptThreshold(double t) {
		acceptThreshold = getAcceptThreshold(t);
	}

	private boolean isAcceptable(double offeredUtilFromOpponent) {
		return offeredUtilFromOpponent > acceptThreshold;
	}

	private boolean shouldTerminate(double offeredUtilFromOpponent) {
		return utilitySpace.getReservationValue() > acceptThreshold;
	}

	private double distance(Bid b1, Bid b2) throws Exception {
		double sweight = 0.0;
		double r = 0.0;

		List<Issue> issues = b1.getIssues();
		for (int i = 0; i < issues.size(); i++) {
			Issue issue = issues.get(i);
			double diff = 0.0;

			switch (issue.getType()) {
			case DISCRETE: {
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) issue;
				ValueDiscrete v1d = (ValueDiscrete) b1
						.getValue(issue.getNumber());
				ValueDiscrete v2d = (ValueDiscrete) b2
						.getValue(issue.getNumber());
				if (!v1d.equals(v2d)) {
					diff = 1.0;
				}

				break;
			}
			case REAL: {
				IssueReal lIssueReal = (IssueReal) issue;
				ValueReal v1r = (ValueReal) b1.getValue(issue.getNumber());
				ValueReal v2r = (ValueReal) b2.getValue(issue.getNumber());
				double abs_diff = Math.abs(v1r.getValue() - v2r.getValue());
				diff = abs_diff / (lIssueReal.getUpperBound()
						- lIssueReal.getLowerBound());
				break;
			}
			case INTEGER: {
				IssueInteger lIssueInteger = (IssueInteger) issue;
				ValueInteger v1i = (ValueInteger) b1
						.getValue(issue.getNumber());
				ValueInteger v2i = (ValueInteger) b2
						.getValue(issue.getNumber());
				double abs_diff = Math.abs(v1i.getValue() - v2i.getValue());
				diff = abs_diff / (lIssueInteger.getUpperBound()
						- lIssueInteger.getLowerBound());
				break;
			}

			default:
				throw new Exception("issue type " + issue.getType()
						+ " not supported by Flinch");
			}

			r += diff * 1.0;
			sweight += 1.0;
		}

		assert r / sweight >= 0.0 && r / sweight <= 1.0;

		return r / sweight;
	}

	private double kernel_function(double x) {
		if (x <= 1.0 && x >= -1.0) {
			return Math.pow(1 - x * x, 3);
		}

		return 0.0;
	}

	private double kernel(Bid b1, Bid b2) throws Exception {
		return kernel_function(distance(b1, b2) / h);
	}

	private double estimatedOpponentUtility(Bid b) throws Exception {
		if (history.size() < MINIMUM_TRAIN_DATA) {
			return 1.0;
		}

		int n = history.size();
		double gamma = 1.0;

		double ret = 0.0;
		double k = 1.0;
		double denom = 0.0;

		for (BidDetails bd : history) {
			ret += k * kernel(bd.getBid(), b);
			denom += k;
			k *= gamma;
		}

		return ret / denom;
	}

	private Bid chooseNextBid() throws Exception {
		Bid[] candidates = search(1.0);
		double f = 0.98;
		while (candidates.length == 0) {
			candidates = search(f);
			f *= f;
		}

		// dynamic parameter
		h = 0.3 + (1.0 - 0.3) * Math.random();
		Kop = 0.5 + (0.9 - 0.5) * Math.pow(timeline.getTime(), 0.5); // @POINT

		double[] opUtils = new double[candidates.length];
		double[] myUtils = new double[candidates.length];
		double maxOpUtil = 0.0;
		double maxMyUtil = 0.0;

		double[] score = new double[candidates.length];

		for (int i = 0; i < candidates.length; i++) {
			myUtils[i] = utilitySpace.getUtility(candidates[i])
					/ estimatedMaxUtil;

			if (myUtils[i] > maxMyUtil) {
				maxMyUtil = myUtils[i];
			}
		}
		if (maxMyUtil > 0.0) {
			for (int i = 0; i < candidates.length; i++) {
				myUtils[i] /= maxMyUtil;
				assert myUtils[i] >= 0 && myUtils[i] <= 1.0;
			}
		}

		for (int i = 0; i < candidates.length; i++) {
			opUtils[i] = estimatedOpponentUtility(candidates[i]);

			if (opUtils[i] > maxOpUtil) {
				maxOpUtil = opUtils[i];
			}
		}
		if (maxOpUtil > 0.0) {
			for (int i = 0; i < candidates.length; i++) {
				opUtils[i] /= maxOpUtil;
				assert opUtils[i] >= 0 && opUtils[i] <= 1.0;
			}
		}

		int maxi = 0;
		for (int i = 0; i < candidates.length; i++) {
			score[i] = myUtils[i] * (1 - Kop) + opUtils[i] * Kop;
			if (score[i] > score[maxi])
				maxi = i;
		}

		// select max
		int i = 0;
		for (int j = 1; j < candidates.length; j++) {
			if (score[j] > score[i])
				i = j;
		}
		Bid result = candidates[i];

		for (BidDetails bd : history) {
			double s = bd.getMyUndiscountedUtil();
			if (s > utilitySpace.getUtility(result)) {
				result = bd.getBid();
			}
		}

		return result;
	}

	// ==============================
	// LocalSearch
	// ================================

	public Bid getRandomBid() throws Exception {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>(); // pairs
																		// <issuenumber,chosen
																		// value
																		// string>
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random randomnr = new Random();

		Bid bid = null;

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
				values.put(lIssueReal.getNumber(),
						new ValueReal(lIssueReal.getLowerBound() + (lIssueReal
								.getUpperBound() - lIssueReal.getLowerBound())
								* (optionInd) / (lIssueReal
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
						+ " not supported by Flinch LocalSearch");
			}
		}

		bid = new Bid(utilitySpace.getDomain(), values);

		return bid;
	}

	double sq(double x) {
		return x * x;
	}

	// ======================
	// GA functions
	// =======================

	private double GA_fitness(Bid b) throws Exception {
		return utilitySpace.getUtility(b);
	}

	private Bid GA_select(Bid[] population, double[] score, double N)
			throws Exception { // currently
								// using
								// roulette
		double r = Math.random();

		double s = 0.0;
		for (int i = 0; i < POP_SIZE; i++) {
			double f = score[i];
			if (s <= r && r < s + f / N) {
				return population[i];
			}

			s += f / N;
		}

		assert false;
		return population[POP_SIZE - 1];
	}

	private Bid GA_crossover(Bid b1, Bid b2) throws Exception {
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random randomnr = new Random();
		int k = (int) Math.random() * issues.size();
		Bid ret = new Bid(b1);

		for (int i = k; i < issues.size(); i++) {
			int issuenr = issues.get(i).getNumber();
			ret = ret.putValue(issuenr, b2.getValue(issuenr));
		}

		return ret;
	}

	private Bid GA_mutate(Bid b) throws Exception {
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random randomnr = new Random();
		int k = randomnr.nextInt(issues.size());
		int issuenr = issues.get(k).getNumber();

		Value v = null;

		switch (issues.get(k).getType()) {
		case DISCRETE: {
			IssueDiscrete lIssueDiscrete = (IssueDiscrete) issues.get(k);
			int optionIndex = randomnr
					.nextInt(lIssueDiscrete.getNumberOfValues());
			v = lIssueDiscrete.getValue(optionIndex);
			break;
		}
		case REAL: {
			IssueReal lIssueReal = (IssueReal) issues.get(k);
			int optionInd = randomnr
					.nextInt(lIssueReal.getNumberOfDiscretizationSteps() - 1);
			v = new ValueReal(lIssueReal.getLowerBound()
					+ (lIssueReal.getUpperBound() - lIssueReal.getLowerBound())
							* (optionInd)
							/ (lIssueReal.getNumberOfDiscretizationSteps()));
			break;
		}
		case INTEGER: {
			IssueInteger lIssueInteger = (IssueInteger) issues.get(k);
			int optionIndex2 = lIssueInteger.getLowerBound()
					+ randomnr.nextInt(lIssueInteger.getUpperBound()
							- lIssueInteger.getLowerBound());
			v = new ValueInteger(optionIndex2);
			break;
		}

		default:
			throw new Exception("issue type " + issues.get(k).getType()
					+ " not supported by Flinch LocalSearch");
		}
		assert v != null;

		b = b.putValue(issuenr, v);
		return b;
	}

	private void updateCandidatesSet(Set<Bid> ret, Bid[] population, double f)
			throws Exception {
		double t = timeline.getTime();
		for (int i = 0; i < POP_SIZE; i++) {
			double score = utilitySpace.getUtility(population[i]);

			if (defaultEMU || score > estimatedMaxUtil) {
				defaultEMU = false;
				estimatedMaxUtil = score;
			}

			if (acceptThreshold * f < score) {
				ret.add(population[i]);
			}
		}
	}

	private Bid[] GA(double f) throws Exception {
		Set<Bid> ret = new HashSet<Bid>();
		Bid[] population = new Bid[POP_SIZE];
		double[] score = new double[POP_SIZE];

		for (int i = 0; i < POP_SIZE; i++) {
			population[i] = getRandomBid();
		}

		int iter = 0;
		while (iter < GA_MAX_ITERATIONS) {
			iter++;

			updateCandidatesSet(ret, population, f);

			Bid[] newpopulation = new Bid[POP_SIZE];
			int to_select = (int) Ps * POP_SIZE;

			double sum_f = 0.0;
			for (int i = 0; i < POP_SIZE; i++) {
				score[i] = GA_fitness(population[i]);
				sum_f += score[i];
			}

			for (int i = 0; i < to_select; i++) {
				newpopulation[i] = new Bid(GA_select(population, score, sum_f));
			}

			for (int i = 0; i < POP_SIZE - to_select; i++) {
				Bid b1 = GA_select(population, score, sum_f);
				Bid b2 = GA_select(population, score, sum_f);

				newpopulation[i + to_select] = GA_crossover(b1, b2);
			}

			for (int i = to_select; i < POP_SIZE; i++) {
				if (Math.random() < Pm) {
					newpopulation[i] = GA_mutate(newpopulation[i]);
				}
			}

			population = newpopulation;
		}
		updateCandidatesSet(ret, population, f);

		return ret.toArray(new Bid[0]);
	}

	public Bid[] search(double f) throws Exception {
		return GA(f);
	}

	@Override
	public String getDescription() {
		return "ANAC2014 compatible with non-linear utility spaces";
	}
}
