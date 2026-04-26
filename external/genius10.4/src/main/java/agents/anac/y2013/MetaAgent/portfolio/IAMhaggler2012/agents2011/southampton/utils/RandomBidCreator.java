package agents.anac.y2013.MetaAgent.portfolio.IAMhaggler2012.agents2011.southampton.utils;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.utility.AdditiveUtilitySpace;

public class RandomBidCreator implements BidCreator {

	protected Random random;

	public RandomBidCreator() {
		random = new Random();
	}

	/**
	 * Get a random bid.
	 * 
	 * @param utilitySpace
	 *            The utility space to generate the random bid from.
	 * @return a random bid.
	 */
	private Bid getRandomBid(AdditiveUtilitySpace utilitySpace) {
		Domain domain = utilitySpace.getDomain();
		HashMap<Integer, Value> values = new HashMap<Integer, Value>();
		List<Issue> issues = domain.getIssues();
		Bid bid = null;
		for (Iterator<Issue> iterator = issues.iterator(); iterator.hasNext();) {
			Issue issue = (Issue) iterator.next();
			switch (issue.getType()) {
			case DISCRETE:
				generateValue(values, (IssueDiscrete) issue);
				break;

			case REAL:
				generateValue(values, (IssueReal) issue);
				break;

			case INTEGER:
				generateValue(values, (IssueInteger) issue);
				break;
			}
		}

		try {
			bid = new Bid(domain, values);
		} catch (Exception e) {
		}
		return bid;
	}

	protected void generateValue(HashMap<Integer, Value> values,
			IssueDiscrete issue) {
		int randomDiscrete = random.nextInt(issue.getNumberOfValues());
		values.put(Integer.valueOf(issue.getNumber()),
				issue.getValue(randomDiscrete));
	}

	protected void generateValue(HashMap<Integer, Value> values, IssueReal issue) {
		double randomReal = issue.getLowerBound() + random.nextDouble()
				* (issue.getUpperBound() - issue.getLowerBound());
		values.put(Integer.valueOf(issue.getNumber()),
				new ValueReal(randomReal));
	}

	protected void generateValue(HashMap<Integer, Value> values,
			IssueInteger issue) {
		int randomInteger = issue.getLowerBound()
				+ random.nextInt(issue.getUpperBound() - issue.getLowerBound()
						+ 1);
		values.put(Integer.valueOf(issue.getNumber()), new ValueInteger(
				randomInteger));
	}

	/**
	 * Get a random bid (above a minimum utility value if possible).
	 * 
	 * @param utilitySpace
	 *            The utility space to generate the random bid from.
	 * @param min
	 *            The minimum utility value.
	 * @return a random bid (above a minimum utility value if possible).
	 */
	protected Bid getRandomBid(AdditiveUtilitySpace utilitySpace, double min) {
		if (min > 1.0)
			return null;
		int i = 0;
		while (true) {
			Bid b = getRandomBid(utilitySpace);
			try {
				double util = utilitySpace.getUtility(b);
				if (util >= min) {
					// printVal(util);
					return b;
				}
			} catch (Exception e) {
			}
			i++;
			if (i == 500) {
				min -= 0.01;
				i = 0;
			}
		}
	}

	/**
	 * Get a random bid (within a utility range if possible).
	 * 
	 * @param utilitySpace
	 *            The utility space to generate the random bid from.
	 * @param min
	 *            The minimum utility value.
	 * @param max
	 *            The maximum utility value.
	 * @return a random bid (within a utility range if possible).
	 */
	public Bid getRandomBid(AdditiveUtilitySpace utilitySpace, double min,
			double max) {
		// printRange(min, max);
		// System.out.println("Get bid in range ["+min+", "+max+"]");
		int i = 0;
		while (true) {
			if (max >= 1) {
				return getRandomBid(utilitySpace, min);
			}
			Bid b = getRandomBid(utilitySpace);
			try {
				double util = utilitySpace.getUtility(b);
				if (util >= min && util <= max) {
					// printVal(util);
					return b;
				}
			} catch (Exception e) {
			}
			i++;
			if (i == 500) {
				max += 0.01;
				i = 0;
			}
		}
	}

	@Override
	public Bid getBid(AdditiveUtilitySpace utilitySpace, double min, double max) {
		return getRandomBid(utilitySpace, min, max);
	}

	/*
	 * private void printVal(double util) { for(int i = 0; i < util*100; i++) {
	 * System.out.print(" "); } System.out.println("^"); }
	 * 
	 * private void printRange(double min, double max) { min = Math.max(min, 0);
	 * max = Math.min(max, 1); int i = 0; for(; i < min*100; i++) {
	 * System.out.print(" "); } for(; i < max*100; i++) { System.out.print("-");
	 * } System.out.println(); }
	 */

	@Override
	public Bid logBid(Bid opponentBid, double time) {
		return null;
	}
}
