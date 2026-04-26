package negotiator.boaframework.opponentmodel.agentx;

import java.util.ArrayList;
import java.util.List;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * Class for processing discrete issues, weighting them based on opponent bids.
 * Also contains functionality for opponent stubbornness.
 * 
 * @author E. Jacobs
 *
 */
public class DiscreteIssueProcessor {

	private ArrayList<IssueDiscrete> issueList = new ArrayList<IssueDiscrete>();
	private ArrayList<Double> weightList = new ArrayList<Double>();
	private ArrayList<Double> changeList = new ArrayList<Double>();
	private ArrayList<Bid> pastBidList = new ArrayList<Bid>();

	private Bid previousBid;

	private int nIssues;
	private int nBidRepeats = 0;
	private long nPossibleBids;
	private int nMaxBidMemory = 5;
	private int nBidsInMemory = 0;
	private int nBidsProcessed = 0;
	private int nRepeatLimitOverrides;
	private int nRepeatLimit;
	private int averageBidsPerSecond = 100;
	private int totalTime = 180;

	private double nBidsPerSecond;

	private double stubbornness = 0;
	private double corr; // weighting correction factor

	/**
	 * Creates a DiscreteIssueProcessor for a domain. Use this for opponent
	 * modeling
	 * 
	 * @param d
	 *            The domain
	 */
	public DiscreteIssueProcessor(Domain d) {

		List<Issue> issues = d.getIssues();

		for (Issue i : issues) {
			issueList.add((IssueDiscrete) i);
		}

		nIssues = issueList.size();

		for (int i = 0; i < nIssues; i++) {
			weightList.add(1.0 / nIssues);
			changeList.add(1.0);
		}

		nPossibleBids = d.getNumberOfPossibleBids();
		nRepeatLimit = (totalTime * averageBidsPerSecond) / (int) nPossibleBids;
	}

	/**
	 * Creates a DiscreteIssueProcessor for a domain, given a certain utility.
	 * Use this for modeling your own agent
	 * 
	 * @param u
	 * @param d
	 */
	public DiscreteIssueProcessor(AdditiveUtilitySpace u, Domain d) {

		List<Issue> issues = d.getIssues();

		for (Issue i : issues) {
			issueList.add((IssueDiscrete) i);
			weightList.add(u.getWeight(i.getNumber()));
			changeList.add(1.0);
		}

		nIssues = issueList.size();
	}

	public ArrayList<IssueDiscrete> getIssueList() {
		return issueList;
	}

	/**
	 * Adapts the different issue weights according to the current bid the
	 * opponent has made. Issues with more changes get a lower weight
	 * 
	 * @param currentBid
	 *            Bid done by the opponent
	 * @param time
	 *            Time at which the bid was done
	 */
	public void adaptWeightsByBid(Bid currentBid, double time) {

		processBid(currentBid);

		// correction factor for weight adaptation is
		// number of bids per second
		if (time == 0) {
			nBidsPerSecond = 0;
		} else {
			nBidsPerSecond = (double) nBidsProcessed
					/ (time * (double) totalTime);
		}

		corr = 1 / nBidsPerSecond;

		if (time > 0.05) {
			// more stubborn gets even less added change
			corr = corr * (1 - stubbornness);
		}

		if (previousBid != null) {

			IssueDiscrete issue;

			for (int i = 0; i < nIssues; i++) {

				issue = issueList.get(i);

				ValueDiscrete oldValue = null;
				ValueDiscrete newValue = null;

				try {
					oldValue = (ValueDiscrete) previousBid.getValue(issue
							.getNumber());
					newValue = (ValueDiscrete) currentBid.getValue(issue
							.getNumber());
				} catch (Exception e) {
				}

				if (!oldValue.equals(newValue)) {
					changeList.set(i,
							changeList.get(i) + corr / (changeList.get(i)));
				}
			}
		}

		adaptWeightsByChangeList();

		previousBid = currentBid;
	}

	/**
	 * Gives a descending list of the issues ordered by weight, so the highest
	 * weighted issue is first in the list
	 * 
	 * @return A list of discrete issues, ordered by weight
	 */
	public ArrayList<IssueDiscrete> getOrderedIssueList() {

		ArrayList<IssueDiscrete> orderedIssueList = new ArrayList<IssueDiscrete>();
		ArrayList<IssueDiscrete> otherIssueList = issueList;
		ArrayList<Double> otherWeightList = weightList;

		int maxIndex;

		while (otherWeightList.size() > 1) {

			maxIndex = getIndexHighestWeight(otherWeightList);

			orderedIssueList.add(otherIssueList.get(maxIndex));

			otherIssueList.remove(maxIndex);
			otherWeightList.remove(maxIndex);

		}

		orderedIssueList.add(otherIssueList.get(0));

		return orderedIssueList;
	}

	/**
	 * Gives the weight that belongs to the given issue
	 * 
	 * @param i
	 *            Issue for which the weight is required
	 * @return The weight for the given issue
	 */
	public double getWeightByIssue(IssueDiscrete i) {

		return weightList.get(issueList.indexOf(i));
	}

	/**
	 * Gives the stubbornness of the opponent; closer to 1 is more stubborn.
	 * 
	 * @return
	 */
	public double getStubbornness() {
		return stubbornness;
	}

	/**
	 * Processes each bid to give an indication of the stubbornness of the
	 * opponent
	 * 
	 * @param bid
	 */
	private void processBid(Bid bid) {

		nBidsProcessed++;

		// find the number of bid repeats and change in the list
		// for a new bid, add to list if there is space in the list
		// else decrease the number of repeats
		if (pastBidList.contains(bid)) {
			nBidRepeats++;
		} else if (nBidsInMemory < nMaxBidMemory) {
			pastBidList.add(bid);
			nBidsInMemory++;
		} else {
			nBidRepeats--;
		}

		// check if the repeat limit is crossed, or comes under 0
		// in both cases, reset repeats and allow more different bids to be
		// stored
		// adapt the number of limit overrides accordingly
		if (nBidRepeats > nRepeatLimit) {

			nBidRepeats = 0;
			nMaxBidMemory++;
			nRepeatLimitOverrides++;

		} else if (nBidRepeats < 0) {

			nBidRepeats = 0;
			nMaxBidMemory++;

			if (nRepeatLimitOverrides > 1) {
				nRepeatLimitOverrides--;
			}
		}

		// if more then 20 bids are allowed to be stored, reset to 5 bids
		// storage
		// and add the last 5 bids
		// if there are 5 bids or less, just keep the list.
		if (nMaxBidMemory > 20) {

			ArrayList<Bid> tempBidList = new ArrayList<Bid>();

			if (nBidsInMemory >= 5) {
				for (int i = nBidsInMemory - 5; i < nBidsInMemory; i++) {
					tempBidList.add(pastBidList.get(i));
				}

				pastBidList.clear();
				pastBidList = tempBidList;

				nBidsInMemory = 5;
			}

			nMaxBidMemory = 10;

		}

		if (nBidsProcessed == 0) {
			stubbornness = 0;
		} else {
			stubbornness = nRepeatLimitOverrides
					/ ((double) nBidsProcessed / (double) nRepeatLimit);
		}

		// System.out.println("Updated stubbornness - new value is " +
		// stubbornness + ", total changes:" + changes);

	}

	/**
	 * Returns the index of the highest weight from some weightList
	 * 
	 * @param wList
	 *            The weightList
	 * @return The index of the highest weight.
	 */
	private int getIndexHighestWeight(ArrayList<Double> wList) {

		int maxIndex = 0;
		int nWeights = wList.size();

		for (int i = 1; i < nWeights; i++) {
			if (wList.get(i) > wList.get(maxIndex)) {
				maxIndex = i;
			}
		}

		return maxIndex;
	}

	/**
	 * Sets all the weights based on the list of changes
	 */
	private void adaptWeightsByChangeList() {

		double inverseChangeTotal = 0;
		double normalizedWeight = 0;

		for (int i = 0; i < nIssues; i++) {
			inverseChangeTotal += 1 / changeList.get(i);
		}

		for (int i = 0; i < nIssues; i++) {

			normalizedWeight = (1 / changeList.get(i)) / inverseChangeTotal;

			weightList.set(i, normalizedWeight);
		}
	}

	@Override
	public String toString() {

		String str = "";

		for (int i = 0; i < nIssues; i++) {
			str += "Issue: " + issueList.get(i).getName() + ", changes: "
					+ changeList.get(i) + ", weight: " + weightList.get(i)
					+ "\n";
		}

		return str;

	}

}
