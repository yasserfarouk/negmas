package agents.anac.y2012.MetaAgent.agents.MrFriendly;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.ValueDiscrete;
import genius.core.timeline.TimeLineInfo;

public class OpponentModel {

	/**
	 * HashMap that keeps track for each issue the number of times it has been
	 * changed, with the issue as its key
	 */
	private HashMap<Issue, Integer> issueChanged;

	/**
	 * Keeps a HashMap 'matrix' containing references for each issue to their
	 * set of values, which refer to the number of times this value has been
	 * offered
	 */
	private HashMap<Issue, HashMap<String, Integer>> valuesOffered;

	/**
	 * Keeps a HashMap 'matrix' containing references for each issue to their
	 * set of values, which refer to the number of times this value has been
	 * offered
	 */
	private HashMap<Issue, ArrayList<Double>> relativeFrequency;

	/**
	 * Counts the number of bids made
	 */
	private int bidCounter;

	/**
	 * Contains a list of Issues for this domain
	 */
	private List<Issue> issueList;

	/**
	 * boolean holding 'false' when one of the issues is not discrete
	 */
	private boolean isProperlyInitialized;

	/**
	 * The estimated weights mapped to each issue
	 */
	private HashMap<Issue, Double> issueWeights;

	/**
	 * gaussian kernel
	 */
	private double[] kernel;

	/**
	 * true iff we are negotiating on a domain with utility discount
	 */
	private boolean discountedDomain;

	/**
	 * discount factor
	 */
	private double discountFactor;

	/**
	 * Needed to calculate discounted utility
	 */
	private TimeLineInfo timeline;

	/**
	 * Constructor, needs a list of Issues for this domain
	 * 
	 * @param iList
	 *            ArrayList
	 */
	public OpponentModel(List<Issue> iList, double discountFactor,
			TimeLineInfo tl) {
		issueList = iList;
		issueChanged = new HashMap<Issue, Integer>();
		valuesOffered = new HashMap<Issue, HashMap<String, Integer>>();
		relativeFrequency = new HashMap<Issue, ArrayList<Double>>();
		issueWeights = new HashMap<Issue, Double>();
		bidCounter = 0;
		isProperlyInitialized = true;
		for (Issue i : issueList) {
			issueChanged.put(i, 0);
			addValuesOffered(i);
			relativeFrequency.put(i, new ArrayList<Double>());
		}
		kernel = new double[100];
		for (int i = 1; i < 101; i++) {
			kernel[i - 1] = -5 + (.1 * i);
		}
		this.discountFactor = discountFactor;
		if (discountFactor != 0 && discountFactor != 1) {
			discountedDomain = true;
		}
		timeline = tl;
	}

	/**
	 * Updates the model by analyzing the latest bid our opponent has made,
	 * given as a parameter
	 * 
	 * @param nextBid
	 *            Bid
	 * @throws Exception
	 */
	private void updateModel(Bid nextBid) throws Exception {
		bidCounter++;

		// hashmap to map value-names to the times they have been offered
		HashMap<String, Integer> hm;

		// receiveMessage the frequencies and relative frequencies of values per
		// issue
		for (Issue issue : issueList) {
			// get all value-frequency tuples for this issue
			hm = valuesOffered.get(issue);

			// increment the value that was offered for this issue
			// FIXME this is where the nullpointer comes from, figure it out..
			int c = hm
					.get(((ValueDiscrete) nextBid.getValue(issue.getNumber()))
							.getValue()) + 1;
			hm.put(((ValueDiscrete) nextBid.getValue(issue.getNumber()))
					.getValue(), c);
			valuesOffered.put(issue, hm);

			// calculate and add the relative frequency for the offered value
			// for this issue
			relativeFrequency.get(issue).add(c / ((double) bidCounter));
		}

		// some complicated stuff going on here, check report for explanation
		// (chapter on KDE)
		ArrayList<Double> frequencyComponents;
		double[] probabilityDensity;
		double[] expectedValue = new double[100];
		double totalValue = 0;
		for (Issue i : issueList) {
			probabilityDensity = new double[100];
			frequencyComponents = relativeFrequency.get(i);

			for (double d : frequencyComponents) {
				for (int index = 0; index < 100; index++) {
					probabilityDensity[index] += (Math.exp(-Math.pow(
							(kernel[index] - d), 2) / 2) / (index + 1));
				}
			}
			expectedValue[i.getNumber()] = 0.0;
			for (int j = 0; j < 100; j++) {
				expectedValue[i.getNumber()] += (kernel[j] * probabilityDensity[j]) / 100;
			}
			totalValue += expectedValue[i.getNumber()];
		}
		for (Issue i : issueList) {
			issueWeights.put(i, expectedValue[i.getNumber()] / totalValue);
		}
		// System.out.println("\n\nISSUE WEIGHTS: "+issueWeights.toString()+"\n\n");
	}

	/**
	 * returns the weight our opponent gives to every issue as a HashMap wherein
	 * each Issue refers to the weight it gets
	 * 
	 * @return HashMap
	 */
	public HashMap<Issue, Double> getIssueWeights() {
		return issueWeights;
	}

	/**
	 * Gets the estimated most-preferred value that our opponent wants for given
	 * Issue
	 * 
	 * @param i
	 *            Issue
	 * @return String
	 */
	public String getPreferredValueForIssue(Issue i) {
		int max = 0;
		String current = "";
		for (java.util.Map.Entry<String, Integer> e : valuesOffered.get(i)
				.entrySet()) {
			if (e.getValue() >= max) {
				current = e.getKey();
				max = e.getValue();
			}
		}
		return current;
	}

	/**
	 * Estimates the utility of a particular bid for our opponent
	 * 
	 * @param bid
	 *            Bid
	 * @return double
	 * @throws Exception
	 */
	public double getEstimatedUtility(Bid bid) {
		HashMap<String, Integer> valueCounts;

		double totalUtil = 0;
		for (Issue i : issueList) {
			// we know it must be discrete because we only have discrete
			// issues..
			IssueDiscrete id = (IssueDiscrete) i;
			valueCounts = valuesOffered.get(i);
			int maxcount = 0;
			for (ValueDiscrete vald : id.getValues()) {
				maxcount = (valueCounts.get(vald.getValue()) > maxcount) ? valueCounts
						.get(vald.getValue()) : maxcount;
			}

			try {
				// normalize the frequency-counts by the highest count (which
				// will be utility 1)
				totalUtil += (valueCounts.get(((ValueDiscrete) bid.getValue(i
						.getNumber())).getValue()) / (double) maxcount)
						* issueWeights.get(i);
			} catch (Exception e) {
				return -1; // something went wrong - no bids yet probably, just
							// return -1
			}

		}

		if (discountedDomain) {
			// discountedUtility = originalUtility * d^t
			totalUtil = totalUtil
					* Math.pow(discountFactor, timeline.getTime());
		}

		return totalUtil;
	}

	/**
	 * Returns true iff this opponentmodel was properly initialized (ie. no REAL
	 * or INTEGER issues..)
	 * 
	 * @return boolean
	 */
	public boolean isProperlyInitialized() {
		return isProperlyInitialized;
	}

	/**
	 * Helper method to initialize HashMap 'matrix'
	 * 
	 * @param i
	 *            Issue
	 */
	private void addValuesOffered(Issue i) {
		switch (i.getType()) {
		case DISCRETE:
			IssueDiscrete id = (IssueDiscrete) i;
			List<ValueDiscrete> valsD = id.getValues();
			HashMap<String, Integer> counters = new HashMap<String, Integer>();
			for (ValueDiscrete v : valsD) {
				counters.put(v.getValue(), 0);
			}
			valuesOffered.put(i, counters);
			break;
		case REAL:
			// opponentmodel does not yet know how to handle REAL or INTEGER
			// values;
			// just dont use the model in this case
			isProperlyInitialized = false;
			break;
		case INTEGER:
			isProperlyInitialized = false;
			break;
		}
	}

	/**
	 * Updates the model using the received bid
	 * 
	 * @param lastOpponentBid
	 */
	public void addOpponentBid(Bid lastOpponentBid) {
		try {
			updateModel(lastOpponentBid);
		} catch (Exception e) {
			return;
		}
	}

	/**
	 * Returns true iff we suspect our opponent might be stalling
	 * 
	 * @param consecutiveDiff
	 *            number of consecutive unique bids our opponent has made
	 * @return boolean
	 */
	public boolean isStalling(int consecutiveDiff) {
		// if there were any non-unique bids among the last 3 offered bids, our
		// opponent
		// has stalled
		return (consecutiveDiff < 3);
	}

}
