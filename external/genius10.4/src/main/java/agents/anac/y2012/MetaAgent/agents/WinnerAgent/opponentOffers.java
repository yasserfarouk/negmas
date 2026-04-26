package agents.anac.y2012.MetaAgent.agents.WinnerAgent;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.utility.AdditiveUtilitySpace;

import java.util.Set;
import java.util.Vector;

public class opponentOffers {

	private Hashtable<Issue, Hashtable<Key, Integer>> _offers; // a data
																// structure
																// that indexes
																// opponent
																// offers by
																// issues and
																// counts the
																// number of
																// times each
																// value was
																// proposed
	private List<Issue> _issues; // the issues in this domain
	private Hashtable<Issue, List<Key>> _sortedValuesKeys; // for each issue,
															// holds the list of
															// possible values
															// by decreasing
															// number of times
															// proposed
	private int _totalNumOfOffers; // counter for the number of offers received
									// from the opponent
	private Hashtable<Issue, Double> _issueWeights; // holds the approximate
													// learned weights of the
													// issues
	private Hashtable<Issue, Hashtable<Key, Double>> _valuesUtil; // holds the
																	// approximate
																	// learned
																	// utility
																	// for each
																	// value
	private double _ourAvgUtilFromOppOffers; // the average utility for our
												// agent from the opponent
												// offers so far
	private double _ourMaxUtilFromOppOffers; // the max utility for our agent
												// from the opponent offers so
												// far
	private double _oppConcessionRate; // the opponents concession rate (1=no
										// concession, 1.5 means he offered us
										// twice utility than before)
	private AdditiveUtilitySpace _utilitySpace;// the utility space
	private double _avgFlag;
	private ArrayList<Set<Bid>> _oppoentOffersByUtility; // bids offered by
															// opponent
															// separated to bins
															// according to
															// their utility for
															// us

	/**
	 * creating and initializing the opponent's offers and data structure
	 */
	public opponentOffers(AdditiveUtilitySpace utilitySpace, double avgFlag) {
		// initialize data structures
		_avgFlag = avgFlag;
		_utilitySpace = utilitySpace;
		_offers = new Hashtable<Issue, Hashtable<Key, Integer>>();
		_sortedValuesKeys = new Hashtable<Issue, List<Key>>();
		_issues = _utilitySpace.getDomain().getIssues();
		_totalNumOfOffers = 0;
		_ourAvgUtilFromOppOffers = 0;
		_ourMaxUtilFromOppOffers = 0;
		_oppConcessionRate = 1; // no compromising from opponent
		_issueWeights = new Hashtable<Issue, Double>();
		_valuesUtil = new Hashtable<Issue, Hashtable<Key, Double>>();
		intializeOpponentOffersSets();
		// createFrom the data structure with opponent offers indexed by issues
		// and values
		try {
			// for each issue
			for (Issue lIssue : _issues) {
				_sortedValuesKeys.put(lIssue, new ArrayList<Key>());
				// createFrom a hash table of the issue's values (those are the
				// keys)
				// and for each value we save the number of times it was offered
				Hashtable<Key, Integer> h = new Hashtable<Key, Integer>();
				// taking care of the issue possible values
				switch (lIssue.getType()) {
				// the issue is discrete
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					for (Value v : lIssueDiscrete.getValues()) {
						DiscreteKey k = new DiscreteKey(v.toString());
						h.put(k, 0);
						_sortedValuesKeys.get(lIssue).add(k);
					}
					break;
				// the issue is real
				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					BinCreator bcReal = new RealBinCreator();
					ArrayList<DiscretisizedKey> realBins = bcReal.createBins(
							lIssueReal.getLowerBound(),
							lIssueReal.getUpperBound());
					for (DiscretisizedKey key : realBins) {
						h.put(key, 0);
						_sortedValuesKeys.get(lIssue).add(key);
					}
					break;
				// the issue is integer
				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					BinCreator bcInt = new IntBinCreator();
					ArrayList<DiscretisizedKey> intBins = bcInt.createBins(
							lIssueInteger.getLowerBound(),
							lIssueInteger.getUpperBound());
					for (DiscretisizedKey key : intBins) {
						h.put(key, 0);
						_sortedValuesKeys.get(lIssue).add(key);
					}
					break;
				default:
					throw new Exception("issue type " + lIssue.getType()
							+ " not supported by this agent");
				}
				_offers.put(lIssue, h);

			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * sorts the list of keys for values for the given issue according to the
	 * value counts invoked each time a new bid arrives and updates the location
	 * in the list of the proposed value if needed
	 */
	private void updateValuesList(Issue lIssue, Key updatedVal) {
		Integer updatedValCount = (Integer) _offers.get(lIssue).get(updatedVal);
		List<Key> keys = _sortedValuesKeys.get(lIssue);
		keys.remove(updatedVal);
		Key currKey;
		int index = -1;
		Integer currentValCount = 100000;
		while (updatedValCount < currentValCount && index < keys.size() - 1) {
			index++;
			currKey = keys.get(index);
			currentValCount = (Integer) _offers.get(lIssue).get(currKey);

		}
		if (index < keys.size())
			keys.add(index, updatedVal);
		else
			keys.add(updatedVal);
	}

	/**
	 * invoked when a new bid is proposed by the opponent. for each issue, the
	 * count for the proposed value in this bid is updated. in addition, the
	 * sorted values list for this issue is updated if required
	 * 
	 * @param b
	 *            receiveMessage the bidding statistics using bid b.
	 * @return
	 * @throws Exception
	 */
	public boolean updateBid(Bid b) throws Exception {
		_totalNumOfOffers++; // updates number of received offers
		double utilForUsFromBid = _utilitySpace.getUtility(b);
		double prevAvgUtil = _ourAvgUtilFromOppOffers;
		_ourAvgUtilFromOppOffers = (((_ourAvgUtilFromOppOffers) * (_totalNumOfOffers - 1)) + utilForUsFromBid)
				/ _totalNumOfOffers;
		_oppConcessionRate = Math
				.max(1, _ourAvgUtilFromOppOffers / prevAvgUtil);// assuming no
																// opponent will
																// harden it's
																// position in
																// the long
																// term, we
																// allow min
																// concession of
																// 1
		if (utilForUsFromBid > _ourMaxUtilFromOppOffers)
			_ourMaxUtilFromOppOffers = utilForUsFromBid;

		for (Issue lIssue : _issues) {
			int issueNum = lIssue.getNumber();
			Value v = b.getValue(issueNum);
			Integer currentCount = 0;
			switch (v.getType()) {
			case DISCRETE:
				DiscreteKey k = new DiscreteKey(v.toString());
				currentCount = (Integer) _offers.get(lIssue).get(k);
				_offers.get(lIssue).put(k, ++currentCount);
				updateValuesList(lIssue, k);
				break;
			case REAL:
				double realVal = ((ValueReal) v).getValue();
				Enumeration<Key> realKeys = _offers.get(lIssue).keys();
				while (realKeys.hasMoreElements()) {
					DiscretisizedKey currKey = (DiscretisizedKey) realKeys
							.nextElement();
					if (currKey.isInRange(realVal)) {
						currentCount = (Integer) _offers.get(lIssue).get(
								currKey);
						_offers.get(lIssue).put(currKey, ++currentCount);
					}
					updateValuesList(lIssue, currKey);
				}
				break;
			case INTEGER:
				double intVal = ((ValueInteger) v).getValue();
				Enumeration<Key> intKeys = _offers.get(lIssue).keys();
				while (intKeys.hasMoreElements()) {
					DiscretisizedKey currKey = (DiscretisizedKey) intKeys
							.nextElement();
					if (currKey.isInRange(intVal)) {
						currentCount = (Integer) _offers.get(lIssue).get(
								currKey);
						_offers.get(lIssue).put(currKey, ++currentCount);
					}
					updateValuesList(lIssue, currKey);
				}
				break;
			default:
				throw new Exception("issue type " + lIssue.getType()
						+ " not supported by this agent");
			}
		}// for

		// add bid to the appropriate set according to its utility
		if (utilForUsFromBid >= 0.5) {
			int indexInArray = (int) Math.floor((utilForUsFromBid - 0.5) * 20);
			_oppoentOffersByUtility.get(indexInArray).add(b);
		}

		return _ourAvgUtilFromOppOffers >= _avgFlag;
	}

	/**
	 * get our average utility from the opponent's offer
	 * 
	 * @return average utility offered by the opponent.
	 */
	public double getOurAvgUtilFromOppOffers() {
		return _ourAvgUtilFromOppOffers;
	}

	/**
	 * get our max utility from the opponent's offer
	 * 
	 * @return max utility offered by the opponent.
	 */
	public double getOurMaxUtilFromOppOffers() {
		return _ourMaxUtilFromOppOffers;
	}

	/**
	 * get the opponent's concession rate
	 * 
	 * @return opponent's concession rate
	 */
	public double getOppConcessionRate() {
		return _oppConcessionRate;
	}

	/**
	 * calculates an approximation of the opponent's utility for the given bid
	 * issue weights are determined as explained in getIssueWeights method value
	 * utilities are determined as explained in getValueUtility method
	 * 
	 * @param b
	 *            bid from which the utility must be estimated.
	 * @return estimated utility of the opponent's bid.
	 */
	public double getOpponentUtility(Bid b) {
		// check if no bids were made yet, if so nothing we can say
		if (_totalNumOfOffers == 0) {
			return 0;
		}
		double oppUtil = 0;
		Value currVal;
		try {
			for (Issue currIssue : _issues) {
				currVal = b.getValue(currIssue.getNumber());
				switch (currVal.getType()) {
				case DISCRETE:
					DiscreteKey k = new DiscreteKey(currVal.toString());
					oppUtil = oppUtil + _issueWeights.get(currIssue)
							* _valuesUtil.get(currIssue).get(k);

					break;
				case REAL:
					double realVal = ((ValueReal) currVal).getValue();
					Enumeration<Key> realKeys = _offers.get(currIssue).keys();
					while (realKeys.hasMoreElements()) {
						DiscretisizedKey currKey = (DiscretisizedKey) realKeys
								.nextElement();
						if (currKey.isInRange(realVal)) {
							oppUtil = oppUtil + _issueWeights.get(currIssue)
									* _valuesUtil.get(currIssue).get(currKey);
							break;
						}
					}
					break;
				case INTEGER:
					double intVal = ((ValueInteger) currVal).getValue();
					Enumeration<Key> intKeys = _offers.get(currIssue).keys();
					while (intKeys.hasMoreElements()) {
						DiscretisizedKey currKey = (DiscretisizedKey) intKeys
								.nextElement();
						if (currKey.isInRange(intVal)) {
							oppUtil = oppUtil + _issueWeights.get(currIssue)
									* _valuesUtil.get(currIssue).get(currKey);
							break;
						}
					}
					break;
				default:
					throw new Exception("issue type " + currIssue.getType()
							+ " not supported by this agent");
				}
			}// for
			return oppUtil;
		} catch (Exception e) {
			System.out.println("Exception in get opponent utility: "
					+ e.getMessage());
			return 0;
		}
	}

	/**
	 * calculates the normalized weights of the issues according to their
	 * variance issue with the lowest variance has the highest weight (assuming
	 * that this is the reason the other agent won't compromise a lot about this
	 * issue). we compare the maximal counts of values offered by the opponent
	 * as an approximation to the variance (when the maximal count is higher the
	 * variance is lower). we define stub(i) for an issue as
	 * count(mostFrequentValueOffered)/sum for all
	 * issues(mostFrequentValueOffered) this models the opponent's stubbornness
	 * for each issue. the weight is calculated by stub(i,j)/sum(stub(i))
	 */
	private Hashtable<Issue, Double> getIssueWeights() {
		Hashtable<Issue, Double> weights = new Hashtable<Issue, Double>();
		double totalNormalizedCount = totalNormalizedCounts();
		double currNormalaizedCount;
		double currNormalaizedWeight;
		for (Issue currIssue : _issues) {
			currNormalaizedCount = (_offers.get(currIssue)
					.get(_sortedValuesKeys.get(currIssue).get(0)));
			currNormalaizedWeight = currNormalaizedCount / totalNormalizedCount;
			weights.put(currIssue, currNormalaizedWeight);
		}
		return weights;
	}

	/**
	 * returns the total count of all the issues values in the opponent's offers
	 */
	private double totalNormalizedCounts() {
		double totalNormalizedCount = 0;
		for (Issue currIssue : _issues) {
			totalNormalizedCount = totalNormalizedCount
					+ _offers.get(currIssue).get(
							_sortedValuesKeys.get(currIssue).get(0));
		}
		return totalNormalizedCount;
	}

	/**
	 * calculates the utility for the given value we approximate the utility by
	 * dividing the number of offers made with this value by the total number of
	 * offers made. a high ratio indicates a high utility from this value
	 */
	private double getValueUtility(Issue issue, Key value) {
		double utility = 0;
		double valueCount = _offers.get(issue).get(value);
		utility = valueCount / _totalNumOfOffers;
		return utility;
	}

	/**
	 * recalculate weights based on new learning
	 */
	public void updateWeightsAndUtils() {
		_issueWeights = getIssueWeights();
		try {
			for (Issue currIssue : _issues) {
				Hashtable<Key, Double> utils = new Hashtable<Key, Double>();
				for (Key k : _sortedValuesKeys.get(currIssue)) {
					utils.put(k, new Double(getValueUtility(currIssue, k)));
				}
				_valuesUtil.put(currIssue, utils);
			}
		} catch (Exception e) {
			System.out.println("Exception in updateWeightsAndUtils: "
					+ e.getMessage());
		}
	}

	/**
	 * returns a list of the issues, sorted by their the agent's current belief
	 * of their importance to the opponent (assuming that a lesser variance in
	 * values for an issue implies a higher importance for that issue. we
	 * approximate the variance by the count of the most frequent proposed
	 * value)
	 */
	public List<Issue> getIssuesByCounts() {
		List<Issue> sortedIssues = new ArrayList<Issue>();
		try {
			Integer currIssueMaxCount = 1000000;
			Integer newIssueMaxCount;
			Issue currIssue;
			for (Issue newIssue : _issues) {
				List<Key> newIssuekeys = _sortedValuesKeys.get(newIssue);
				int index = 0;
				newIssueMaxCount = _offers.get(newIssue).get(
						newIssuekeys.get(0));
				List<Key> currIssuekeys;
				while (newIssueMaxCount < currIssueMaxCount
						&& index < sortedIssues.size()) {
					currIssue = sortedIssues.get(index);
					currIssuekeys = _sortedValuesKeys.get(currIssue);
					currIssueMaxCount = _offers.get(currIssue).get(
							currIssuekeys.get(0));
					index++;
				}
				// found place in list
				if (index < sortedIssues.size()) {
					sortedIssues.add(index, newIssue);
					index = 0;
				} else {
					sortedIssues.add(newIssue);
					index = 0;
				}
			}
		} catch (Exception e) {
			System.out.println("Exception in getIssuesByCounts:"
					+ e.getMessage());
			return null;
		}
		return sortedIssues;
	}

	/**
	 * printing the utility function of the opponent
	 */
	public void printUtilityFunction() {
		Issue currIssue;
		Entry<Issue, Double> currEntry;
		Iterator<Entry<Issue, Double>> it = _issueWeights.entrySet().iterator();
		while (it.hasNext()) {
			currEntry = it.next();
			currIssue = currEntry.getKey();
			System.out.println("Issue: " + currIssue.getName() + " weight: "
					+ currEntry.getValue());
			Entry<Key, Double> currValueEntry;
			Iterator<Entry<Key, Double>> valueIt = _valuesUtil.get(currIssue)
					.entrySet().iterator();
			while (valueIt.hasNext()) {
				currValueEntry = valueIt.next();
				System.out.println(" value " + currValueEntry.getKey()
						+ "\t\tutility: " + currValueEntry.getValue()
						+ "\t\tcount "
						+ _offers.get(currIssue).get(currValueEntry.getKey()));
			}
		}
	}

	/*
	 * returns a vector with all bids proposed by opponent over the given
	 * threshold
	 */
	public Vector<Bid> getOpponentBidsAboveThreshold(double threshold) {
		Vector<Bid> bids = new Vector<Bid>();
		int stopIndex = (int) Math.floor((threshold - 0.5) * 20); // index of
																	// the
																	// lowest
																	// bin we
																	// should
																	// return
																	// offers
																	// from
		try {
			for (int i = 10; i >= stopIndex; i--) {

				Set<Bid> currSet = _oppoentOffersByUtility.get(i);
				for (Bid b : currSet) {
					if ((i != stopIndex)
							|| (_utilitySpace.getUtility(b) > threshold)) {
						bids.add(b);
					}

				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		return bids;
	}

	/*
	 * initialize the sets of bids proposed by the opponent
	 */
	private void intializeOpponentOffersSets() {
		_oppoentOffersByUtility = new ArrayList<Set<Bid>>();
		// createFrom 11 bins (from 0.5 and on in 0.05 differences)
		for (int i = 0; i < 11; i++) {
			_oppoentOffersByUtility.add(i, new HashSet<Bid>());
		}
	}

	/**
	 * overriding toString method
	 */
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("-----------Opponent Offers----------\n");
		Iterator<Entry<Issue, Hashtable<Key, Integer>>> issueIterator = _offers
				.entrySet().iterator();
		Issue currIssue;
		Entry<Issue, Hashtable<Key, Integer>> currEntry;
		Hashtable<Key, Integer> currValues;
		while (issueIterator.hasNext()) {
			currEntry = issueIterator.next();
			currIssue = currEntry.getKey();
			sb.append("Issue: " + currIssue + "\n");
			currValues = currEntry.getValue();
			Iterator<Entry<Key, Integer>> valueIterator = currValues.entrySet()
					.iterator();
			Key currKey;
			Entry<Key, Integer> currValueEntry;
			while (valueIterator.hasNext()) {
				currValueEntry = valueIterator.next();
				currKey = currValueEntry.getKey();
				sb.append("Value: " + currKey + " Count: "
						+ currValueEntry.getValue() + "\n");
			}
			List<Key> sortedKeys = _sortedValuesKeys.get(currIssue);
			sb.append("sorted values size: " + sortedKeys.size() + "\n");
			sb.append("sorted values: " + sortedKeys.toString() + "\n");
		}
		return sb.toString();
	}
}
