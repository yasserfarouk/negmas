package agents.anac.y2012.MetaAgent.agents.WinnerAgent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.Vector;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * @author WIN-TEAM Amir Harel, Ma'ayan Gafny, Michal Dahan, Ofra Amir Agent
 *         Description is attached in a separate PDF file
 */
public class WinnerAgent2 extends Agent {
	private Action actionOfPartner = null;
	private Vector<Offer> _allReasonableOffers = null;
	private Offer[] _myOffers;
	private int _currentActionsIndex;
	private opponentOffers _opponentOffers;
	private int _numOfPhases;
	private int _nextPhase;

	// **** Stable parameters ****
	private double _minimumAcceptenceUtility = 0.9;
	private final double _finalMinimalAcceptedUtility = 0.65; // the lowest we
																// are prepared
																// to accept in
																// a non
																// zero-sum game
	private final double _zeroSumMinimumAcceptence = 0.5;
	private double _minimumJointUtility = 0.85;
	private final int _numOfPredefinedOffers = 20;
	private double _discountFactor;
	private double _ourConcession;
	private final double _ourMinWeight = 0.7;
	private final double _notZeroSumThreshold = 0.8;
	private boolean _notZeroSumFlag = false;
	private double _zeroSumUtil = 0.1;
	// set the minimum and maximum number of phases - each phase the opponent
	// utility is updated
	// the number of phases is calculated using these parameters and the
	// discount factor
	private int _initialNumOfPhases = 7;
	private int _maxNumOfPhases = 20;
	// use to randomly select the shuffle index AND the offer index
	private final double _randomlySelectProb = 0.30;

	/**
	 * init is called when a next session starts with the same opponent.
	 */
	public void init() {
		_opponentOffers = new opponentOffers(
				(AdditiveUtilitySpace) utilitySpace, _notZeroSumThreshold);
		_allReasonableOffers = (_allReasonableOffers == null ? createSortedOffersArray()
				: _allReasonableOffers);
		_myOffers = randomlySelectOffers(1);
		_currentActionsIndex = 0;
		_nextPhase = 1;

		// computing the number of phases according to the discount factor
		_discountFactor = utilitySpace.getDiscountFactor();
		if (_discountFactor == 0 || _discountFactor == 1) {
			_numOfPhases = _initialNumOfPhases;
			_discountFactor = 1; // equivalent to 0, easier to use for
									// calculations
		} else
			_numOfPhases = (int) (Math.min(Math
					.round(((double) _initialNumOfPhases) / _discountFactor),
					((double) _maxNumOfPhases)));
		_ourConcession = (0.9 - _finalMinimalAcceptedUtility)
				/ (_numOfPhases - 1);
	}

	/**
	 * Creates a sorted array of all possible offers The offers are sorted by
	 * the joint utility of our utility and the opponent utility
	 * 
	 * @return an sorted array of all offers
	 */
	private Vector<Offer> createSortedOffersArray() {
		// a set of all bids
		Set<Bid> bids = new HashSet<Bid>();
		try {

			// createFrom the list of all discrete\discretisized values possible
			// in
			// this domain
			ArrayList<Vector<? extends Value>> issuesVec = createIssueVec();
			List<Issue> issues = utilitySpace.getDomain().getIssues();
			HashMap<Integer, Integer> indexMap = new HashMap<Integer, Integer>();
			for (int i = 0; i < issues.size(); i++) {
				indexMap.put(i, issues.get(i).getNumber());
			}

			// Cartesian product of all possible values
			Loop FirstLoop = new Loop(issuesVec.get(0), indexMap);
			Loop currentLoop = FirstLoop;
			for (int i = 1; i < issuesVec.size(); i++) {
				Loop newLoop = currentLoop.setNext(issuesVec.get(i));
				currentLoop = newLoop;
			}
			Vector<HashMap<Integer, Value>> bidsMap = new Vector<HashMap<Integer, Value>>();
			FirstLoop.iteration(new HashMap<Integer, Value>(), 0, bidsMap,
					10000);
			for (HashMap<Integer, Value> map : bidsMap) {
				Bid bid = new Bid(utilitySpace.getDomain(), map);
				bids.add(bid);
			}

			// put the offers into the returned vector
			Vector<Offer> offers = new Vector<Offer>();
			for (Bid bid : bids) {
				if (utilitySpace.getUtility(bid) >= 0.5)
					offers.add(new Offer(getAgentID(), bid));
			}

			// sort the offers by our utility
			sortOffersByJointUtilities(offers, 1);

			return offers;

		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	/**
	 * Creates the Issue vector
	 * 
	 * @return
	 * @throws Exception
	 */
	private ArrayList<Vector<? extends Value>> createIssueVec()
			throws Exception {
		ArrayList<Vector<? extends Value>> issuesVec = new ArrayList<Vector<? extends Value>>();
		for (Issue lIssue : utilitySpace.getDomain().getIssues()) {
			switch (lIssue.getType()) {
			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
				Vector<Value> discreteVec = new Vector<Value>();

				for (ValueDiscrete v : lIssueDiscrete.getValues()) {
					discreteVec.add(v);
				}
				issuesVec.add(discreteVec);
				break;
			case REAL:
				IssueReal lIssueReal = (IssueReal) lIssue;
				BinCreator bcReal = new RealBinCreator();
				Vector<? extends Value> realVec = bcReal.createValuesVector(
						lIssueReal.getLowerBound(), lIssueReal.getUpperBound());
				issuesVec.add(realVec);
				break;
			case INTEGER:
				IssueInteger lIssueInt = (IssueInteger) lIssue;

				BinCreator bcInt = new IntBinCreator();
				Vector<? extends Value> intVec = bcInt.createValuesVector(
						lIssueInt.getLowerBound(), lIssueInt.getUpperBound());
				issuesVec.add(intVec);
				break;
			default:
				throw new Exception("issue type " + lIssue.getType()
						+ " not supported by this agent");
			}
		}
		return issuesVec;
	}

	/**
	 * Gets out utility weight and select in a randomized way a set of offers
	 * from all the best possible offers
	 * 
	 * @param w
	 * @return
	 */
	private Offer[] randomlySelectOffers(double w) {
		Set<Integer> offersIndex = new HashSet<Integer>();
		try {
			// adding the current best offer for this agent
			offersIndex.add(0);
			int currOfferInd = 1;
			int maxTries = 5000;
			int maxIndex = _allReasonableOffers.size();

			while (offersIndex.size() < _numOfPredefinedOffers
					&& maxIndex > offersIndex.size() && --maxTries > 0) {

				// check if the current index is in array bounds
				if (currOfferInd >= _allReasonableOffers.size()
						||
						// check if the current offer utility is better then the
						// minimal joint utility
						(getJointUtility(
								_allReasonableOffers.get(currOfferInd), w) < _minimumJointUtility)) {
					maxIndex = currOfferInd;
					currOfferInd = 1;
				}

				if ((utilitySpace.getUtility(_allReasonableOffers.get(
						currOfferInd).getBid()) >= _minimumAcceptenceUtility)
						&& (Math.random() <= _randomlySelectProb
								+ (0.05 * _nextPhase) + (0.01 * currOfferInd))) {
					offersIndex.add(currOfferInd);
				}
				currOfferInd++;
			}

			// get offers proposed by the opponent that are above our current
			// threshold
			Vector<Bid> opponentGoodBids = _opponentOffers
					.getOpponentBidsAboveThreshold(_minimumAcceptenceUtility);

			// put the offers into the returned array
			Offer[] offers = new Offer[offersIndex.size()
					+ opponentGoodBids.size()];
			currOfferInd = 0;

			// first - add good bids that we previously received from the
			// opponent (assuming that if he proposed them, they are good for
			// him and are
			// more likely to be accepted)
			for (Bid b : opponentGoodBids) {
				offers[currOfferInd++] = new Offer(getAgentID(), b);
			}
			for (Integer i : offersIndex) {
				offers[currOfferInd++] = _allReasonableOffers.get(i);
			}
			return offers;

		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	/**
	 * Retrieve the version.
	 * 
	 * @return a string with the version number.
	 */
	@Override
	public String getVersion() {
		return "1.5";
	}

	/**
	 * Receive a message from the opponent.
	 * 
	 * @param opponentAction
	 *            the action of the opponents
	 */
	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
	}

	/**
	 * Sort the array of possible offers by the joint utility of our utility and
	 * the opponent utility
	 * 
	 * @param toSort
	 *            - an array of offers, w - our utility weight
	 */
	private void sortOffersByJointUtilities(Vector<Offer> toSort, final double w) {
		// sort the array by both utilities

		Collections.sort(toSort, new Comparator<Offer>() {
			// the comparator
			public int compare(Offer o1, Offer o2) {
				int ans;
				try {
					double o1Utility = utilitySpace.getUtility(o1.getBid());
					double o2Utility = utilitySpace.getUtility(o2.getBid());
					double o1OpUtility = _opponentOffers.getOpponentUtility(o1
							.getBid());
					double o2OpUtility = _opponentOffers.getOpponentUtility(o2
							.getBid());

					if (w == 1) // only our utility counts (but using opponent
								// utility as tie-break)
					{
						if (o1Utility > o2Utility)
							ans = -1;
						else if (o1Utility == o2Utility) {
							ans = ((o1OpUtility >= o2OpUtility) ? -1 : 1);
						} else
							ans = 1;
					} else // use weighted utilities
					{
						// the joint utility of each offer
						double o1mixedUtility = getJointUtility(o1, w);
						double o2mixedUtility = getJointUtility(o2, w);
						double o1Inequality = getOfferInequality(o1);
						double o2Inequality = getOfferInequality(o2);

						if (_nextPhase <= 3
								|| (o1Inequality <= 0.3 && o2Inequality <= 0.3)) // both
																					// offers
																					// utility
																					// difference
																					// is
																					// low
																					// or
																					// haven't
																					// learned
																					// enough
																					// yet
						{
							if (o1mixedUtility > o2mixedUtility)
								ans = -1;
							else if (o1mixedUtility == o2mixedUtility) {
								if (o1OpUtility > o2OpUtility)
									ans = -1;
								else
									ans = 1;
							} else
								ans = 1;
						} else if (o1Inequality > 0.3 && o2Inequality > 0.3) // both
																				// offers
																				// utility
																				// difference
																				// is
																				// high
						{
							if (o1OpUtility > o2OpUtility)
								ans = -1;
							else
								ans = 1;
						} else // one difference is high, other is low
						{
							if (o1Inequality > 0.3) // o1 utility difference is
													// high
							{
								ans = 1;
							} else // o2 utility difference is high
							{
								ans = -1;
							}
						}
					}
					return ans;
				} catch (Exception e) {
					e.printStackTrace();
				}
				return (_opponentOffers.getOpponentUtility(o1.getBid()) >= _opponentOffers
						.getOpponentUtility(o2.getBid()) ? -1 : 1);
			}
		});
	}

	/**
	 * calculate the joint utility of the given offer, using weight w
	 */
	private double getJointUtility(Offer o, final double w) {
		double mixedUtility = 0;
		try {
			if (w == 1) // only our utility counts
			{
				mixedUtility = utilitySpace.getUtility(o.getBid());
			} else {
				// both utilities count, each has it's weight
				// the opponent utility is approximated by learning
				mixedUtility = (w * utilitySpace.getUtility(o.getBid()))
						+ ((1 - w) * _opponentOffers.getOpponentUtility(o
								.getBid()));
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return mixedUtility;
	}

	/**
	 * returns the difference between our utility from an offer and the opponent
	 * utility from the same offer
	 */
	private double getOfferInequality(Offer o) {
		double diff = 0;
		try {
			diff = Math.abs(utilitySpace.getUtility(o.getBid())
					- _opponentOffers.getOpponentUtility(o.getBid()));
		} catch (Exception e) {
			e.printStackTrace();
		}
		return diff;
	}

	/**
	 * Choose the next action.
	 * 
	 * @return the next action to perform
	 */
	public Action chooseAction() {
		Action action = null;
		double ourWeight;
		try {
			// check to see in which phase we are (according to the time and
			// number of phases)
			if (timeline.getTime() > (((double) _nextPhase) / _numOfPhases)) {
				// determine our utility weight according to the phase
				ourWeight = Math.max(_ourMinWeight, 1 - (0.05 * _nextPhase));
				_minimumJointUtility = _minimumJointUtility - 0.05;

				// receiveMessage minimal acceptance utility based on discount
				// factor
				// and opponents concession rate and best offer
				double assumedBestOfferWeGetNext = _opponentOffers
						.getOurMaxUtilFromOppOffers()
						* (Math.max(_opponentOffers.getOppConcessionRate(), 1.1))
						* (Math.pow(_discountFactor, (double) (_nextPhase + 1)
								/ _numOfPhases));
				_minimumAcceptenceUtility = Math.max(
						(_minimumAcceptenceUtility - _ourConcession), Math.min(
								((_minimumAcceptenceUtility - _ourConcession)),
								assumedBestOfferWeGetNext));
				_minimumAcceptenceUtility = Math.max(0.65,
						_minimumAcceptenceUtility); // make sure we don't go
													// under 0.65 unless it's
													// zero-sum game
				// receiveMessage our knowledge about the opponent from what
				// we've
				// learned so far
				_opponentOffers.updateWeightsAndUtils();
				if (_opponentOffers.getOurAvgUtilFromOppOffers() <= _zeroSumUtil) {
					_minimumAcceptenceUtility = Math.max(
							_minimumAcceptenceUtility - 0.05,
							_zeroSumMinimumAcceptence);
				}
				updateBids(ourWeight);

				// receiveMessage to the next phase
				_nextPhase++;
			}

			// if the opponent hadn't proposed an offer, propose it an offer
			if (actionOfPartner == null) {
				action = _myOffers[updateIndex()];
			}

			// if the opponent had proposed an offer, check if its utility is
			// above the acceptance threshold.
			// If so- accept, else- offer an offer to the opponent.
			else if (actionOfPartner instanceof Offer) {
				// receiveMessage opponentOffers with the new bid
				Offer proposed = (Offer) actionOfPartner;
				if (_opponentOffers.updateBid(proposed.getBid())) {
					if (_notZeroSumFlag == false)
						updateNotZeroSum();
				}
				if (utilitySpace.getUtility(proposed.getBid()) > _minimumAcceptenceUtility) {
					action = new Accept(getAgentID(), proposed.getBid());
				} else {
					action = _myOffers[updateIndex()];
				}
			}
		} catch (Exception e) {
			System.out.println("Exception in ChooseAction:" + e.getMessage());
			action = new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid());
		}

		return action;
	}

	/**
	 * Updates the minimum acceptance utility and the minimum joint utility.
	 * 
	 * @return the next action to perform
	 */
	private void updateNotZeroSum() {
		double opponentAvgUtil = _opponentOffers.getOurAvgUtilFromOppOffers();
		_minimumAcceptenceUtility = Math.max(_minimumAcceptenceUtility,
				opponentAvgUtil);
		_minimumJointUtility = Math.max(_minimumJointUtility, opponentAvgUtil);
		_ourConcession = 0.01;
		_notZeroSumFlag = true;
	}

	/**
	 * Choose the next index to be used in the offers array. The array ordering
	 * is chosen stochastically according to the shuffle probability defined.
	 * 
	 * @return the next array index to use
	 */
	private int updateIndex() {
		int curr = _currentActionsIndex;
		_currentActionsIndex = (_currentActionsIndex++) % _myOffers.length;
		if (Math.random() <= _randomlySelectProb) {
			shuffle(_myOffers);
		}
		return curr;
	}

	/**
	 * Update the bids in the bids array. The array is sorted according to the
	 * utilities. Then 50% of the offers (the ones with less utility) are
	 * removed and are replaced with new offers. If more than 50% of the offers
	 * have a utility of more than 85% of the maximum utility, remove them as
	 * well. The new offers have a utility value between 80% of the maximum
	 * utility offer to 90% of the minimal utility.
	 */
	public void updateBids(double ourW) throws Exception {
		// sort all the possible offers according to both utilities
		sortOffersByJointUtilities(_allReasonableOffers, ourW);
		_myOffers = randomlySelectOffers(ourW);
		_currentActionsIndex = 0;
	}

	/**
	 * Sort the offers in the array in random order.
	 * 
	 * @param array
	 *            - an array of offers
	 */
	public void shuffle(Offer[] array) {
		Random rng = new Random();

		// i is the number of items remaining to be shuffled.
		for (int i = array.length; i > 1; i--) {
			// Pick a random element to swap with the i-th element.
			int j = rng.nextInt(i); // 0 <= j <= i-1 (0-based array)

			// Swap array elements.
			Offer tmp = array[j];
			array[j] = array[i - 1];
			array[i - 1] = tmp;
		}
	}
}
