package agents.anac.y2013.TMFAgent;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.NegotiationResult;
import genius.core.SupportedNegotiationSetting;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.DefaultAction;
import genius.core.actions.EndNegotiation;
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
import genius.core.utility.EvaluatorInteger;
import genius.core.utility.EvaluatorReal;

public class TMFAgent extends Agent {

	private ArrayList<ComparableBid> relevantBids;
	private ArrayList<HashMap<Object, Double>> opponentUtilityEstimator;
	private HashMap<Bid, Integer> previousOpponentBids;
	private Bid lastBid;
	private ArrayList<Double> maxValue;
	private int lastPropose = 0;
	private Bid BestOpponentBid;
	private double firstOpponentBidUtility = 0;
	private double EstimatedRTT = 0;
	private double devRTT = 0;
	private double prevTime = 0;

	private boolean isFirstBid;

	@Override
	public void init() {
		isFirstBid = true;

		// init opponentUtilityEstimator
		Serializable s = loadSessionData();
		maxValue = new ArrayList<Double>();
		if (s != null) {
			opponentUtilityEstimator = (ArrayList<HashMap<Object, Double>>) s;
			int i = 0;
			for (HashMap<Object, Double> issue : opponentUtilityEstimator) {
				maxValue.add(0.0);
				for (double curr : issue.values()) {
					if (maxValue.get(i) < curr)
						maxValue.set(i, curr);
				}
				i++;
			}
		}

		else {
			opponentUtilityEstimator = new ArrayList<HashMap<Object, Double>>();

			List<Issue> issues = this.utilitySpace.getDomain().getIssues();
			for (Issue issue : issues) {
				int max_i = opponentUtilityEstimator.size();
				opponentUtilityEstimator.add(new HashMap<Object, Double>());
				if (issue.getType() == ISSUETYPE.DISCRETE) {
					for (ValueDiscrete vd : ((IssueDiscrete) issue)
							.getValues()) {
						opponentUtilityEstimator.get(max_i).put(vd, 0.0);
					}
				} else if (issue.getType() == ISSUETYPE.INTEGER) {
					int k = Math.min(10, ((IssueInteger) issue).getUpperBound()
							- ((IssueInteger) issue).getLowerBound());
					for (int i = 0; i <= k; i++) {
						opponentUtilityEstimator.get(max_i).put(i, 0.0);
					}
					/*
					 * 
					 * 
					 * 
					 * opponentUtilityEstimator.add(new HashMap<Value,
					 * Double>()); EvaluatorInteger ei = new EvaluatorInteger();
					 * 
					 * //putting the opponent best offer if
					 * (ei.getEvaluation(((IssueInteger)issue).getLowerBound())
					 * <
					 * ei.getEvaluation(((IssueInteger)issue).getUpperBound()))
					 * opponentUtilityEstimator.get(max_i).put(new
					 * ValueInteger(((IssueInteger)issue).getLowerBound()),
					 * 0.0); else opponentUtilityEstimator.get(max_i).put(new
					 * ValueInteger(((IssueInteger)issue).getUpperBound()),
					 * 0.0);
					 */
				} else if (issue.getType() == ISSUETYPE.REAL) {
					int k = 10;
					for (int i = 0; i < k; i++) {
						opponentUtilityEstimator.get(max_i).put(i + 0.0, 0.0);
					}

					/*
					 * opponentUtilityEstimator.add(new HashMap<Value,
					 * Double>()); opponentUtilityEstimator.get(max_i).put(new
					 * ValueReal(((IssueReal)issue).getLowerBound()), 0.0);
					 * opponentUtilityEstimator.get(max_i).put(new
					 * ValueReal(((IssueReal)issue).getUpperBound()), 0.0);
					 */
				}
				maxValue.add(0.0);
			}
		}
		// init relevantBids - sorted from highest utility bid to lowest.
		ArrayList<Bid> allBids = GetDiscreteBids();
		relevantBids = new ArrayList<ComparableBid>();
		for (Bid b : allBids) {
			try {
				relevantBids
						.add(new ComparableBid(b, utilitySpace.getUtility(b)));
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		Collections.sort(relevantBids);

		// init previousOpponentBids
		previousOpponentBids = new HashMap<Bid, Integer>();

		BestOpponentBid = null;
		lastBid = null;
		lastPropose = 0;
	}

	public void BeginSession(int sessionNumber) {
		isFirstBid = true;

	}

	@Override
	public void endSession(NegotiationResult res) {
		if (res.getLastAction() instanceof Accept)
			AddOpponentBidToModel(res.getLastBid(), true);

		saveSessionData(opponentUtilityEstimator);
	}

	@Override
	public String getVersion() {
		return "1.2";
	}

	@Override
	public String getName() {
		return "TMF-Agent";
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		// estimating time of one bid-response time (like Round Trip Time in
		// communication)
		double currRTT = timeline.getTime() - prevTime;
		prevTime = timeline.getTime();
		EstimatedRTT = 0.6 * EstimatedRTT + 0.4 * currRTT;
		devRTT = 0.75 * devRTT + 0.25 * Math.abs(currRTT - EstimatedRTT);

		Bid b = DefaultAction.getBidFromAction(opponentAction);
		if (b == null)
			return;
		isFirstBid = false;
		Integer i = previousOpponentBids.get(b);
		if (i == null) {
			i = 0;
			AddOpponentBidToModel(b, false);
		}
		previousOpponentBids.put(b, i + 1);

		if (lastBid == null) {
			// remove all unrelevance bids = all bids below the first opponent
			// bid or all bids below the reservation value.
			try {
				double minimumRelevant = Math.max(utilitySpace.getUtility(b),
						utilitySpace.getReservationValueUndiscounted());
				while (relevantBids.size() > 0 && (utilitySpace
						.getUtility(relevantBids.get(relevantBids.size()
								- 1).bid) < minimumRelevant))
					relevantBids.remove(relevantBids.size() - 1);
				BestOpponentBid = b;

				firstOpponentBidUtility = utilitySpace.getUtility(b);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		lastBid = b;
		try {
			if (utilitySpace.getUtility(BestOpponentBid) < utilitySpace
					.getUtility(b))
				BestOpponentBid = b;
		} catch (Exception e) {
		}
	}

	private void AddOpponentBidToModel(Bid b, boolean isAgreed) {
		// go over every issue of the bid
		for (int i = 0; i < utilitySpace.getDomain().getIssues().size(); i++) {
			// extract the value of that issue
			Object v = null;
			Issue issue = utilitySpace.getDomain().getIssues().get(i);
			Value v1 = null;
			try {
				v1 = b.getValue(issue.getNumber());
			} catch (Exception e) {
				e.printStackTrace();
				return;
			}

			if (issue.getType() == ISSUETYPE.DISCRETE) {
				v = v1;
			} else if (issue.getType() == ISSUETYPE.INTEGER) {
				// throw the value to the closest bucket
				int currValue = ((ValueInteger) v1).getValue();
				int k = Math.min(10, ((IssueInteger) issue).getUpperBound()
						- ((IssueInteger) issue).getLowerBound());
				int bucket = (int) Math.round((double) (currValue
						- ((IssueInteger) issue).getLowerBound())
						/ (((IssueInteger) issue).getUpperBound()
								- ((IssueInteger) issue).getLowerBound())
						* k);
				v = bucket; // not a real value - it just the "bucket name"
			} else if (issue.getType() == ISSUETYPE.REAL) {
				// throw the value to the closest bucket
				double currValue = ((ValueReal) v1).getValue();
				int k = 10;
				int bucket = (int) Math.round((currValue
						- ((IssueInteger) issue).getLowerBound())
						/ (((IssueInteger) issue).getUpperBound()
								- ((IssueInteger) issue).getLowerBound())
						* k);
				v = bucket + 0.0; // not a real value - it just the
									// "bucket name"
			} else
				return;

			// get the previous data and enter the new one (add 1-t)
			HashMap<Object, Double> hm = opponentUtilityEstimator.get(i);
			Double d = null;
			if (hm.containsKey(v))
				d = hm.get(v);
			else
				d = 0.0;
			double currData;
			if (isAgreed)
				currData = d + (maxValue.get(i) - d) / 2;
			else
				currData = d + 1.0 - timeline.getTime();
			opponentUtilityEstimator.get(i).put(v, currData);

			// set max value (for later normalization)
			if (currData > maxValue.get(i))
				maxValue.set(i, currData);

		}
	}

	@Override
	public Action chooseAction() {
		// a plan so timeout won't occur
		if (!HaveMoreTimeToBid(true)) { // if have time for one last bid, offer
										// the opponent Best possible bid for us
										// (or accept it, if the opponent just
										// offer it)
			try {
				if (utilitySpace.getUtility(BestOpponentBid) <= utilitySpace
						.getUtility(lastBid)) {
					if (utilitySpace.getUtility(lastBid) > utilitySpace
							.getReservationValueUndiscounted())
						return new Accept(getAgentID(), lastBid);
				} else if (utilitySpace
						.getUtility(BestOpponentBid) > utilitySpace
								.getReservationValueUndiscounted())
					return new Offer(getAgentID(), BestOpponentBid);
			} catch (Exception e) {
				e.printStackTrace();
			}
			return new EndNegotiation(getAgentID()); // if the opponent's best
														// offer is
			// worse than the reservation value,
			// end the negotiation.
		}
		if (!HaveMoreTimeToBid(false)) { // if have no more time for bidding (it
											// about to have time-out), accept
											// the opponent's offer, if it's
											// high than the reservation value
			try {
				if (utilitySpace.getUtility(lastBid) > utilitySpace
						.getReservationValueUndiscounted())
					return new Accept(getAgentID(), lastBid);
			} catch (Exception e) {
				e.printStackTrace();
			}
			return new EndNegotiation(getAgentID()); // else, take the
														// reservation value.

		}

		// if have time to negotiate properly..
		Action a = null;

		if (isFirstBid) {
			isFirstBid = false;
			try {
				return new Offer(getAgentID(), utilitySpace.getMaxUtilityBid());
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		double alpha = getHardness();
		ComparableBid cb = relevantBids.get(0); // get <best bid , it's
												// utility>;
		double best; // best bid's utility
		try {
			best = alpha * cb.utility
					+ (1 - alpha) * GetEstimatedOpponentUtility(cb.bid);
		} catch (Exception e1) {
			e1.printStackTrace();
			best = 0.0;
		}
		int j = Math.max(1, lastPropose - 3000); // for efficient reasons, don't
													// go over all possible bids
													// - only on 10,000 bids at
													// most (around our last
													// proposal)
		for (int i = j; i < relevantBids.size()
				&& i < lastPropose + 7000; i++) {
			double curr;
			ComparableBid currBid = relevantBids.get(i);

			try {
				curr = alpha * currBid.utility + (1 - alpha)
						* GetEstimatedOpponentUtility(currBid.bid);
			} catch (Exception e) {
				e.printStackTrace();
				curr = 0.0;
			}

			if (curr > best) {
				cb = currBid;
				best = curr;
			}

		}

		try {
			if (lastBid != null
					&& utilitySpace.getUtility(lastBid) > cb.utility)
				/*
				 * if the opponent offered better offer (to us) than what we
				 * decided to offer, we accept this offer.
				 */
				a = new Accept(getAgentID(), lastBid);
			else if (cb.utility < utilitySpace
					.getReservationValueUndiscounted())
				/*
				 * if our offer is worse than the reservation value we can
				 * achieve, we'll end the negotiation.
				 */
				a = new EndNegotiation(getAgentID());
			else
				a = new Offer(getAgentID(), cb.bid);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return a;
	}

	private ArrayList<Bid> GetDiscreteBids() {
		ArrayList<Bid> bids = new ArrayList<Bid>();
		HashMap<Integer, Value> issusesFirstValue = new HashMap<Integer, Value>();

		// initial bids list - contains only one bid.
		for (Issue issue : utilitySpace.getDomain().getIssues()) {
			Value v = null;
			if (issue.getType() == ISSUETYPE.INTEGER)
				v = new ValueInteger(((IssueInteger) issue).getLowerBound());
			else if (issue.getType() == ISSUETYPE.REAL)
				v = new ValueReal(((IssueReal) issue).getLowerBound());
			else if (issue.getType() == ISSUETYPE.DISCRETE)
				v = ((IssueDiscrete) issue).getValue(0);
			issusesFirstValue.put(issue.getNumber(), v);
		}
		try {
			bids.add(new Bid(utilitySpace.getDomain(), issusesFirstValue));
		} catch (Exception e) {
			return null;
		}

		for (Issue issue : utilitySpace.getDomain().getIssues()) { // for every
																	// issue
			ArrayList<Bid> tempBids = new ArrayList<Bid>(); // createFrom a list
															// of
															// bids
			ArrayList<Value> issueValues = new ArrayList<Value>();
			if (issue.getType() == ISSUETYPE.DISCRETE) {
				ArrayList<ValueDiscrete> valuesD = (ArrayList<ValueDiscrete>) ((IssueDiscrete) issue)
						.getValues(); // get
										// list
										// of
										// options/values
										// for
										// this
										// issue
				for (Value v : valuesD) {
					issueValues.add(v);
				}
			} else if (issue.getType() == ISSUETYPE.INTEGER) {
				int k = Math.min(10, ((IssueInteger) issue).getUpperBound()
						- ((IssueInteger) issue).getLowerBound());
				for (int i = 0; i <= k; i++) {
					ValueInteger vi = (ValueInteger) GetRepresentorOfBucket(i,
							issue, k, true);
					issueValues.add(vi);
				}
			} else if (issue.getType() == ISSUETYPE.REAL) {
				int k = 10;
				for (int i = 0; i <= k; i++) {
					ValueReal vr = (ValueReal) GetRepresentorOfBucket(i, issue,
							k, false);
					issueValues.add(vr);
				}
			}

			for (Bid bid : bids) { // for each bid seen so far (init bids list)
				for (Value value : issueValues) { // for every value
					HashMap<Integer, Value> bidValues = new HashMap<Integer, Value>(); // make
																						// new
																						// ("empty")
																						// bid
																						// -
																						// only
																						// values.
					for (Issue issue1 : utilitySpace.getDomain().getIssues())
						// go over all issues
						try {
							bidValues.put(issue1.getNumber(),
									bid.getValue(issue1.getNumber())); // each
																		// issue
																		// is
																		// entered
						} catch (Exception e) {
							e.printStackTrace();
						}
					bidValues.put(issue.getNumber(), value);
					try {
						Bid newBid = new Bid(utilitySpace.getDomain(),
								bidValues);
						tempBids.add(newBid);
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
			}
			bids = tempBids;
		}
		return bids;
	}

	private double GetEstimatedOpponentUtility(Bid b) {
		double d = 0.0;
		int count = 0;
		// go over all issues
		for (HashMap<Object, Double> h : opponentUtilityEstimator) {
			try {
				// add the normalize utility of that bid's value
				Issue issue = utilitySpace.getDomain().getIssues().get(count);
				int i = issue.getNumber(); // issue id
				if (issue.getType() == ISSUETYPE.DISCRETE)
					d += h.get(b.getValue(i)) / maxValue.get(count);
				else if (issue.getType() == ISSUETYPE.INTEGER) {
					Value v = b.getValue(i);
					ValueInteger vi = (ValueInteger) v;
					d += h.get(vi.getValue()) / maxValue.get(count);
				}

				else if (issue.getType() == ISSUETYPE.REAL) {
					Value v = b.getValue(i);
					ValueReal vr = (ValueReal) v;
					d += h.get(vr.getValue()) / maxValue.get(count);
				}
			} catch (Exception e) {
				return count == 0 ? d : d / count;
			}
			count++;
		}
		// make an average of all utilities (assuming weight is equal).
		return d / count;
	}

	private double getHardness() {
		double alpha = 0, x = timeline.getTime(),
				y = utilitySpace.getDiscountFactor() <= 0.0 ? 0.0
						: 1 - utilitySpace.getDiscountFactor();
		double weight = (1 - firstOpponentBidUtility) * 2 / 3;
		alpha = 1 - weight * Math.pow(x, 65) - (1 - weight) * Math.pow(y, 3);
		alpha = alpha / (x * y + 1);

		return alpha;
	}

	private boolean HaveMoreTimeToBid(boolean wantToMakeTwoProposals) {
		if (wantToMakeTwoProposals
				&& 1 - timeline.getTime() > 2 * EstimatedRTT + devRTT) {
			return true;
		}
		if (!wantToMakeTwoProposals
				&& 1 - timeline.getTime() > EstimatedRTT + devRTT) {
			return true;
		}
		return false;
	}

	private Value GetRepresentorOfBucket(int bucket, Issue issue, int k,
			boolean isInteger) {
		double ans = 0;

		if (isInteger) {
			EvaluatorInteger ei = new EvaluatorInteger();
			boolean upperIsTheBest = ei.getEvaluation(
					((IssueInteger) issue).getUpperBound()) > ei.getEvaluation(
							((IssueInteger) issue).getLowerBound());
			if (upperIsTheBest) {
				if (bucket < k) {
					ans = ((double) (bucket + 1)) / k;
					ans = ans
							* (((IssueInteger) issue).getUpperBound()
									- ((IssueInteger) issue).getLowerBound())
							+ ((IssueInteger) issue).getLowerBound() - 1;
				} else
					ans = ((IssueInteger) issue).getUpperBound();
			} else {
				ans = ((double) (bucket)) / k;
				ans = ans
						* (((IssueInteger) issue).getUpperBound()
								- ((IssueInteger) issue).getLowerBound())
						+ ((IssueInteger) issue).getLowerBound();
			}
			return new ValueInteger((int) Math.round(ans));
		}

		EvaluatorReal ei = new EvaluatorReal();
		boolean upperIsTheBest = ei
				.getEvaluation(((IssueReal) issue).getUpperBound()) > ei
						.getEvaluation(((IssueReal) issue).getLowerBound());
		if (upperIsTheBest) {
			if (bucket < k) {
				ans = ((double) (bucket + 1)) / k;
				ans = ans
						* (((IssueReal) issue).getUpperBound()
								- ((IssueReal) issue).getLowerBound())
						+ ((IssueReal) issue).getLowerBound();
			} else
				ans = ((IssueReal) issue).getUpperBound();
		} else {
			ans = ((double) (bucket)) / k;
			ans = ans
					* (((IssueReal) issue).getUpperBound()
							- ((IssueReal) issue).getLowerBound())
					+ ((IssueReal) issue).getLowerBound();
		}
		return new ValueReal(ans);

	}

	@Override
	public SupportedNegotiationSetting getSupportedNegotiationSetting() {
		return SupportedNegotiationSetting.getLinearUtilitySpaceInstance();
	}

	@Override
	public String getDescription() {
		return "ANAC2012";
	}

}
