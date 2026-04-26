package agents.anac.y2013.MetaAgent;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.Domain;
import genius.core.NegotiationResult;
import genius.core.SupportedNegotiationSetting;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.issue.ISSUETYPE;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Objective;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EVALUATORTYPE;
import genius.core.utility.Evaluator;
import genius.core.utility.EvaluatorDiscrete;
import genius.core.utility.EvaluatorInteger;
import genius.core.utility.EvaluatorReal;

public class MetaAgent2013 extends Agent {
	private Agent myAgent;
	private boolean agentInit;
	private Action actionOfPartner;
	HashMap<String, Double> features;
	double opponentFirstBid, slopeBBUA, absSlopeBBUA, avgUtilBBUA;
	boolean isUpdated;
	AgentManager manager = new AgentManager();

	@Override
	@SuppressWarnings("deprecation")
	public void init() {
		agentInit = false;
		opponentFirstBid = 0;
		slopeBBUA = 0;
		absSlopeBBUA = 0;
		avgUtilBBUA = 0;
		Serializable s = loadSessionData();
		if (s != null) {
			manager = (AgentManager) s;
			myAgent = manager.SelectBestAgent();
			myAgent.internalInit(this.sessionNr, this.sessionsTotal,
					this.startTime, this.totalTime, this.timeline,
					this.utilitySpace, this.parametervalues, this.getAgentID());
			myAgent.setName(this.getName());
			myAgent.init();
			agentInit = true;
		}
		isUpdated = false;
	}

	@Override
	public void endSession(NegotiationResult res) {
		if (!isUpdated) {
			manager.UpdateUtility("", res.getMyDiscountedUtility());
			isUpdated = true;
		}
		saveSessionData(manager);
	}

	private void getDomainParams() {
		Domain d = this.utilitySpace.getDomain();
		List<Issue> a = d.getIssues();

		// params = new double[14];
		// //(Intercept),firstBBUA,slopeBBUA,absSlopeBBUA,avgUtilBBUA,DiscountFactor,DomainSize,numOfIssues,MinSize,MaxSize,AvgUtil,AvgUtilStdev,WeightStdev,Begins
		features = new HashMap<String, Double>();

		int min = Integer.MAX_VALUE, max = -1, size = 1, sumSize = 0;// ,
																		// relCount
																		// = 0;

		// params 5+6+7 - Expected Utility of role, Standard deviation of
		// Utility of role, Standard deviation of weights of role
		double EU = 0, stdevU = 0, stdevW = 0, sW = 0, ssW = 0, countW = 0,
				relevantEU = 0;
		Iterator<Entry<Objective, Evaluator>> issue = ((AdditiveUtilitySpace) utilitySpace)
				.getEvaluators().iterator();
		List<Double> ssWList = new ArrayList<Double>();
		while (issue.hasNext()) { // every issue
			Entry<Objective, Evaluator> entry = issue.next();
			Evaluator e = entry.getValue();
			double weight = e.getWeight();
			countW++;
			sW += weight;
			ssWList.add(weight);
			double tempEU = 0, tempStdevU = 0;
			if (e.getType() == EVALUATORTYPE.DISCRETE) {
				Iterator<ValueDiscrete> v = ((EvaluatorDiscrete) e).getValues()
						.iterator();
				List<Double> s = new ArrayList<Double>();
				double sumU = 0;
				while (v.hasNext()) {
					ValueDiscrete vd = v.next();
					try {
						double val = ((EvaluatorDiscrete) e).getEvaluation(vd);
						s.add(val);
						sumU += val;
						if (opponentFirstBid > 0 && opponentFirstBid <= val) {
							// relCount ++;
							relevantEU += val;
						}
					} catch (Exception e1) {
						// System.out.println("META-Agent IO exception: " +
						// e1.toString());
					}
				}
				int currSize = s.size();
				min = (min > currSize) ? currSize : min;
				max = (max < currSize) ? currSize : max;
				sumSize += currSize;
				size = size * currSize;
				tempEU = sumU / currSize;
				Iterator<Double> valIt = s.iterator();
				while (valIt.hasNext()) {
					tempStdevU += Math.pow(valIt.next() - tempEU, 2);
				}
				tempStdevU = Math.sqrt(tempStdevU / ((double) currSize - 1));
			} else if (e.getType() == EVALUATORTYPE.INTEGER) {
				tempEU = ((double) (((EvaluatorInteger) e).getUpperBound()
						+ ((EvaluatorInteger) e).getLowerBound())) / 2;
				tempStdevU = Math.sqrt((Math
						.pow(((EvaluatorInteger) e).getUpperBound() - tempEU, 2)
						+ Math.pow(
								((EvaluatorInteger) e).getLowerBound() - tempEU,
								2))
						/ 2);
			} else if (e.getType() == EVALUATORTYPE.REAL) {
				tempEU = (((EvaluatorReal) e).getUpperBound()
						+ ((EvaluatorReal) e).getLowerBound()) / 2;
				tempStdevU = Math.sqrt((Math
						.pow(((EvaluatorReal) e).getUpperBound() - tempEU, 2)
						+ Math.pow(((EvaluatorReal) e).getLowerBound() - tempEU,
								2))
						/ 2);
			} else {
				tempEU = 0.5;
				tempStdevU = 0;
			}

			EU += tempEU * weight;
			stdevU += tempStdevU * weight;
		}
		Iterator<Double> wIt = ssWList.iterator();
		double avgW = sW / countW;
		double avgSize = ((double) sumSize)
				/ ((double) ((AdditiveUtilitySpace) utilitySpace)
						.getEvaluators().size());

		while (wIt.hasNext()) {
			ssW += Math.pow(wIt.next() - avgW, 2);
		}
		stdevW = countW <= 1 ? 0 : Math.sqrt(ssW / (countW - 1));

		double relsumUtility = 0, relcountUtility = 0, relstdevUtility = 0,
				stdevUtil = 0, countbids = 0, sumbids = 0, relevantStdevU = 0;
		ArrayList<Double> bidsUtil = new ArrayList<Double>();
		ArrayList<Bid> bids = GetDiscreteBids();
		for (Bid bid : bids) {
			try {
				bidsUtil.add(utilitySpace.getUtility(bid));
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		for (double bidUtil : bidsUtil) {
			stdevUtil += Math.pow(bidUtil, 2);
			sumbids += bidUtil;
			countbids++;
			if (bidUtil >= opponentFirstBid) {
				relsumUtility += bidUtil;
				relcountUtility++;
				relstdevUtility += Math.pow(bidUtil, 2);
			}
		}
		if (relcountUtility > 0) {
			relevantEU = relsumUtility / relcountUtility;
			relevantStdevU = Math.sqrt((relstdevUtility / relcountUtility)
					- Math.pow(relevantEU, 2));
		}
		if (countbids > 0) {
			EU = sumbids / countbids;
			stdevU = Math.sqrt((stdevUtil / countbids) - Math.pow(EU, 2));
		}

		features.put("(Intercept)", 1.0);
		features.put("UtilityOfFirstOpponentBid", opponentFirstBid); // firstBBUA
		features.put("ReservationValue",
				this.utilitySpace.getReservationValue()); // RV
		features.put("DiscountFactor",
				this.utilitySpace.getDiscountFactor() == 0 ? 1
						: this.utilitySpace.getDiscountFactor()); // DiscountFactor
		features.put("numOfIssues", a.size() + 0.0);
		features.put("DomainSize", size + 0.0);
		features.put("AvgUtil", EU);
		features.put("AvgUtilStdev", stdevU);
		features.put("AvgSize", avgSize);
		features.put("WeightStdev", stdevW);
		features.put("RelevantEU", relevantEU);
		features.put("RelevantStdevU", relevantStdevU);

		// params[13] = first ? 1 : 0;//Begins

	}

	@Override
	@SuppressWarnings("deprecation")
	public Action chooseAction() {
		try {
			if (!agentInit) {
				if (actionOfPartner == null) // first bid, and Meta-Agent is the
												// first bidder (still not
												// initialized any agent)
				{
					// System.out.println("actionOfPartner == null.");
					return new Offer(this.getAgentID(),
							utilitySpace.getMaxUtilityBid());
				} else // second bid, or first bid as responder (and not
						// initiator)
				{
					// System.out.println("actionOfPartner not null.");
					agentInit = true;
					if (actionOfPartner instanceof Offer) {
						// System.out.println("actionOfPartner is Offer.");
						opponentFirstBid = utilitySpace
								.getUtility(((Offer) actionOfPartner).getBid());
					}

					// initialize agent.
					getDomainParams();
					UpdateAllAgents();
					manager.SetAvgUtil(
							features.get("RelevantEU") * Math
									.pow(utilitySpace.getDiscountFactor(), 0.5),
							features.get("RelevantStdevU"));
					myAgent = manager.SelectBestAgent();
					myAgent.internalInit(this.sessionNr, this.sessionsTotal,
							this.startTime, this.totalTime, this.timeline,
							this.utilitySpace, this.parametervalues,
							this.getAgentID());
					myAgent.setName(this.getName());
					myAgent.init();
					// send the message that received to the new agent..
					myAgent.ReceiveMessage(actionOfPartner);
				}
			}
			Action a = myAgent.chooseAction();
			if (a == null) {
				return new Offer(this.getAgentID(),
						utilitySpace.getMaxUtilityBid());
			} else { // handle reservation value, because all agent are from
						// Genius version 3.1 (and don't know what is
						// reservation value)
						// System.out.println("Action a is not null.");
				double time = timeline.getTime();
				if (a instanceof Offer
						&& this.utilitySpace.getReservationValueWithDiscount(
								time) >= utilitySpace.getUtilityWithDiscount(
										((Offer) a).getBid(), time)) {
					a = new EndNegotiation(getAgentID());
				}

			}
			return a;

		} catch (Exception e) {
			// System.out.println("META-AGENT error in ChooseAction: " +
			// e.toString());
			return null;
		}
	}

	private void UpdateAllAgents() {
		Set<String> agents = manager.GetAgents();
		for (String agentName : agents) {
			double predicted = Parser.getMean(manager.GetAgentData(agentName),
					features);
			manager.UpdateUtility(agentName, predicted);
		}

	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
		if (agentInit)
			myAgent.ReceiveMessage(opponentAction);
	}

	@Override
	public String getName() {
		return "Meta-Agent 2013";
	}

	@Override
	public String getVersion() {
		return "2.0";
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
		return "ANAC2013";
	}
}
