package agents.anac.y2015.xianfa;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.issue.ISSUETYPE;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;

/**
 * This is your negotiation party.
 */
public class XianFaAgent extends AbstractNegotiationParty {

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

	Bid partnerBid = null;
	Bid myBid = null;
	int rounds = 0;
	double cRateA = 1;
	double cRateB = 1;
	double sRateA = 0;
	double sRateB = 0;
	String opponentA = "";
	String opponentB = "";
	double threshold = 0.99;
	double discount;
	int supposedNumber = 1;
	int totalA = 1;
	int totalB = 1;
	boolean firstRound = true;
	boolean startTiming = false;
	int timeRds = 0;
	double prevTime = 0;
	double avgRdTime = 0;
	double resValue;
	Bid opponentABest = null;
	int act = 0;
	int attitude = 0;

	/**
	 * fields for stats tracking
	 */
	boolean debug = false;
	Statistician stat;
	int stat_bidsInList = -1;
	int stat_bidsNotInList = 0;
	int stat_goodBids = 0;
	int stat_BAccepts = 0;
	int stat_offerHistBest = 0;
	double stat_avgTime = 0;
	double stat_thatTime = 0;
	boolean myTurnPrev = false;
	ArrayList<Bid> myOfferedBids = new ArrayList<Bid>();
	int myUniqueOffers = 1;
	double aU = 1;
	double oU = 1;

	/**
	 * 
	 * @author Kevin
	 *
	 */

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		// if (utilitySpace.getDomain().getNumberOfPossibleBids() > 200000)
		// attitude = 2;
		// else attitude = 0;
		try {
			initTree();
			calculate();
			initBS();
			initOpp();
			if (debug)
				stat = new Statistician(this);
		} catch (Exception e) {
			e.printStackTrace();
		}
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
	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		updateSelf();
		if (debug) {
			if (rounds % 2 == 0)
				stat.log();
		}
		Action action = null;
		if (timeline.getTime() > 0.9) {
			if (startTiming == false) {
				startTiming = true;
				prevTime = timeline.getTime();
				timeRds++;
			} else {
				avgRdTime = (avgRdTime * (timeRds - 1))
						+ (timeline.getTime() - prevTime);
				avgRdTime = avgRdTime / timeRds;
				prevTime = timeline.getTime();
				timeRds++;
			}
		}
		try {
			if (firstRound)
				action = chooseBid(null);

			else {
				if (acceptable()) {
					if (debug)
						stat.log();
					if (aU > utilitySpace.getUtility(partnerBid))
						aU = utilitySpace.getUtility(partnerBid);
					act = 1;
					action = new Accept(getPartyId(), partnerBid);
				} else if (timeline.getTime() > (1 - (avgRdTime * 1.2))) {
					if (opponentABest != null) {
						if (utilitySpace.getUtility(opponentABest) > resValue) {
							if (utilitySpace.getUtility(opponentABest) > (1.03
									* utilitySpace.getUtility(partnerBid))) {
								stat_offerHistBest++;
								if (oU > utilitySpace.getUtility(opponentABest))
									oU = utilitySpace.getUtility(opponentABest);
								action = new Offer(getPartyId(), opponentABest);
							} else {
								act = 1;
								action = new Accept(getPartyId(), partnerBid);
							}
						} else {
							action = chooseBid(partnerBid);
						}
					} else {
						action = chooseBid(partnerBid);
					}
				} else {
					action = chooseBid(partnerBid);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("catch chooseaction");
			System.out.println("Exception in ChooseAction:" + e.getMessage());
			act = 1;
			action = new Accept(getPartyId(), partnerBid); // best guess if
															// things go wrong.
		}
		if (timeline.getTime() == 1) {
			act = 3;
			action = new EndNegotiation(getPartyId());
		}
		if (act == 1) {
			// System.out.println("Current action: Accept");
		} else if (act == 2) {
			partnerBid = new Bid(myBid);
			// System.out.println("Current action: Offer");
		} else if (act == 3) {
			// System.out.println("Current action: End Negotiation");
		} else {
			// System.out.println("Current action: Rubbish");
		}
		return action;
	}

	/**
	 * 
	 * @param partnerBid
	 *            the bid the partner placed
	 * @return next action
	 */
	private Action chooseBid(Bid partnerBid) {
		Bid nextBid = null;
		try {
			nextBid = new Bid(getBid());
		} catch (Exception e) {
			System.out.println("Problem with received bid:" + e.getMessage()
					+ ". cancelling bidding");
		}
		if (nextBid == null) {
			act = 1;
			return (new Accept(getPartyId(), partnerBid));
		}
		try {
			if (partnerBid != null) {
				if (utilitySpace.getUtility(partnerBid) >= utilitySpace
						.getUtility(nextBid)) {
					act = 1;
					return (new Accept(getPartyId(), partnerBid));
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
			act = 1;
			// best guess if things go wrong.
			return (new Accept(getPartyId(), partnerBid));
		}
		try {
			if (oU > utilitySpace.getUtility(nextBid))
				oU = utilitySpace.getUtility(nextBid);
		} catch (Exception e) {
			e.printStackTrace();
		}

		act = 2;
		myBid = new Bid(nextBid);
		return (new Offer(getPartyId(), nextBid));
	}

	private Bid getBid() throws Exception {
		Bid bid = null;
		if (attitude == 0) {
			bid = getOb();
			return bid;
		} else if (attitude == 1) {
			bid = offerBid();
			return bid;
		}

		return bid;
	}

	private void updateSelf() {
		myTurnPrev = true;
		rounds++;
		if (!opponentA.equals("") && !opponentB.equals("")) {
			// conceding Rate : 1 - Conceder; 0 - Boulware
			cRateA = 1 - (Math.min(0.95,
					((double) bidSet.size() / totalA) * 1000));
			cRateB = 1 - (Math.min(0.95,
					((double) bidSetB.size() / totalB) * 1000));
			// similarity Rate : 1 - Weak opposition; 0 - Strong opposition
			sRateA = findHighestSimilarity(opponentA);
			sRateB = findHighestSimilarity(opponentB);
			if (rounds % 200 == 0) {
				double t = timeline.getTime();
				threshold = threshold - ((Math.max(cRateA, cRateB) * 0.00515)
						* ((1 - t) * (1 - t) * (1 / discount)));
				threshold = threshold
						- ((Math.max(1 - sRateA, 1 - sRateB) * 0.0116)
								* ((1 - t) * (1 - t) * (1 / discount)));
			}
			// printConsole();
		}
		if (attitude == 2)
			return;
		if (attitude == 1)
			return;
		if (rounds > 2000 || timeline.getTime() > 0.1) {
			attitude = 1;
			return;
		}
	}

	public void printConsole() {
		if (rounds % 2 == 0) {
			System.out.println("This is round: " + rounds);
			System.out.println("Time is: " + timeline.getTime());
			System.out.println("Short List size: " + shortList.size());
			for (int i = 0; i < shortList.size(); i++) {
				try {
					System.out
							.println(utilitySpace.getUtility(shortList.get(i)));
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			try {
				System.out.println("utility of partnerBid: "
						+ utilitySpace.getUtility(partnerBid));
			} catch (Exception e) {
				e.printStackTrace();
			}
			System.out.println("Reservation value is: " + resValue);
			System.out.println("Discount factor is: " + discount);
			System.out.println("c rate A: " + cRateA);
			System.out.println("c rate B: " + cRateB);
			System.out.println("s rate A: " + sRateA);
			System.out.println("s rate B: " + sRateB);
			System.out.println("threshold: " + threshold);
			System.out.println("My lowest accepted utility: " + aU);
			System.out.println("My lowest offered utility: " + oU);
			System.out.println("");
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

		// Assign party ids to the two opponents
		if (sender != null && firstRound) {
			if (!(opponentA.equals("") || opponentB.equals("")))
				firstRound = false;
			if (!myTurnPrev) {
				if (!opponentA.equals("")) {
					opponentB = sender.toString();
				}
			} else
				opponentA = sender.toString();
		}

		if (action instanceof Offer) {
			partnerBid = new Bid(((Offer) action).getBid());
			try {
				updateOpp(partnerBid, sender.toString(), 0);
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else if (action instanceof Accept) {
			if (partnerBid != null) {
				try {
					updateOpp(partnerBid, sender.toString(), 1);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			if (myTurnPrev) {
				stat_goodBids++;
			} else {
				try {
					if (opponentABest == null)
						opponentABest = new Bid(partnerBid);
					else if (utilitySpace
							.getUtility(opponentABest) < getUtilitySpace()
									.getUtility(partnerBid)) {
						opponentABest = new Bid(partnerBid);
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
				stat_BAccepts++;
			}
		}
		if (myTurnPrev)
			myTurnPrev = false;
	}

	double mean = 0;
	double median = 0;
	double searchThreshold = 0.82;

	public void calculate() throws Exception {
		discount = utilitySpace.getDiscountFactor();
		double add = 0.1;
		double r = utilitySpace.getReservationValueUndiscounted();
		if (r < 0.65)
			add = 0.15;
		resValue = Math.max(r, 0.5) + ((r / 0.9) * add);
		if (resValue > 0.99) {
			resValue = 0.99;
		}
		int size = allBidsList.size();
		quickSort(allBidsList, 0, size - 1);
		System.out.println("All bids List size is " + size);
	}

	public double findHighestSimilarity(String opponent) {
		Bid bid = generateRandomBid();
		if (opponent.equals(opponentA)) {
			for (AIssue iss : issuesA) {
				Value val = iss.getDesiredVal();
				bid = bid.putValue(iss.getIssNr(), val);
			}
		} else if (opponent.equals(opponentB)) {
			for (AIssue iss : issuesB) {
				Value val = iss.getDesiredVal();
				bid = bid.putValue(iss.getIssNr(), val);
			}
		}
		try {
			return utilitySpace.getUtility(bid);
		} catch (Exception e) {
			e.printStackTrace();
		}
		// System.out.println("error with find Highest similarity");
		return 0;
	}

	public void quickSort(ArrayList<Bid> list, int n, int m) throws Exception {
		int pivot_pos;
		if (n >= m) {
			return;
		}
		pivot_pos = partition(list, n, m);
		quickSort(list, n, pivot_pos - 1);
		quickSort(list, pivot_pos + 1, m);
	}

	public int partition(ArrayList<Bid> list, int low, int high)
			throws Exception {
		int last_small;
		double pivot;
		int mid = (low + high) / 2;
		Collections.swap(list, low, mid);
		pivot = utilitySpace.getUtility(list.get(low));
		last_small = low;
		for (int i = low + 1; i <= high; i++) {
			if (utilitySpace.getUtility(list.get(i)) < pivot) {
				Collections.swap(list, ++last_small, i);
			}
		}
		Collections.swap(list, low, last_small);
		return last_small;
	}

	/**
	 * bidding strategy
	 */

	Tree tree;
	private int maxDepth;
	private EvaluatorDiscrete ed;
	private IssueDiscrete id;
	private IssueInteger ii;
	ArrayList<Bid> allBidsList = new ArrayList<Bid>();
	ArrayList<Bid> shortList = new ArrayList<Bid>();
	Bid startingBid;

	public void initTree() throws Exception {
		this.tree = new Tree();
		this.maxDepth = utilitySpace.getDomain().getIssues().size();
		tree.addNewDepth();

		for (int curDepth = 0; curDepth < maxDepth; curDepth++) { // for this
																	// depth
			tree.addNewDepth(); // initialize new level
			if (utilitySpace.getDomain().getIssues().get(curDepth)
					.getType() == ISSUETYPE.DISCRETE) {
				id = (IssueDiscrete) utilitySpace.getDomain().getIssues()
						.get(curDepth);
				ed = (EvaluatorDiscrete) ((AdditiveUtilitySpace) utilitySpace)
						.getEvaluator(curDepth + 1);
				for (int curMember = 0; curMember < tree
						.getSizeOfLevel(curDepth); curMember++) { // for
																	// all
																	// nodes
																	// in
																	// this
																	// depth
					for (int i = 0; i < id.getNumberOfValues(); i++) { // for
																		// all
																		// evaluations
																		// for
																		// next
																		// issue
																		// below
																		// this
																		// depth
						Node node = new Node(id.getNumber(), id.getValue(i),
								curDepth, ed.getEvaluation(id.getValue(i)));
						tree.getMemberInLevel(curDepth, curMember).add(node);
						tree.addNodeInDepth(node, curDepth + 1);
					}
				}
			} else if (utilitySpace.getDomain().getIssues().get(curDepth)
					.getType() == ISSUETYPE.INTEGER) {
				ii = (IssueInteger) utilitySpace.getDomain().getIssues()
						.get(curDepth);
				for (int curMember = 0; curMember < tree
						.getSizeOfLevel(curDepth); curMember++) { // for
																	// all
																	// nodes
																	// in
																	// this
																	// depth
					Node node = new Node(ii.getNumber(),
							new ValueInteger(ii.getUpperBound()), curDepth, 1);
					tree.getMemberInLevel(curDepth, curMember).add(node);
					tree.addNodeInDepth(node, curDepth + 1);
				}
			}
		}
		for (int i = 0; i < tree.getSizeOfLevel(tree.getLevels() - 1); i++) {
			Bid bid = createBid(getBid(i));
			allBidsList.add(bid);
		}
	}

	public void initBS() throws Exception {
		createLists();
		for (int i = 0; i < tree.getLevels(); i++) {
			for (int j = 0; j < tree.getSizeOfLevel(i); j++) {
				Node node = tree.getMemberInLevel(i, j);
				node = null;
			}
		}
		tree = null;
	}

	private void createLists() throws Exception {
		for (int i = allBidsList.size() - 1; i > -1; i--) {
			if (utilitySpace.getUtility(allBidsList.get(i)) > resValue) {
				shortList.add(new Bid(allBidsList.get(i)));
			} else
				break;
		}
		startingBid = new Bid(shortList.get(0));
		myOfferedBids.add(startingBid);
		System.out.println("utility of starting bid is: "
				+ utilitySpace.getUtility(startingBid));
		// quickSort(shortList, 0, shortList.size()-1);
	}

	public Bid getOb() {
		return startingBid;
	}

	public Bid offerBid() throws Exception {
		if (rounds > (3000 * supposedNumber)
				|| timeline.getTime() > (0.124 * supposedNumber)) {
			supposedNumber++;
		}
		if (myOfferedBids.size() < supposedNumber) {
			String opponent;
			String other;
			int pos = 0;
			double otherMax = 0;
			if (cRateA > cRateB) {
				opponent = opponentA;
				other = opponentB;
			} else {
				opponent = opponentB;
				other = opponentA;
			}
			for (int i = 0; i < shortList.size(); i++) {
				if (utilitySpace.getUtility(shortList.get(i)) > threshold
						&& !myOfferedBids.contains(shortList.get(i))) {
					if (favor(shortList.get(i), opponent) > 0.8) {
						if (favor(shortList.get(i), other) > 0.8) {
							myOfferedBids.add(shortList.get(i));
							return shortList.get(i);
						} else if (favor(shortList.get(i), other) > otherMax) {
							otherMax = favor(shortList.get(i), other);
							pos = i;
						}
					}
				}
			}
			return shortList.get(pos);
		}
		return myOfferedBids.get(new Random().nextInt(myOfferedBids.size()));
	}

	public Bid createBid(Bid bid) {
		Bid newBid = new Bid(bid);
		return newBid;
	}

	public Bid getBid(int position) throws Exception {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>();
		maxDepth = tree.getLevels() - 1;
		values.put(tree.getMemberInLevel(maxDepth, position).getNumber(),
				tree.getMemberInLevel(maxDepth, position).getValue());
		Node node = tree.getMemberInLevel(maxDepth, position).getParent();
		while (maxDepth != 0) {
			values.put(node.getNumber(), node.getValue());
			node = node.getParent();
			maxDepth = maxDepth - 1;
		}
		Bid bid = new Bid(utilitySpace.getDomain(), values);
		return bid;
	}

	public ArrayList<Bid> insertionSort(ArrayList<Bid> list, int n)
			throws Exception {
		for (int i = 1; i < n; i++) {
			for (int j = i; (j > 0); j--) {
				// System.out.println("j is: " + j);
				if (utilitySpace.getUtility(list.get(j)) > utilitySpace
						.getUtility(list.get(j - 1))) {
					Collections.swap(list, j, j - 1);
				} else
					break;
			}
		}
		return list;
	}

	public double favor(Bid bid, String opp) throws Exception {
		double score = 0;
		if (opp.equals(opponentA)) {
			for (Issue iss : bid.getIssues()) {
				for (int i = 0; i < issuesA.size(); i++) {
					if (issuesA.get(i).getIssNr() == iss.getNumber()) {
						double s = issuesA.get(i)
								.getVal(bid.getValue(iss.getNumber()));
						if (s > 1)
							s = 1;
						score = score + s;
						break;
					}
				}
			}
			score = score / (utilitySpace.getDomain().getIssues().size());
		} else if (opp.equals(opponentB)) {
			for (Issue iss : bid.getIssues()) {
				for (int i = 0; i < issuesB.size(); i++) {
					if (issuesB.get(i).getIssNr() == iss.getNumber()) {
						double s = issuesB.get(i)
								.getVal(bid.getValue(iss.getNumber()));
						if (s > 1)
							s = 1;
						score = score + s;
						break;
					}
				}
			}
			score = score / (utilitySpace.getDomain().getIssues().size());
		}
		return score;
	}

	/**
	 * opponent model
	 */

	ArrayList<Bid> bidSet = new ArrayList<Bid>();
	ArrayList<AIssue> issuesA = new ArrayList<AIssue>();
	ArrayList<Bid> bidSetB = new ArrayList<Bid>();
	ArrayList<AIssue> issuesB = new ArrayList<AIssue>();
	int totalBids = 0;

	public void initOpp() {
		for (int i = 0; i < utilitySpace.getDomain().getIssues().size(); i++) {
			AIssue iss = new AIssue(
					utilitySpace.getDomain().getIssues().get(i).getNumber());
			if (utilitySpace.getDomain().getIssues().get(i)
					.getType() == ISSUETYPE.DISCRETE) {
				IssueDiscrete id = (IssueDiscrete) utilitySpace.getDomain()
						.getIssues().get(i);
				for (int j = 0; j < id.getValues().size(); j++) {
					iss.getValues().put(id.getValue(j), 0.0);
				}
			} else if (utilitySpace.getDomain().getIssues().get(i)
					.getType() == ISSUETYPE.INTEGER) {
				IssueInteger ii = (IssueInteger) utilitySpace.getDomain()
						.getIssues().get(i);
				iss.getValues().put(new ValueInteger(ii.getUpperBound()), 0.0);
			}
			issuesA.add(iss);
		}
		for (int i = 0; i < utilitySpace.getDomain().getIssues().size(); i++) {
			AIssue iss = new AIssue(
					utilitySpace.getDomain().getIssues().get(i).getNumber());
			if (utilitySpace.getDomain().getIssues().get(i)
					.getType() == ISSUETYPE.DISCRETE) {
				IssueDiscrete id = (IssueDiscrete) utilitySpace.getDomain()
						.getIssues().get(i);
				for (int j = 0; j < id.getValues().size(); j++) {
					iss.getValues().put(id.getValue(j), 0.0);
				}
			} else if (utilitySpace.getDomain().getIssues().get(i)
					.getType() == ISSUETYPE.INTEGER) {
				IssueInteger ii = (IssueInteger) utilitySpace.getDomain()
						.getIssues().get(i);
				iss.getValues().put(new ValueInteger(ii.getUpperBound()), 0.0);
			}
			issuesB.add(iss);
		}
	}

	public void updateOpp(Bid bid, String sender, int act) throws Exception {
		boolean newVariation = true;
		if (sender.equals(opponentA)) {
			if (act == 0) {
				totalA++;
				for (int i = 0; i < bidSet.size(); i++) {
					if (bidSet.get(i).equals(bid)) {
						newVariation = false;
						break;
					}
				}
				if (newVariation == true) {
					bidSet.add(bid);
				}
				for (int i = 0; i < bid.getIssues().size(); i++) {
					for (int j = 0; j < issuesA.size(); j++) {
						if (bid.getIssues().get(i).getNumber() == issuesA.get(j)
								.getIssNr()) {
							if (bid.getIssues().get(i)
									.getType() == ISSUETYPE.DISCRETE) {
								issuesA.get(j)
										.setVal(bid.getValue(bid.getIssues()
												.get(i).getNumber()), totalA,
												0);
							} else {
								// ignore
							}
						}
					}
				}
			} else {
				for (int i = 0; i < bid.getIssues().size(); i++) {
					for (int j = 0; j < issuesA.size(); j++) {
						if (bid.getIssues().get(i).getNumber() == issuesA.get(j)
								.getIssNr()) {
							if (bid.getIssues().get(i)
									.getType() == ISSUETYPE.DISCRETE) {
								issuesA.get(j)
										.setVal(bid.getValue(bid.getIssues()
												.get(i).getNumber()), totalA,
												1);
							} else {
								// ignore
							}
						}
					}
				}
			}

		} else if (sender.equals(opponentB)) {
			if (act == 0) {
				totalB++;
				for (int i = 0; i < bidSetB.size(); i++) {
					if (bidSetB.get(i).equals(bid)) {
						newVariation = false;
						break;
					}
				}
				if (newVariation == true) {
					bidSetB.add(bid);
				}
				for (int i = 0; i < bid.getIssues().size(); i++) {
					for (int j = 0; j < issuesB.size(); j++) {
						if (bid.getIssues().get(i).getNumber() == issuesB.get(j)
								.getIssNr()) {
							if (bid.getIssues().get(i)
									.getType() == ISSUETYPE.DISCRETE) {
								issuesB.get(j)
										.setVal(bid.getValue(bid.getIssues()
												.get(i).getNumber()), totalB,
												0);
							} else {
								// ignore
							}
						}
					}
				}
			} else {
				for (int i = 0; i < bid.getIssues().size(); i++) {
					for (int j = 0; j < issuesB.size(); j++) {
						if (bid.getIssues().get(i).getNumber() == issuesB.get(j)
								.getIssNr()) {
							if (bid.getIssues().get(i)
									.getType() == ISSUETYPE.DISCRETE) {
								issuesB.get(j)
										.setVal(bid.getValue(bid.getIssues()
												.get(i).getNumber()), totalB,
												1);
							} else {
								// ignore
							}
						}
					}
				}
			}
		}
	}

	/**
	 * acceptance criteria
	 */

	public boolean acceptable() throws Exception {
		double myUtility = utilitySpace.getUtility(partnerBid);
		if ((myUtility > threshold) && (myUtility > resValue)) {
			return true;
		} else
			return false;
	}

	@Override
	public String getDescription() {
		return "ANAC2015";
	}

}