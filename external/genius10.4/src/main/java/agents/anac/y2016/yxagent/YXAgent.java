package agents.anac.y2016.yxagent;

import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.UtilitySpace;

public class YXAgent extends AbstractNegotiationParty {
	private UtilitySpace utilitySpace;
	private AdditiveUtilitySpace additiveUtilitySpace;
	private List<Issue> issues;
	private Action myAction = null;
	private Bid myLastBid = null;
	private Bid lastOpponentBid = null;
	private double myUtility = 0.;
	private double oppUtility = 0.;
	private double rv;
	private double discountFactor;
	private int rounds;
	private boolean updatedValueIntegerWeight;
	private boolean issueContainIntegerType;
	private boolean searchedDiscountWithRV;

	private ArrayList<Object> opponents;
	private ArrayList<Bid> allBids;
	private HashMap<Object, ArrayList<Bid>> oppBidHistory;
	private HashMap<Object, HashMap<Issue, Double>> oppIssueWeight;
	private HashMap<Object, HashMap<Issue, ArrayList<ValueInteger>>> oppIssueIntegerValue;
	private HashMap<Object, HashMap<Issue, HashMap<Value, Double>>> oppValueFrequency;
	private HashMap<Object, Double> oppSumDiff;
	private Object hardestOpp;

	private long startTime;
	private long currTime;
	private long diff;
	private double totalTime;
	private long normalizedTime;

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		this.utilitySpace = getUtilitySpace();
		this.rv = utilitySpace.getReservationValue();
		discountFactor = getUtilitySpace().getDiscountFactor();
		issues = utilitySpace.getDomain().getIssues();
		allBids = new ArrayList<Bid>();
		opponents = new ArrayList<Object>();
		oppBidHistory = new HashMap<Object, ArrayList<Bid>>();
		oppIssueWeight = new HashMap<Object, HashMap<Issue, Double>>();
		oppIssueIntegerValue = new HashMap<Object, HashMap<Issue, ArrayList<ValueInteger>>>();
		oppValueFrequency = new HashMap<Object, HashMap<Issue, HashMap<Value, Double>>>();
		oppSumDiff = new HashMap<Object, Double>();
		rounds = 0;
		updatedValueIntegerWeight = false;
		issueContainIntegerType = false;
		searchedDiscountWithRV = false;
		initTime();
		issueContainIntegerType();
	}

	@Override
	public void receiveMessage(AgentID sender, Action opponentAction) {
		super.receiveMessage(sender, opponentAction);

		// initialize below
		if (sender != null) {// sender == null is for Inform messages
			if (!opponents.contains(sender)) {
				opponents.add(sender);
				initOpp(sender);
			}
		}

		if (sender != null && opponentAction != null) {
			if (opponentAction instanceof Offer) {

				lastOpponentBid = ((Offer) opponentAction).getBid();
				oppUtility = utilitySpace.getUtility(lastOpponentBid);
				allBids.add(lastOpponentBid);

				updateOpp(sender);
			}
		}
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {
		updateTime();
		rounds++;
		Bid testBid = null;
		Value v;
		double calUtil = 0;
		double calThreshold;
		double tempThreshold;
		double minimalThreshold = 0.7;
		double deductThreshold = 0.1;
		double calculatedThreshold = 1 - (opponents.size() * deductThreshold);

		tempThreshold = Math.max(minimalThreshold, calculatedThreshold);
		tempThreshold = Math.max(tempThreshold, rv);

		// YXAgent Start 1st
		if (!possibleActions.contains(Accept.class)) {
			do {
				myLastBid = generateRandomBid();
				myUtility = utilitySpace.getUtility(myLastBid);

			} while (myUtility < minimalThreshold); // YXAgent starts 1st, so
													// YXAgent cant use
													// calculatedThreshold as
													// opponents.size = 0
			return myAction = new Offer(getPartyId(), myLastBid);
		}

		// YXAgent Start 2nd
		if (!searchedDiscountWithRV && rounds > 1) {
			if (evaluateDiscountFactorNReservationValue()) {
				return new EndNegotiation(getPartyId());
			}
		}

		do {
			testBid = generateRandomBid();
			myUtility = utilitySpace.getUtility(testBid);

		} while (myUtility < tempThreshold);

		// Acceptance Criteria
		if (rounds > 10 && normalizedTime <= 0.9) {// Initialization of Value
													// Integer is available
													// after 11 rounds
			for (Issue issue : issues) {
				v = lastOpponentBid.getValue(issue.getNumber());
				// Retrieve ToughestOpp issue weight and value weight
				if (issue.getType().toString().equals("INTEGER")) {
					for (Value vv : oppValueFrequency.get(hardestOpp).get(issue)
							.keySet()) { // Usual
											// way
											// of
											// key
											// retrieval
											// of
											// Hashmap
											// does
											// not
											// work
											// for
											// INTEGER
											// type,
											// thats
											// why
											// need
											// to
											// loop
											// through
						if (Integer.parseInt(vv.toString()) == Integer
								.parseInt(v.toString())) {
							calUtil += oppIssueWeight.get(hardestOpp).get(issue)
									* oppValueFrequency.get(hardestOpp)
											.get(issue).get(vv);
							break;
						}
					}
				} else {
					calUtil += oppIssueWeight.get(hardestOpp).get(issue)
							* oppValueFrequency.get(hardestOpp).get(issue)
									.get(v);
				}
			}

			try {
				calThreshold = (calUtil
						- ((opponents.size() * deductThreshold) * 3 / 4));
				calThreshold = Math.max(tempThreshold, calThreshold);
				myAction = oppUtility > calThreshold
						? new Accept(getPartyId(), lastOpponentBid)
						: new Offer(getPartyId(), testBid);
			} catch (Exception e) {
				e.printStackTrace();
			}
			return myAction;
		} else { // Run when rounds <= 10 or normalizedTime > 0.9
			try {
				myAction = oppUtility > tempThreshold
						? new Accept(getPartyId(), lastOpponentBid)
						: new Offer(getPartyId(), testBid);
			} catch (Exception e) {
				e.printStackTrace();
			}
			return myAction;
		}
	}

	// ---------------------------------------------------------------------------------------------------------//
	// Check Discount Factor wrt Reservation Value
	private boolean evaluateDiscountFactorNReservationValue() {
		double init = 1.0;
		double initRV1 = 0.75;
		double initRV2 = 0.86;
		double initRV3 = 0.95;
		double selfDiscount = 0.998;
		double deduction = 0.005;
		boolean endNegotiation = false;

		if (opponents.size() >= 3) { // Applicable for 3 or more opponent
										// scenario
			for (double i = init; i >= 0.; i -= 0.01) {
				String df = String.format("%.2f", i);
				double df1 = Double.parseDouble(df);
				if (discountFactor == df1 && rv >= initRV1) {
					endNegotiation = true;
				}
				initRV1 -= 0.005; // Every "i" decrement 0.01, RV -= 0.005
			}
			searchedDiscountWithRV = true;
		}

		if (opponents.size() == 2) { // Applicable for 2 opponent scenario only
			for (double i = init; i >= 0.; i -= 0.01) {
				String df = String.format("%.2f", i);
				double df1 = Double.parseDouble(df);
				if (discountFactor == df1 && rv >= initRV2) {
					endNegotiation = true;
				}
				initRV2 -= 0.006; // Every "i" decrement 0.01, RV -= 0.006
			}
			searchedDiscountWithRV = true;
		}

		if (opponents.size() == 1) { // Applicable for 1 opponent scenario only
			for (double i = init; i >= 0.; i -= 0.01) {
				String df = String.format("%.2f", i);
				double df1 = Double.parseDouble(df);
				if (discountFactor == df1 && rv >= initRV3) {
					endNegotiation = true;
				}
				initRV3 = (initRV3 - deduction) * selfDiscount; // Every "i"
																// decrement
																// 0.01, RV =
																// (RV -
																// deduction) *
																// selfDiscount;
			}
			searchedDiscountWithRV = true;
		}
		return endNegotiation;
	}

	// ---------------------------------------------------------------------------------------------------------//
	// Update Opponent
	private void updateOpp(AgentID sender) {

		if (rounds <= 10) {
			updateValueIntegerWeight(sender);
		}
		updateModelOppIssueWeight(sender);
		updateModelOppValueWeight(sender);
		oppBidHistory.get(sender).add(lastOpponentBid);
		retrieveToughestOpp();
		// printableWeight();
		// printableValue();
		// printableIntegerIssue();
		// printableToughest();
	}

	private void updateValueIntegerWeight(AgentID sender) {
		for (Issue issue : issues) {
			if (issue.getType().toString().equals("INTEGER")) {
				ValueInteger currIssueValueInteger = (ValueInteger) lastOpponentBid
						.getValue(issue.getNumber());
				oppIssueIntegerValue.get(sender).get(issue)
						.add(currIssueValueInteger);
			}
		}
		if (rounds == 10 && !updatedValueIntegerWeight) { // rounds = [0, 10]
															// meant 11 value
															// points
			int sumLower;
			int sumUpper;
			for (Object o1 : oppIssueIntegerValue.keySet()) {
				for (Issue issue : issues) {
					if (issue.getType().toString().equals("INTEGER")) {
						sumLower = 0;
						sumUpper = 0;
						int lowerB = ((IssueInteger) issue).getLowerBound();
						int upperB = ((IssueInteger) issue).getUpperBound();
						int mid = (lowerB + upperB) / 2;
						double average = 1. / (upperB - lowerB);
						double adjustment;

						// Loop to retrieve sum of points which lie on upper or
						// lower range
						for (int i = 0; i < oppIssueIntegerValue.get(o1)
								.get(issue).size(); i++) {
							int vv = Integer.parseInt(oppIssueIntegerValue
									.get(o1).get(issue).get(i).toString());
							if (vv <= mid) {
								sumLower += 1;
							} else {
								sumUpper += 1;
							}
						}
						ValueInteger vL = new ValueInteger(lowerB);
						ValueInteger vU = new ValueInteger(upperB);

						if (sumLower > sumUpper) {
							oppValueFrequency.get(o1).get(issue).put(vL, 1.);
							oppValueFrequency.get(o1).get(issue).put(vU, 0.);

							adjustment = 1.;
							// for loop assigning values between
							// lowerbound(exclusive) and upperbound(exclusive)
							for (int i = lowerB + 1; i < upperB; i++) {
								adjustment -= average;
								ValueInteger vI = new ValueInteger(i);
								oppValueFrequency.get(o1).get(issue).put(vI,
										adjustment);
							}
						} else {
							oppValueFrequency.get(o1).get(issue).put(vU, 1.);
							oppValueFrequency.get(o1).get(issue).put(vL, 0.);

							adjustment = 0.;
							// for loop assigning values between
							// lowerbound(exclusive) and upperbound(exclusive)
							for (int i = lowerB + 1; i < upperB; i++) {
								adjustment += average;
								ValueInteger vI = new ValueInteger(i);
								oppValueFrequency.get(o1).get(issue).put(vI,
										adjustment);
							}
						}
					}
				}
			}
			updatedValueIntegerWeight = true;
		}
	}

	private void updateModelOppIssueWeight(AgentID sender) {
		if (oppBidHistory.get(sender).size() != 0 && rounds >= 10) {
			Bid previousRoundBid = oppBidHistory.get(sender)
					.get(oppBidHistory.get(sender).size() - 1);// -1
																// since
																// current
																// lastOpponentBid
																// not
																// added
																// yet
			double issueWeightFormula = (Math.pow((1 - normalizedTime), 10))
					/ (issues.size() * 100);
			double issueWeightInteger = (Math.pow((1 - normalizedTime), 10))
					/ (issues.size() * 10);
			double issueSum = 0.;
			double normalizedIssueW;

			for (Issue issue : issues) {
				Value prevIssueValue = previousRoundBid
						.getValue(issue.getNumber());
				Value currIssueValue = lastOpponentBid
						.getValue(issue.getNumber());

				if (issueContainIntegerType) { // For Discrete & Integer Mix
												// Domain
					if (issue.getType().toString().equals("INTEGER")) { // Integer
																		// Handling
						if (incrementIntegerIssueWeight(issue, prevIssueValue,
								currIssueValue)) {
							oppIssueWeight.get(sender).put(issue,
									oppIssueWeight.get(sender).get(issue)
											+ issueWeightInteger);
						}
					} else {// Discrete Handling
						if (prevIssueValue == currIssueValue) {
							oppIssueWeight.get(sender).put(issue,
									oppIssueWeight.get(sender).get(issue)
											+ issueWeightInteger);
						}
					}
				} else { // For Discrete Only Domain
					if (prevIssueValue == currIssueValue) {
						oppIssueWeight.get(sender).put(issue,
								oppIssueWeight.get(sender).get(issue)
										+ issueWeightFormula);
					}
				}
				issueSum += oppIssueWeight.get(sender).get(issue);
			}

			// After Sum computed, Normalized Issue Weight
			for (Issue issue : issues) {
				normalizedIssueW = oppIssueWeight.get(sender).get(issue)
						/ issueSum;
				oppIssueWeight.get(sender).put(issue, normalizedIssueW);
			}
		}
	}

	public boolean incrementIntegerIssueWeight(Issue issue,
			Value prevIssueValue, Value currIssueValue) {
		double[] arraySlot = new double[5];
		boolean sameSection = false;
		double increment;// Store
		int pastV = Integer.parseInt(prevIssueValue.toString()); // past bid
																	// value
		int currV = Integer.parseInt(currIssueValue.toString()); // current bid
																	// value
		int lowerB = ((IssueInteger) issue).getLowerBound();
		int upperB = ((IssueInteger) issue).getUpperBound();
		double formula = (upperB - lowerB + 1) / 5;

		increment = lowerB + formula;

		for (int i = 0; i < 4; i++) {
			arraySlot[i] = increment;
			increment += formula;
		}
		arraySlot[4] = upperB;

		if (pastV <= arraySlot[0] && currV <= arraySlot[0]) {
			sameSection = true;
		} else if (pastV <= arraySlot[1] && currV <= arraySlot[1]
				&& pastV > arraySlot[0] && currV > arraySlot[0]) {
			sameSection = true;
		} else if (pastV <= arraySlot[2] && currV <= arraySlot[2]
				&& pastV > arraySlot[1] && currV > arraySlot[1]) {
			sameSection = true;
		} else if (pastV <= arraySlot[3] && currV <= arraySlot[3]
				&& pastV > arraySlot[2] && currV > arraySlot[2]) {
			sameSection = true;
		} else if (pastV <= arraySlot[4] && currV <= arraySlot[4]
				&& pastV > arraySlot[3] && currV > arraySlot[3]) {
			sameSection = true;
		}
		return sameSection;
	}

	private void updateModelOppValueWeight(AgentID sender) {
		double maxValueBase;
		double currValueBase;
		double normalizedValue;
		double valueWeightFormula = Math.pow(0.2, normalizedTime) / 30000;
		double valueWeightInteger = Math.pow(0.2, normalizedTime) / 955;
		// (Normal Rounds avg - 11000 rounds, Integer issue type round avg - 350
		// rounds)

		for (Issue issue : issues) {
			if (issueContainIntegerType) { // For Discrete & Integer Mix Domain
				if (!issue.getType().toString().equals("INTEGER")) {
					Value value = lastOpponentBid.getValue(issue.getNumber());
					oppValueFrequency.get(sender).get(issue).put(value,
							oppValueFrequency.get(sender).get(issue).get(value)
									+ valueWeightInteger);
				}

			} else { // For Discrete Only Domain
				Value value = lastOpponentBid.getValue(issue.getNumber());
				oppValueFrequency.get(sender).get(issue).put(value,
						oppValueFrequency.get(sender).get(issue).get(value)
								+ valueWeightFormula);
			}
		}

		// Normalization of Value Weight
		for (Issue issue : issues) {
			if (!issue.getType().toString().equals("INTEGER")) {
				maxValueBase = 0;

				// Compute Max Value for specific issue
				for (Value v : oppValueFrequency.get(sender).get(issue)
						.keySet()) {
					currValueBase = oppValueFrequency.get(sender).get(issue)
							.get(v);
					if (currValueBase > maxValueBase) {
						maxValueBase = currValueBase;
					}
				}

				// After Max Value computed, Normalized Value Weight
				for (Value v : oppValueFrequency.get(sender).get(issue)
						.keySet()) {
					normalizedValue = oppValueFrequency.get(sender).get(issue)
							.get(v) / maxValueBase;
					oppValueFrequency.get(sender).get(issue).put(v,
							normalizedValue);
				}
			}
		}
	}

	private void retrieveToughestOpp() {
		double sumDiff;
		double maxDiff;
		double diff;
		double hardestOppSumDiff = 0.;

		// Compute SumDiff of each opponent using value weight
		for (Object o : oppValueFrequency.keySet()) {
			sumDiff = 0;
			for (Issue issue : issues) {
				maxDiff = 0;
				for (Value v : oppValueFrequency.get(o).get(issue).keySet()) {
					if (!issue.getType().toString().equals("INTEGER")) { // Exclusive
																			// of
																			// Integer
																			// Type
																			// Issues
						diff = 1 - oppValueFrequency.get(o).get(issue).get(v);
						if (diff > maxDiff) {
							maxDiff = diff;
						}
					}
				}
				sumDiff += maxDiff;
			}
			oppSumDiff.put(o, sumDiff); // Store opponent with sumDiff
		}

		// Retrieve Toughest Opponent
		for (Object o : oppSumDiff.keySet()) {
			if (oppSumDiff.get(o) > hardestOppSumDiff) {
				hardestOppSumDiff = oppSumDiff.get(o);
				hardestOpp = o; // Update hardest opponent
			}
		}
	}

	/*
	 * private void printableWeight(){ PrintWriter aout = null; try { aout = new
	 * PrintWriter(new FileWriter("negotiationWeight")); } catch (IOException
	 * e1) { e1.printStackTrace(); }
	 * 
	 * StringBuilder rez;
	 * 
	 * for (Object a1 : oppIssueWeight.keySet()){ rez = new StringBuilder() ;
	 * rez.append(a1); rez.append("|"); rez.append(oppIssueWeight.get(a1));
	 * rez.append("|"); rez.append("\n"); aout.println(rez.toString()); }
	 * 
	 * rez = new StringBuilder(); rez.append("\n");
	 * rez.append("Number of rounds: " + rounds); aout.println(rez.toString());
	 * aout.close(); }
	 * 
	 * private void printableValue(){ PrintWriter aout = null; try { aout = new
	 * PrintWriter(new FileWriter("negotiationValue")); } catch (IOException e1)
	 * { e1.printStackTrace(); }
	 * 
	 * StringBuilder rez;
	 * 
	 * for (Object a1 : oppValueFrequency.keySet()){ rez = new StringBuilder();
	 * rez.append(a1); rez.append("|"); rez.append(oppValueFrequency.get(a1));
	 * rez.append("|"); rez.append("\n"); aout.println(rez.toString()); }
	 * 
	 * rez = new StringBuilder(); rez.append("\n");
	 * rez.append("Number of rounds: " + rounds); aout.println(rez.toString());
	 * aout.close(); }
	 * 
	 * private void printableIntegerIssue(){ PrintWriter aout = null; try { aout
	 * = new PrintWriter(new FileWriter("negotiationIntegerIssue")); } catch
	 * (IOException e1) { e1.printStackTrace(); }
	 * 
	 * StringBuilder rez;
	 * 
	 * for (Object a1 : oppIssueIntegerValue.keySet()){ rez = new
	 * StringBuilder(); rez.append(a1); rez.append("|");
	 * rez.append(oppIssueIntegerValue.get(a1)); rez.append("|");
	 * rez.append("\n"); aout.println(rez.toString()); } aout.close(); }
	 * 
	 * private void printableToughest(){ PrintWriter aout = null; try { aout =
	 * new PrintWriter(new FileWriter("negotiationHardestOpp")); } catch
	 * (IOException e1) { e1.printStackTrace(); }
	 * 
	 * StringBuilder rez;
	 * 
	 * for (Object a1 : oppSumDiff.keySet()){ rez = new StringBuilder();
	 * rez.append(a1); rez.append("|"); rez.append(oppSumDiff.get(a1));
	 * rez.append("|"); rez.append("\n"); aout.println(rez.toString()); } rez =
	 * new StringBuilder(); rez.append("\n"); rez.append("Toughest Opp is: " +
	 * hardestOpp); aout.println(rez.toString());
	 * 
	 * rez = new StringBuilder(); rez.append("\n");
	 * rez.append("Number of rounds: " + rounds); aout.println(rez.toString());
	 * aout.close(); }
	 */

	// ---------------------------------------------------------------------------------------------------------//
	// Initialize Opponent
	private void initOpp(AgentID sender) {// Initialize new Agent and Issues
		try {
			issueIntegerHandler(sender);
			initModelOppIssueWeight(sender);
			initModelOppValueFrequency(sender);
			oppBidHistory.put(sender, new ArrayList<Bid>());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void issueIntegerHandler(AgentID sender) {
		oppIssueIntegerValue.put(sender,
				new HashMap<Issue, ArrayList<ValueInteger>>());
		for (Issue issue : issues) {
			if (issue.getType().toString().equals("INTEGER")) {
				oppIssueIntegerValue.get(sender).put(issue,
						new ArrayList<ValueInteger>());
			}
		}
	}

	private void initModelOppIssueWeight(AgentID sender) {
		int numIssues;
		double avgW;
		oppIssueWeight.put(sender, new HashMap<Issue, Double>());
		for (Issue issue : issues) {
			oppIssueWeight.get(sender).put(issue, 0.);
		}

		numIssues = oppIssueWeight.get(sender).keySet().size();
		avgW = (double) 1 / numIssues;
		for (Issue i : oppIssueWeight.get(sender).keySet()) {
			oppIssueWeight.get(sender).put(i, avgW);
		}
	}

	private void initModelOppValueFrequency(AgentID sender) {
		ArrayList<Value> values = new ArrayList<Value>();
		oppValueFrequency.put(sender,
				new HashMap<Issue, HashMap<Value, Double>>());
		for (Issue issue : issues) {
			oppValueFrequency.get(sender).put(issue,
					new HashMap<Value, Double>());
			if (!issue.getType().toString().equals("INTEGER")) {
				values = getValues(issue);
				for (Value value : values) {
					oppValueFrequency.get(sender).get(issue).put(value, 1.);
				}
			}
		}
	}

	public ArrayList<Value> getValues(Issue issue) {
		ArrayList<Value> values = new ArrayList<Value>();
		switch (issue.getType()) {
		case DISCRETE:
			List<ValueDiscrete> valuesDis = ((IssueDiscrete) issue).getValues();
			for (Value value : valuesDis) {
				values.add(value);
			}
			break;
		case INTEGER:
			int min_value = ((IssueInteger) issue).getLowerBound();
			int max_value = ((IssueInteger) issue).getUpperBound();
			for (int i = min_value; i <= max_value; i++) {
				Object valueObject = new Integer(i);
				values.add((Value) valueObject);
			}
			break;
		case REAL:
			double min_value1 = ((IssueReal) issue).getLowerBound();
			double max_value1 = ((IssueReal) issue).getUpperBound();
			for (int i = (int) min_value1; i <= max_value1; i++) {
				Object valueObject = new Integer(i);
				values.add((Value) valueObject);
			}
			break;
		default:
			try {
				throw new Exception("issue type: " + issue.getType());
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return values;
	}

	// ---------------------------------------------------------------------------------------------------------//
	// Check for Integer Type in Issue
	private void issueContainIntegerType() {
		for (Issue issue : issues) {
			if (issue.getType().toString().equals("INTEGER")) {
				issueContainIntegerType = true;
				break;
			}
		}
	}

	// ---------------------------------------------------------------------------------------------------------//
	// Initialize and Update Time
	private void updateTime() {
		currTime = System.currentTimeMillis();
		diff = currTime - startTime;
		normalizedTime = (long) (diff / totalTime);
	}

	private void initTime() {
		startTime = System.currentTimeMillis();
		totalTime = 3 * 60 * 1000; // 3min total time converted to millsecond
		diff = 0;
	}

	// ---------------------------------------------------------------------------------------------------------//
	// Generate Random Bid
	@Override
	protected Bid generateRandomBid() {
		Bid randomBid = null;
		HashMap<Integer, Value> values = new HashMap<Integer, Value>();
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		for (Issue currentIssue : issues) {
			try {
				values.put(currentIssue.getNumber(),
						getRandomValue(currentIssue));
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		try {
			randomBid = new Bid(utilitySpace.getDomain(), values);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return randomBid;
	}

	@Override
	protected Value getRandomValue(Issue currentIssue) throws Exception {
		Value currentValue = null;
		int index = 0;

		Random randomnr = new Random();
		switch (currentIssue.getType()) {
		case DISCRETE:
			IssueDiscrete lIssueDiscrete = (IssueDiscrete) currentIssue;
			index = randomnr.nextInt(lIssueDiscrete.getNumberOfValues());
			currentValue = lIssueDiscrete.getValue(index);
			break;
		case REAL:
			IssueReal lIssueReal = (IssueReal) currentIssue;
			index = randomnr
					.nextInt(lIssueReal.getNumberOfDiscretizationSteps());
			currentValue = new ValueReal(
					lIssueReal.getLowerBound() + (((lIssueReal.getUpperBound()
							- lIssueReal.getLowerBound()))
							/ (lIssueReal.getNumberOfDiscretizationSteps()))
							* index);
			break;
		case INTEGER:
			IssueInteger lIssueInteger = (IssueInteger) currentIssue;
			index = randomnr.nextInt(lIssueInteger.getUpperBound()
					- lIssueInteger.getLowerBound() + 1);
			currentValue = new ValueInteger(
					lIssueInteger.getLowerBound() + index);
			break;
		default:
			throw new Exception(
					"issue type " + currentIssue.getType() + " not supported");
		}
		return currentValue;
	}

	/**
	 * @return the agentID
	 */
	public String getName() {
		return "YXAgent";
	}

	@Override
	public String getDescription() {
		return "ANAC2016";
	}

}
