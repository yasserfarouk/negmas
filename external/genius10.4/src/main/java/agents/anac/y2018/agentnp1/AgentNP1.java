package agents.anac.y2018.agentnp1;

import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

import agents.anac.y2018.agentnp1.etc.bidSearch;
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
import genius.core.issue.ValueInteger;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.utility.UtilitySpace;

public class AgentNP1 extends AbstractNegotiationParty {
	private NegotiationInfo info;
	private UtilitySpace utilitySpace;
	private List<Issue> issues;
	private Action myAction;
	private Bid myLastBid;
	private Bid lastOppBid; // the last bid
	private Bid offerBid;
	private double geneUtil; //my agent generates a utility
	private double recvUtil; // opponent's agenet generates a utility for my agent
	private double rv; // reservation value
	private double df; // discount factor
	private int rounds; //negotiation rounds
	private boolean upValueIntWeight;
	private boolean issueContainIntType;
	private boolean DF_RV;
	private ArrayList<AgentID> opponents; // Negotiating partners' list
	private ArrayList<Bid> allBids; // All Bids
	private ArrayList<Double> oppUtil;
	private ArrayList<Double> myUtil;
	private HashMap<AgentID, List<Bid>> oppBidHistory; // opponent bids' history
	private HashMap<AgentID, Map<Issue, Double>> IssueWeight; // issues & weights
	private HashMap<AgentID, Map<Issue, List<ValueInteger>>> IssueIntValue;
	private HashMap<AgentID, Map<Issue, Map<Value, Double>>> ValueFrequency;
	private HashMap<AgentID, Double> oppSumDiff;
	private AgentID hardestOpp;
	private long startTime; // negotiation start time
	private long currTime; // current time
	private long diff;
	private double totalTime;
	private double normalizedTime;
//	private long normalizedTime;
	private bidSearch bidSearch;

	private Map<String, Integer> map;

	public AgentNP1() {
		myAction = null;
		myLastBid = null;
		lastOppBid = null;
		offerBid = null;
		geneUtil = 0.0D;
		recvUtil = 0.0D;
	}

	@Override
	public void init(NegotiationInfo info) { //info,utility space, deadline, time, etc...information

		super.init(info); // call init AbstractNegotiationParty class
		this.info = info;
		utilitySpace = info.getUtilitySpace(); //confirm utility space
		rv = utilitySpace.getReservationValue().doubleValue(); // reservation value
		df = info.getUtilitySpace().getDiscountFactor(); //discount factor
		issues = utilitySpace.getDomain().getIssues(); // issues which set within that domain
		allBids = new ArrayList<Bid>();
		oppUtil = new ArrayList<Double>();
		myUtil = new ArrayList<Double>();    //my own bid utility
		opponents = new ArrayList<AgentID>(); //negotiators list
		oppBidHistory = new HashMap<AgentID, List<Bid>>(); //the history of the negotiators
		IssueWeight = new HashMap<AgentID, Map<Issue, Double>>();
		IssueIntValue = new HashMap<AgentID, Map<Issue, List<ValueInteger>>>();
		ValueFrequency = new HashMap<AgentID, Map<Issue, Map<Value, Double>>>();
		oppSumDiff = new HashMap<AgentID, Double>();
		rounds = 0;
		upValueIntWeight = false;
		issueContainIntType = false;
		DF_RV = false;
		initTime();
		issueContainIntType();

		map = new TreeMap<String, Integer>();

		try {
			bidSearch = new bidSearch(utilitySpace, info);
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public void receiveMessage(AgentID sender, Action opponentAction) {

		super.receiveMessage(sender, opponentAction);
        //System.out.println(sender);
		if (sender != null && !opponents.contains(sender)) { // In the case of the first negotiating partner
			opponents.add(sender); // Add sender to negotiating partner list
			initOpp(sender); // Secure storage area for each sender
		}

		// in case of "Offer" negotiation action
		if (sender != null && opponentAction != null && (opponentAction instanceof Offer)) {
			lastOppBid = ((Offer) opponentAction).getBid(); // Acquire current negotiation opponent bid (ex: price 3.5$, color red, vebdor docomo etc...)
			recvUtil = utilitySpace.getUtility(lastOppBid);
			allBids.add(lastOppBid); // add a bid
		    oppUtil.add(recvUtil);
			updateOpp(sender);
		}
	}

	public Action chooseAction(List<Class<? extends Action>> possibleActions) {

		updateTime();
		double timePerRound = timeline.getTime() / (double) rounds;
		double remainingRounds = (1.0D - timeline.getTime()) / timePerRound;
		double minimalThreshold = 0.6999999999999999;
		double upperThreshold = 0.999999999999999;
		double acceptThreshold = 0.779999999999999;
		minimalThreshold = minimalThreshold * df;
		rounds++;

		if (!possibleActions.contains("negotiator/actions/Accept")) {
			if (timeline.getTime() > 0.0D && timeline.getTime() <= 0.9D) {
				acceptThreshold = 0.81D;
				if (recvUtil > acceptThreshold) {
					return new Accept(getPartyId(), lastOppBid);
				}
			} else if (timeline.getTime() > 0.9D && timeline.getTime() <= 0.99D) {
				acceptThreshold = 0.78D;
				if (recvUtil > acceptThreshold) {
					return new Accept(getPartyId(), lastOppBid);
				} else if (!DF_RV && judge_DF_RV()) {
					return new EndNegotiation(getPartyId());
				}
			} else if (timeline.getTime() > 0.97D ) {
				if (recvUtil > rv) {
					return new Accept(getPartyId(), lastOppBid);
				} else if (recvUtil <= rv) {
					return new EndNegotiation(getPartyId());
				}
			}

		    double max = 0.0D;
		    double sum = 0.0D;

			if (timeline.getTime() < 0.5D) {
				minimalThreshold = 0.96 - (0.195 *timeline.getTime());
				Bid myLastBid = bidSearch.getBid(generateRandomBid(), minimalThreshold);
				geneUtil = utilitySpace.getUtility(myLastBid);
		        allBids.add(myLastBid);  //add the created my own bid
                myUtil.add(geneUtil );
                System.out.println("[AgentNP Bid (" + rounds + ")]: " + myLastBid );
                System.out.println("[AgentNP Utility (" + rounds + ")]: " + geneUtil );
		        return myAction = new Offer(getPartyId(), myLastBid);
			} else if (timeline.getTime() >= 0.5D && timeline.getTime() < 0.7D) {
				minimalThreshold = 0.85 - (0.105 *timeline.getTime());
				Bid myLastBid = bidSearch.getBid(generateRandomBid(), minimalThreshold);
				geneUtil = utilitySpace.getUtility(myLastBid);
		        allBids.add(myLastBid);  //add the created my own bid
                myUtil.add(geneUtil );
                System.out.println("[AgentNP Bid (" + rounds + ")]: " + myLastBid );
                System.out.println("[AgentNP Utility (" + rounds + ")]: " + geneUtil );
		        return myAction = new Offer(getPartyId(), myLastBid);
			} else if (timeline.getTime() >= 0.7D && timeline.getTime() <0.95 ) {
                for (int i = 0; i < oppUtil.size(); i++) {
                	sum += oppUtil.get(i);
                	max = Math.max(max, oppUtil.get(i));
                }
                double ave = sum / oppUtil.size();
                if (max > 0.7799D && ave > 6.999D) {
				    minimalThreshold = ave ;
				    upperThreshold =max ;
                } else {
				    minimalThreshold = ave + 0.095D ;
				    upperThreshold =max + 0.095D ;
                }

				do {
					myLastBid = generateRndBid();
					geneUtil = utilitySpace.getUtility(myLastBid); // get utility value of Bid created randomly
					for (int i = 0; i <myUtil.size(); i++) {
						if (myUtil.get(i) == geneUtil) {
							geneUtil = utilitySpace.getUtility(myLastBid); // get utility value of Bid created randomly
						}
					}

				} while (geneUtil < minimalThreshold || geneUtil > upperThreshold); //When a random value smaller than mini is generated, generation is terminated

				allBids.add(myLastBid);
                myUtil.add(geneUtil );
                System.out.println("[AgentNP Bid (" + rounds + ")]: " + myLastBid );
                System.out.println("[AgentNP Utility (" + rounds + ")]: " + geneUtil );
				return myAction = new Offer(getPartyId(), myLastBid); //contents of bid

			} else if ( timeline.getTime() >= 0.95) {
				for (int i = 0; i < oppUtil.size(); i++) {
                	sum += oppUtil.get(i);
                	max = Math.max(max, oppUtil.get(i));
                }
                double ave = sum / oppUtil.size();
                if (max > 0.7799D && ave > 6.999D) {
				    minimalThreshold = ave ;
				    upperThreshold =max ;
                } else {
				    minimalThreshold = ave + 0.0749 ;
				    upperThreshold =max + 0.0749 ;
                }				do {
				    myLastBid = generateRndBid(); // generate a random bid
				    geneUtil = utilitySpace.getUtility(myLastBid); // get utility value of Bid created randomly

					for (int i = 0; i <myUtil.size(); i++) {
						if (myUtil.get(i) == geneUtil) {
							geneUtil = utilitySpace.getUtility(myLastBid); // get utility value of Bid created randomly
						}
					}

                } while (geneUtil < minimalThreshold || geneUtil > upperThreshold); //When a random value smaller than mini is generated, generation is terminated
                allBids.add(myLastBid);
                myUtil.add(geneUtil );
                System.out.println("[AgentNP Bid (" + rounds + ")]: " + myLastBid );
                System.out.println("[AgentNP Utility (" + rounds + ")]: " + geneUtil );
                return myAction = new Offer(getPartyId(), myLastBid);
			}

			}
		return myAction;
	}

	/**
	 *
	 * @return
	 */

	private boolean judge_DF_RV() {

		double init = 1.0D;
		double initRV = 0.75D;
		boolean endNegotiation = false;

		for (double i = init; i >= 0.0D; i -= 0.01D) {
			String df0 = String.format("%.2f", new Object[] { Double.valueOf(i) });
			double df1 = Double.parseDouble(df0);
			if (df == df1 && rv >= initRV) {
				endNegotiation = true;
			}
			initRV -= 0.0050000000000000001D;
		}
		DF_RV = true;
		return endNegotiation;
	}

	/**
	 *
	 * @param sender
	 */
	private void updateOpp(AgentID sender) {
		if (timeline.getTime() < 0.07D) {
			upValueIntWeight(sender);
		}
		upModelIssWeight(sender);
		updateModelOppValueWeight(sender);
		oppBidHistory.get(sender).add(lastOppBid);
		retrieveToughestOpp();
	}

	/**
	 *
	 * @param sender
	 */
	private void upValueIntWeight(AgentID sender) {
		for (Issue issue : issues) {
			if (issue.getType().toString().equals("INTEGER")) {
				ValueInteger currIssueValueInteger = (ValueInteger) lastOppBid.getValue(issue.getNumber());
				IssueIntValue.get(sender).get(issue).add(currIssueValueInteger); //point
			}
		}

		if (timeline.getTime() < 0.07D && !upValueIntWeight) {
			for (AgentID agent : IssueIntValue.keySet()) {
				for (Issue issue : issues) {
					if (issue.getType().toString().equals("INTEGER")) {
						int sumLower = 0;
						int sumUpper = 0;
						int lowerB = ((IssueInteger) issue).getLowerBound();
						int upperB = ((IssueInteger) issue).getUpperBound();
						int mid = (lowerB + upperB) / 2;
						double average = 1.0D / (double) (upperB - lowerB);

						for (int i = 0; i < IssueIntValue.get(agent).get(issue).size(); i++) {
							int vv = Integer.parseInt(IssueIntValue.get(agent).get(issue).get(i).toString());
							if (vv <= mid)
								sumLower++;
							else
								sumUpper++;
						}

						ValueInteger vL = new ValueInteger(lowerB);
						ValueInteger vU = new ValueInteger(upperB);
						if (sumLower > sumUpper) {
							ValueFrequency.get(agent).get(issue).put(vL, Double.valueOf(1.0D));
							ValueFrequency.get(agent).get(issue).put(vU, Double.valueOf(0.0D));
							double adjustment = 1.0D;
							for (int i = lowerB + 1; i < upperB; i++) {
								adjustment -= average;
								ValueInteger vI = new ValueInteger(i);
								ValueFrequency.get(agent).get(issue).put(vI, Double.valueOf(adjustment));
							}
						} else {
							ValueFrequency.get(agent).get(issue).put(vU, Double.valueOf(1.0D));
							ValueFrequency.get(agent).get(issue).put(vL, Double.valueOf(0.0D));
							double adjustment = 0.0D;
							for (int i = lowerB + 1; i < upperB; i++) {
								adjustment += average;
								ValueInteger vI = new ValueInteger(i);
								ValueFrequency.get(agent).get(issue).put(vI, Double.valueOf(adjustment));
							}
						}
					}
				}
			}

			upValueIntWeight = true;
		}
	}

	private void upModelIssWeight(AgentID sender) {     //calculate weight
		if (oppBidHistory.get(sender).size() != 0 && timeline.getTime() > 0.05D) {
			Bid formerRoundBid = oppBidHistory.get(sender).get((oppBidHistory.get(sender)).size() - 1);
			double issueWeightFormula = Math.pow(1D - normalizedTime, 10D) / (double) (issues.size() * 100);
			double issueWeightInteger = Math.pow(1D - normalizedTime, 10D) / (double) (issues.size() * 10);
			double issueSum = 0.0D;

			for (Issue issue : issues) {
				Value formIssueValue = formerRoundBid.getValue(issue.getNumber()); //selected former value
				Value currIssueValue = lastOppBid.getValue(issue.getNumber());          //selected current value

				if (issueContainIntType) {
					if (issue.getType().toString().equals("INTEGER")) {
						if (incrIntIssWeight(issue, formIssueValue, currIssueValue)) {
							IssueWeight.get(sender).put(issue, Double
									.valueOf(IssueWeight.get(sender).get(issue).doubleValue() + issueWeightInteger));
						}
					} else if (formIssueValue == currIssueValue) {
						IssueWeight.get(sender).put(issue, Double
								.valueOf(IssueWeight.get(sender).get(issue).doubleValue() + issueWeightInteger));
					}
				} else if (formIssueValue == currIssueValue) {
					IssueWeight.get(sender).put(issue,
							Double.valueOf(IssueWeight.get(sender).get(issue).doubleValue() + issueWeightFormula));
				}
				issueSum += IssueWeight.get(sender).get(issue).doubleValue();
			}

			for (Issue issue : issues) {
				double normalizedIssueW = IssueWeight.get(sender).get(issue).doubleValue() / issueSum;
				IssueWeight.get(sender).put(issue, Double.valueOf(normalizedIssueW));      //IssueWeight: weight
			}
		}
	}

	/**
	 *
	 * @param issue
	 * @param formIssueValue
	 * @param currIssueValue
	 * @return
	 */
	public boolean incrIntIssWeight(Issue issue, Value formIssueValue, Value currIssueValue) {

		double arraySlot[] = new double[5];
		boolean sameSection = false;
		int pastV = Integer.parseInt(formIssueValue.toString());
		int currV = Integer.parseInt(currIssueValue.toString());
		int lowerB = ((IssueInteger) issue).getLowerBound();
		int upperB = ((IssueInteger) issue).getUpperBound();
		double formula = ((upperB - lowerB) + 1) / 5;
		double increment = (double) lowerB + formula;
		for (int i = 0; i < 4; i++) {
			arraySlot[i] = increment;
			increment += formula;
		}

		arraySlot[4] = upperB;
		if ((double) pastV <= arraySlot[0] && (double) currV <= arraySlot[0])
			sameSection = true;
		else if ((double) pastV <= arraySlot[1] && (double) currV <= arraySlot[1] && (double) pastV > arraySlot[0]
				&& (double) currV > arraySlot[0])
			sameSection = true;
		else if ((double) pastV <= arraySlot[2] && (double) currV <= arraySlot[2] && (double) pastV > arraySlot[1]
				&& (double) currV > arraySlot[1])
			sameSection = true;
		else if ((double) pastV <= arraySlot[3] && (double) currV <= arraySlot[3] && (double) pastV > arraySlot[2]
				&& (double) currV > arraySlot[2])
			sameSection = true;
		else if ((double) pastV <= arraySlot[4] && (double) currV <= arraySlot[4] && (double) pastV > arraySlot[3]
				&& (double) currV > arraySlot[3])
			sameSection = true;
		return sameSection;
	}

	private void updateModelOppValueWeight(AgentID sender) {

		double valueWeightFormula = Math.pow(0.20000000000000001D, normalizedTime) / 30000D;
		double valueWeightInteger = Math.pow(0.20000000000000001D, normalizedTime) / 955D;

		for (Issue issue : issues) {
			if (issueContainIntType) {
				if (!issue.getType().toString().equals("INTEGER")) {
					Value value = lastOppBid.getValue(issue.getNumber());
					ValueFrequency.get(sender).get(issue).put(value, Double.valueOf(
							ValueFrequency.get(sender).get(issue).get(value).doubleValue() + valueWeightInteger));
				}
			} else {
				Value value = lastOppBid.getValue(issue.getNumber());
				ValueFrequency.get(sender).get(issue).put(value, Double.valueOf(
						ValueFrequency.get(sender).get(issue).get(value).doubleValue() + valueWeightFormula));
			}
		}

		for (Issue issue : issues) {
			if (!issue.getType().toString().equals("INTEGER")) {
				double maxValueBase = 0.0D;
				for (Value value : ValueFrequency.get(sender).get(issue).keySet()) {
					double currValueBase = ValueFrequency.get(sender).get(issue).get(value).doubleValue();
					if (currValueBase > maxValueBase)
						maxValueBase = currValueBase;
				}

				for (Value value : ValueFrequency.get(sender).get(issue).keySet()) {
					double normalizedValue = ValueFrequency.get(sender).get(issue).get(value).doubleValue()
							/ maxValueBase;
					ValueFrequency.get(sender).get(issue).put(value, Double.valueOf(normalizedValue));
				}
			}
		}
	}

	private void retrieveToughestOpp() {

		double hardestOppSumDiff = 0.0D;
		double sumDiff;
		for (AgentID agent : ValueFrequency.keySet()) {
			sumDiff = 0.0D;
			for (Issue issue : issues) {
				double maxDiff = 0.0D;
				for (Value value : ValueFrequency.get(agent).get(issue).keySet()) {
					if (!issue.getType().toString().equals("INTEGER")) {
						double diff = 1.0D - ValueFrequency.get(agent).get(issue).get(value).doubleValue();
						if (diff > maxDiff)
							maxDiff = diff;
					}
				}
				sumDiff += maxDiff;
			}
			oppSumDiff.put(agent, Double.valueOf(sumDiff));
		}

		for (AgentID agent : oppSumDiff.keySet()) {
			if (oppSumDiff.get(agent).doubleValue() > hardestOppSumDiff) {
				hardestOppSumDiff = oppSumDiff.get(agent).doubleValue();
				hardestOpp = agent;
			}
		}
	}

	/**
	 *
	 * @param sender
	 */
	private void initOpp(AgentID sender) {

		try {
			issueIntegerHandler(sender);
			initModelIssueWeight(sender);
			initModelValueFrequency(sender);
			oppBidHistory.put(sender, new ArrayList<Bid>());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 *
	 * @param sender
	 */
	private void issueIntegerHandler(AgentID sender) {

		IssueIntValue.put(sender, new HashMap<Issue, List<ValueInteger>>());
		for (Issue issue : issues) {
			if (issue.getType().toString().equals("INTEGER"))
				IssueIntValue.get(sender).put(issue, new ArrayList<ValueInteger>());
		}
	}

	/**
	 * Weight
	 *
	 * @param sender
	 */
	private void initModelIssueWeight(AgentID sender) {

		IssueWeight.put(sender, new HashMap<Issue, Double>());

		for (Issue issue : issues) {
			IssueWeight.get(sender).put(issue, Double.valueOf(0.0D));
		}

		int numIssues = IssueWeight.get(sender).keySet().size();
		double avgW = 1.0D / (double) numIssues;
		for (Issue issue : IssueWeight.get(sender).keySet()) {
			IssueWeight.get(sender).put(issue, Double.valueOf(avgW));
		}
	}

	/**
	 *
	 * @param sender
	 */
	private void initModelValueFrequency(AgentID sender) {

		ValueFrequency.put(sender, new HashMap<Issue, Map<Value, Double>>());

		for (Issue issue : issues) {
			ValueFrequency.get(sender).put(issue, new HashMap<Value, Double>());

			if (!issue.getType().toString().equals("INTEGER")) {
				for (Value value : getValues(issue)) {
					ValueFrequency.get(sender).get(issue).put(value, Double.valueOf(1.0D));
				}
			}
		}
	}

	/**
	 *
	 * @param issue
	 *            issue
	 * @return
	 */
	public ArrayList<Value> getValues(Issue issue) {

		ArrayList<Value> values = new ArrayList<Value>();
		switch (issue.getType()) {
		case DISCRETE:
			for (Value value : ((IssueDiscrete) issue).getValues()) {
				values.add(value);
			}
			break;

		case REAL:
			double min_value1 = ((IssueReal) issue).getLowerBound();
			double max_value1 = ((IssueReal) issue).getUpperBound();
			for (int i = (int) min_value1; (double) i <= max_value1; i++) {
				Object valueObject = new Integer(i);
				values.add((Value) valueObject);
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

		default:

		}

		return values;
	}

	private void issueContainIntType() {

		for (Issue issue : issues) {
			if (issue.getType().toString().equals("INTEGER")) {
				issueContainIntType = true;
				break;
			}
		}
	}

	private void updateTime() {

		currTime = System.currentTimeMillis();
		diff = currTime - startTime;
		normalizedTime= diff / totalTime;
	}

	private void initTime() {

		startTime = System.currentTimeMillis();
		totalTime = 180000D;
		diff = 0L;
	}

    protected Bid generateRndBid() {

		Bid randomBid = null;
		HashMap<Integer, Value> values = new HashMap<Integer, Value>();
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		for (Issue issue : issues) {
			try {
				values.put(Integer.valueOf(issue.getNumber()), getRandomValue(issue));
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

	public String getAgentName() {
		return "AgentNP1-1";
	}

    @Override
    public String getDescription() {
        return "ANAC2018";
    }

}
