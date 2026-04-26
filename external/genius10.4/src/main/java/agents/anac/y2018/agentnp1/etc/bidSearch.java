package agents.anac.y2018.agentnp1.etc;

import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.parties.NegotiationInfo;
import genius.core.utility.UtilitySpace;

public class bidSearch {
    private NegotiationInfo info;
	private UtilitySpace utilitySpace;
    private Bid maxBid = null; // 最大効用値Bid

    public bidSearch(UtilitySpace utilitySpace, NegotiationInfo info) throws Exception {
        this.info = info;
        this.utilitySpace = utilitySpace;
        initMaxBid(); // 最大効用値Bidの初期探索
//        negoStats.setValueRelativeUtility(maxBid); // 相対効用値を導出する
    }

    private void initMaxBid() throws Exception{
        int tryNum = info.getUtilitySpace().getDomain().getIssues().size(); // 試行回数
        Random rnd = new Random(info.getRandomSeed()); //Randomクラスのインスタンス化
        //maxBid = info.getUtilitySpace().getDomain().getRandomBid(rnd);
        maxBid = info.getUtilitySpace().getMaxUtilityBid();
        for (int i = 0; i < tryNum; i++) {
            try {
                do{
                    SimulatedAnnealingSearch(maxBid, 1.0);
                } while (info.getUtilitySpace().getUtility(maxBid) < info.getUtilitySpace().getReservationValue());
                if(info.getUtilitySpace().getUtility(maxBid) == 1.0){
                    break;
                }
            } catch (Exception e) {
                System.out.println("[Exception_Search] Failed to find the first Bid");
                e.printStackTrace();
            }
        }
//        System.out.println("[isPrinting_Search]:" + maxBid.toString() + " " + info.getUtilitySpace().getUtility(maxBid));
    }

    public Bid getBid(Bid baseBid, double threshold) {
        try {
            Bid bid = getBidbyAppropriateSearch(baseBid, threshold); // 閾値以上の効用値を持つ合意案候補を探索
            // 探索によって得られたBidがthresholdよりも小さい場合，最大効用値Bidを基準とする
            if (info.getUtilitySpace().getUtility(bid) < threshold) {
                bid = new Bid(maxBid);
            }

            Bid tempBid = new Bid(bid);

            // 探索によって得られたBidがthresholdよりも小さい場合
            if (info.getUtilitySpace().getUtility(tempBid) < threshold) {
                return bid;
            } else {
                return tempBid;
            }

        } catch (Exception e) {
            System.out.println("[Error]: failed to search bids");
            e.printStackTrace();
            return baseBid;
        }
    }

    private static int SA_ITERATION = 1;

    private Bid getBidbyAppropriateSearch(Bid baseBid, double threshold) {
        Bid bid = new Bid(baseBid);

        // In case of nonlinear, search is carried out, ★ In case of linear, change to generate bid randomly
        try {
                if(info.getUtilitySpace().getUtility(bid) < threshold){

                	Bid currentBid = null;
                    double currentBidUtil = 0;
                    double min = 1.0;
                    for (int i = 0; i < SA_ITERATION; i++) {
                        currentBid = SimulatedAnnealingSearch(bid, threshold);
                        currentBidUtil = info.getUtilitySpace().getUtility(currentBid);
                        if (currentBidUtil <= min && currentBidUtil >= threshold) {
                            bid = new Bid(currentBid);
                            min = currentBidUtil;
                        }
                    }
                } else {
                    bid = generateRandomBid();
                }
        } catch (Exception e) {
            System.out.println("[Error] failed to SA search");
            System.out.println("[Error] Problem with received bid(SA:last):" + e.getMessage() + ". cancelling bidding");
        }
        return bid;
    }

    // SA
    static double START_TEMPERATURE = 1.0; // start temperature
    static double END_TEMPERATURE = 0.0001; // end temperature
    static double COOL = 0.999; // cooling degree
    static int STEP = 1;// changable width
    static int STEP_NUM = 1; // changable number
    /**
     * SA
     * @param baseBid
     * @param threshold
     * @return
     * @throws Exception
     */
    private Bid SimulatedAnnealingSearch(Bid baseBid, double threshold) throws Exception {
        Bid currentBid = new Bid(baseBid); //Generation of initial value
        double currenBidUtil = info.getUtilitySpace().getUtility(baseBid);
        Bid nextBid = null; // Evaluation Bid
        double nextBidUtil = 0.0;
        ArrayList<Bid> targetBids = new ArrayList<Bid>(); //ArrayList of optimum utility value Bid
        double targetBidUtil = 0.0;
        double p; // Transition probability
        Random randomnr = new Random(); // random number
        double currentTemperature = START_TEMPERATURE; // current temparature
        double newCost = 1.0;
        double currentCost = 1.0;
        List<Issue> issues = info.getUtilitySpace().getDomain().getIssues();

        // Loop until temperature drops sufficiently
        while (currentTemperature > END_TEMPERATURE) {
            nextBid = new Bid(currentBid); // initialize next_bid

            // Get nearby Bid
            for (int i = 0; i < STEP_NUM; i++) {
                int issueIndex = randomnr.nextInt(issues.size()); // Randomly specify issues
                Issue issue = issues.get(issueIndex); // Issue with the specified index
                ArrayList<Value> values = getValues(issue);
                int valueIndex = randomnr.nextInt(values.size()); // Randomly specified within the range of possible values
                nextBid = nextBid.putValue(issue.getNumber(), values.get(valueIndex));
                nextBidUtil = info.getUtilitySpace().getUtility(nextBid);

                // Update maximum utility value Bid
                if (maxBid == null || nextBidUtil >= info.getUtilitySpace().getUtility(maxBid)) {
                    maxBid = new Bid(nextBid);
                }
            }

            newCost = Math.abs(threshold - nextBidUtil);
            currentCost = Math.abs(threshold - currenBidUtil);
            p = Math.exp(-Math.abs(newCost - currentCost) / currentTemperature);
            if (newCost < currentCost || p > randomnr.nextDouble()) {
                currentBid = new Bid(nextBid); // updatevBid
                currenBidUtil = nextBidUtil;
            }

            // update
            if (currenBidUtil >= threshold){
                if(targetBids.size() == 0){
                    targetBids.add(new Bid(currentBid));
                    targetBidUtil = info.getUtilitySpace().getUtility(currentBid);
                } else{
                    if(currenBidUtil < targetBidUtil){
                        targetBids.clear(); // initialize
                        targetBids.add(new Bid(currentBid)); // Add element
                        targetBidUtil = info.getUtilitySpace().getUtility(currentBid);
                    } else if (currenBidUtil == targetBidUtil){
                        targetBids.add(new Bid(currentBid)); // Add element
                    }
                }
            }
            currentTemperature = currentTemperature * COOL; // Lower temperature
        }

        if (targetBids.size() == 0) {
            // If a Bid having a utility value larger than the boundary value is not found, baseBid is returned
            return new Bid(baseBid);
        } else {
            // Returns Bid whose utility value is around the boundary value
            return new Bid(targetBids.get(randomnr.nextInt(targetBids.size())));
        }
    }

	protected Bid generateRandomBid() {

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

	protected Value getRandomValue(Issue currentIssue) throws Exception {

		Value currentValue = null;
		int index = 0;
		Random randomnr = new Random();

		switch (currentIssue.getType()) {
		case REAL:
			IssueReal lIssueReal = (IssueReal) currentIssue;
			index = randomnr.nextInt(lIssueReal.getNumberOfDiscretizationSteps());
			currentValue = new ValueReal(
					lIssueReal.getLowerBound() + ((lIssueReal.getUpperBound() - lIssueReal.getLowerBound())
							/ (double) lIssueReal.getNumberOfDiscretizationSteps()) * (double) index);
			break;

		case DISCRETE:
			IssueDiscrete lIssueDiscrete = (IssueDiscrete) currentIssue;
			index = randomnr.nextInt(lIssueDiscrete.getNumberOfValues());
			currentValue = lIssueDiscrete.getValue(index);
			break;

		case INTEGER:
			IssueInteger lIssueInteger = (IssueInteger) currentIssue;
			index = randomnr.nextInt((lIssueInteger.getUpperBound() - lIssueInteger.getLowerBound()) + 1);
			currentValue = new ValueInteger(lIssueInteger.getLowerBound() + index);
			break;

		default:
			throw new Exception((new StringBuilder("issue type ")).append(currentIssue.getType())
					.append(" not supported").toString());
		}
		return currentValue;
	}

    public ArrayList<Value> getValues(Issue issue) {
        ArrayList<Value> values = new ArrayList<Value>();

        switch(issue.getType()) {
            case DISCRETE:
                List<ValueDiscrete> valuesDis = ((IssueDiscrete)issue).getValues();
                for(Value value:valuesDis){
                    values.add(value);
                }
                break;
            case INTEGER:
                int min_value = ((IssueInteger)issue).getUpperBound();
                int max_value = ((IssueInteger)issue).getUpperBound();
                for(int j=min_value; j<=max_value; j++){
                    Object valueObject = new Integer(j);
                    values.add((Value)valueObject);
                }
                break;
            default:
                try {
                    throw new Exception("issue type \""+ issue.getType() + "\" not supported by" + info.getAgentID().getName());
                } catch (Exception e) {
                    System.out.println("[Exception] Failed to get the value of issues");
                    e.printStackTrace();
                }
        }

        return values;
    }

}
