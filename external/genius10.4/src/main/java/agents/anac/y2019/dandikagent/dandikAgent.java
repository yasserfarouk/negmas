package agents.anac.y2019.dandikagent;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import agents.org.apache.commons.math.stat.regression.OLSMultipleLinearRegression;
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.uncertainty.ExperimentalUserModel;
import genius.core.misc.Pair;


/**
 * This is dandikAgent by Mehmet Mert Ozgun
 */
public class dandikAgent extends AbstractNegotiationParty {

    private Bid currentBid = null;
    double[] sortedUtils;
    int oppMaxBidIndex = 0;
    HashMap uniqueBids = new HashMap();
    List<List<ValueDiscrete>> allIssues = new ArrayList<>();
    int countAll;
    OLSMultipleLinearRegression regression;
    ArrayList<Bid> allBidsList;
    double[] utilities;
    List<List<String>> allIssuesAsString;
    List<List<String>> allPossibleBids;
    List<Pair<List<String>, Double>> ourUtilityModel;
    ArrayList<Bid> sortedList;
    double[] allBidPredictions;
    boolean amIFirst = false;
    List<Bid> bidOrder;
    private double[][] myModel;
    NegotiationInfo info_;
    List<Issue> issues;
    private int updateCount;
    List<Bid> uniqueBids2 = new ArrayList<>();

    @Override
    public void init(NegotiationInfo info) {

        super.init(info);
        info_ = info;
        issues = info.getUserModel().getDomain().getIssues();
        utils.getIssueDiscrete(issues, allIssues);
        allIssuesAsString = issuesAsString();
        allPossibleBids = utils.generateAllPossibleBids(allIssuesAsString, 0);
        reverse(allPossibleBids);
        countAll = utils.getIssueCount(allIssues);
        bidOrder = info.getUserModel().getBidRanking().getBidOrder();

        if(countAll > bidOrder.size()){
            myModel = new double[getUtilitySpace().getDomain().getIssues().size()][];

            // Creating the 2d array by initializing non-fixed size rows
            for (int i = 0; i < getUtilitySpace().getDomain().getIssues().size(); i++) {

                IssueDiscrete issueDiscrete = (IssueDiscrete) generateRandomBid().getIssues().get(i);
                myModel[i] = new double[issueDiscrete.getValues().size()];
            }

            updateWeights();
            System.out.println(Arrays.deepToString(myModel));
        } else {

            double[][] oneHotEncoded = utils.encodeBids(bidOrder, countAll, allIssues);
            utilitySpace = getUtilitySpace();
            double[][] oneHotEncodedAll = utils.encodeListOfStrings(allPossibleBids, countAll, allIssues);
            utilities = new double[bidOrder.size()];

            utilities[0] = 1;
            for (int i = 1; i < utilities.length; i++) {
                utilities[i] = utilities[i - 1] + 1;
            }

            regression = new OLSMultipleLinearRegression();
            regression.newSampleData(utilities, oneHotEncoded);

            allBidPredictions = new double[oneHotEncodedAll.length];

            for (int i = 0; i < oneHotEncodedAll.length; i++) {
                allBidPredictions[i] = utils.predict(regression, oneHotEncodedAll[i]);
            }

            allBidsList = new ArrayList<>();
            for (int i = 0; i < allPossibleBids.size(); i++) {
                allBidsList.add(utils.asBid(info.getUserModel().getDomain(), convertToStringArray(allPossibleBids.get(i).toArray())));
            }

            final List<Bid> allBidsCopy = allBidsList;
            sortedList = new ArrayList<>(allBidsCopy);
            sortedList.sort(Comparator.comparing(s -> allBidPredictions[allBidsCopy.indexOf(s)]));
            sortedUtils = Arrays.stream(allBidPredictions).sorted().toArray();
            ourUtilityModel = new ArrayList<>();
            for (int i = 0; i < sortedList.size(); i++) {
                ourUtilityModel.add(new Pair<>(utils.bidToListOfString(sortedList.get(i)), sortedUtils[i]));
            }


            System.out.println(Arrays.toString(sortedUtils));
        }






    }

    private int getIndexOfValueInIssue(Bid bid, int issueIndex, String value) {
        IssueDiscrete is = (IssueDiscrete) bid.getIssues().get(issueIndex);
        return is.getValueIndex(value);
    }

    private void updateWeights() {
        int numberOfIssues = getUtilitySpace().getDomain().getIssues().size();
        for(int j = 0; j < bidOrder.size(); j++){
            for (int i = 0; i < numberOfIssues; i++) {
                ValueDiscrete currentBidValue = (ValueDiscrete) (bidOrder.get(j).getValue(i + 1));
                int currentBidValueIndex = getIndexOfValueInIssue(bidOrder.get(j), i, currentBidValue.getValue());

                if(bidOrder.indexOf(bidOrder.get(j)) >= bidOrder.size() * 0.9){
                    myModel[i][currentBidValueIndex] += 1;
                    updateCount++;
                }
            }
        }

    }

    @Override
    public Action chooseAction(List<Class<? extends Action>> validActions) {


        if(bidOrder.size() > countAll){

            uniqueBids.put(currentBid, 1);


            List<Double> sortedUtilsList = Arrays.stream(sortedUtils)
                    .boxed()
                    .collect(Collectors.toList());

            int lower = 0;
            if (timeline.getCurrentTime() > 998) {
                lower = (int) (sortedUtilsList.size() - 1);
            } else if (timeline.getCurrentTime() > 975) {
                lower = (int) (0.999 * sortedUtilsList.size());
            } else if (timeline.getCurrentTime() > 950) {
                lower = (int) (0.997 * sortedUtilsList.size());
            } else if (timeline.getCurrentTime() > 900) {
                lower = (int) (0.996 * sortedUtilsList.size());
            } else {
                lower = (int) (0.995 * sortedUtilsList.size());
            }

            int upper = sortedUtilsList.size();

            Double randomElement = sortedUtilsList.get((int) ((Math.random() * (upper - lower)) + lower));

            int indexOfRandom = sortedUtilsList.indexOf(randomElement);



            if (oppMaxBidIndex < sortedList.indexOf(currentBid)) {
                oppMaxBidIndex = sortedList.indexOf(currentBid);
            }

            boolean acceptable = sortedList.indexOf(currentBid) > sortedUtilsList.size() * 0.93;

            if (timeline.getCurrentTime() >= 999) {
                if (oppMaxBidIndex < sortedList.size() * 0.85)
                    return new Offer(getPartyId(), sortedList.get(sortedList.size() - 1));
                else if (acceptable)
                    return new Accept(getPartyId(), currentBid);
                else
                    return new Offer(getPartyId(), sortedList.get(oppMaxBidIndex));
            } else if (timeline.getCurrentTime() < 600 &&
                    sortedList.indexOf(currentBid) > sortedUtilsList.size() * 0.985) {
                return new Accept(getPartyId(), currentBid);
            } else if (timeline.getCurrentTime() < 990 && timeline.getCurrentTime() > 600 &&
                    sortedList.indexOf(currentBid) > sortedUtilsList.size() * 0.985) {
                return new Accept(getPartyId(), currentBid);
            } else if (timeline.getCurrentTime() > 990 && acceptable && timeline.getCurrentTime() < 999) {
                if (oppMaxBidIndex < sortedList.size() / 2)
                    return new Offer(getPartyId(), sortedList.get(sortedList.size() - 1));
                else
                    return new Accept(getPartyId(), currentBid);
            } else {
                return new Offer(getPartyId(), sortedList.get(indexOfRandom));
            }
        } else {

            if(updateCount != 0){
                HashMap<Integer, Value> values = new HashMap();
                for(int i = 0; i < issues.size(); i++){
                    IssueDiscrete issueDiscrete = (IssueDiscrete) generateRandomBid().getIssues().get(i);
                    int rn = rand.nextInt(issueDiscrete.getValues().size()) ;

                    if(myModel[i][rn] == 0)
                        i--;
                    else {
                        Value val = new ValueDiscrete(issueDiscrete.getValue(rn).toString());
                        values.put(i+1, val);
                    }

                }

                Bid b = new Bid(info_.getUserModel().getDomain(), values);

                if(!uniqueBids2.contains(b)){
                    uniqueBids2.add(b);
                }

                if(timeline.getCurrentTime() < 990){
                    return new Offer(getPartyId(), bidOrder.get(bidOrder.size()-1));
                } else{
                    if(uniqueBids2.contains(currentBid)){
                        return new Accept(getPartyId(), currentBid);
                    } else if(estimateUtilitySpace().getUtility(currentBid) > 0.8){
                        return new Accept(getPartyId(), currentBid);
                    } else
                        return new Offer(getPartyId(), b);

                }
            } else {
                if(timeline.getCurrentTime() < 999)
                    return new Offer(getPartyId(), bidOrder.get(bidOrder.size()-1));
                else if (timeline.getCurrentTime() < 1000)
                    return new Offer(getPartyId(), generateRandomBid());
                else
                    return new Accept(getPartyId(), currentBid);

            }


        }

    }

    // If we hold 2 bids, update the weights with it!
    @Override
    public void receiveMessage(AgentID sender, Action action) {
        try {
            if (action == null) {
                System.out.println("Hey! I am first!");
                amIFirst = true;

            }

            super.receiveMessage(sender, action);
            if (action instanceof Offer) {
                currentBid = ((Offer) action).getBid();

            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    private static String[] convertToStringArray(Object[] array) {
        String[] result = new String[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i].toString();
        }
        return result;
    }

    private void reverse(List<List<String>> allPossibleBids) {
        for (List<String> sublist : allPossibleBids)
            Collections.reverse(sublist);
    }

    private List<List<String>> issuesAsString() {
        List<List<String>> allIssuesAsString = new ArrayList<>();
        for (int i = 0; i < allIssues.size(); i++) {
            List<String> current = new ArrayList<>();
            for (int j = 0; j < allIssues.get(i).size(); j++) {
                current.add(allIssues.get(i).get(j).toString());

            }
            allIssuesAsString.add(current);
        }
        return allIssuesAsString;
    }


    @Override
    public String getDescription() {
        return "ANAC2019";
    }

}
