package agents.anac.y2019.garavelagent;

import java.util.List;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

import agents.org.apache.commons.math.stat.regression.OLSMultipleLinearRegression;
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.ValueDiscrete;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.uncertainty.AdditiveUtilitySpaceFactory;
import genius.core.uncertainty.BidRanking;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.misc.Pair;

/**
 * This is your negotiation party.
 */

public class GaravelAgent extends AbstractNegotiationParty {

    private Bid currentBid = null;
    private Bid beforeBid = null;
    private AbstractUtilitySpace utilitySpace = null;
    private double[][] opponentModelValue;
    private double[] opponentModelIssue;
    private int numberOfIssues;
    private ArrayList<Bid> opponentBids;
    private int bidCount = 0;
    double[] estimateUtilities;
    double[] sortedUtils;

    List<List<ValueDiscrete>> allIssues = new ArrayList<>();
    int countAll;
    double[][] omValueNormalized;
    OLSMultipleLinearRegression regression;
    ArrayList<Bid> allBidsList;
    double[] utilities;
    List<List<String>> allIssuesAsString;
    List<List<String>> allPossibleBids;
    List<Pair<List<String>, Double>> ourUtilityModel;
    List<Pair<List<String>, Double>> opponentUtilityModel;
    NegotiationInfo info_;

    ArrayList<Bid> sortedList;
    double[] allBidPredictions;
    Bid maxBid = null;
    List<List<String>> optimalBids;
    Boolean isRegressionPossible = true;
    List<Bid> bidOrder;

    @Override
    public void init(NegotiationInfo info) {

        super.init(info);

        try {
            List<Issue> issues = info.getUserModel().getDomain().getIssues();
            utils.getIssueDiscrete(issues, allIssues);
            allIssuesAsString = issuesAsString();
            allPossibleBids = utils.generateAllPossibleBids(allIssuesAsString, 0);
            utils.reverse(allPossibleBids);
            countAll = utils.getIssueCount(allIssues);
            info_ = info;

            bidOrder = info.getUserModel().getBidRanking().getBidOrder();
            AdditiveUtilitySpaceFactory factory = new AdditiveUtilitySpaceFactory(getDomain());
            BidRanking bidRanking = userModel.getBidRanking();
            factory.estimateUsingBidRanks(bidRanking);
            utilitySpace = getUtilitySpace();

            double[][] oneHotEncoded = utils.encodeBids(bidOrder, countAll, allIssues);
            double[][] oneHotEncodedAll = utils.encodeListOfStrings(allPossibleBids, countAll, allIssues);

            utilities = new double[bidOrder.size()];
            double highBid = info.getUserModel().getBidRanking().getHighUtility();
            double lowBid = info.getUserModel().getBidRanking().getLowUtility();
            utilities[0] = lowBid;
            utilities[utilities.length - 1] = highBid;

            double delta = highBid - lowBid;
            double decrementAmount = delta / (utilities.length - 1);


            for (int i = 1; i < utilities.length - 1; i++) {
                utilities[i] = utilities[i - 1] + decrementAmount;
            }

            regression = new OLSMultipleLinearRegression();
            regression.newSampleData(utilities, oneHotEncoded);

            allBidPredictions = new double[oneHotEncodedAll.length];

            for (int i = 0; i < oneHotEncodedAll.length; i++) {
                allBidPredictions[i] = utils.predict(regression, oneHotEncodedAll[i]);
            }

            double max = Arrays.stream(allBidPredictions).max().getAsDouble();

            for (int i = 0; i < oneHotEncodedAll.length; i++) {
                if (allBidPredictions[i] > 0.9) {
                    allBidPredictions[i] = utils.scale(allBidPredictions[i], 0.9, max);
                }
            }

            Bid sampleBid = generateRandomBid();
            numberOfIssues = sampleBid.getIssues().size();
            // Array to keep issueWeights we plan to sample
            opponentModelIssue = new double[numberOfIssues];

            // Array to keep valueWeights we plan to sample
            opponentModelValue = new double[numberOfIssues][];
            opponentBids = new ArrayList<>();

            // Creating the 2d array by initializing non-fixed size rows
            for (int i = 0; i < numberOfIssues; i++) {
                opponentModelIssue[i] = (double) 1 / numberOfIssues;
                IssueDiscrete issueDiscrete = (IssueDiscrete) sampleBid.getIssues().get(i);
                opponentModelValue[i] = new double[issueDiscrete.getValues().size()];
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

        } catch (Exception e) {
            isRegressionPossible = false;
        }

    }

    private static String[] convertToStringArray(Object[] array) {
        String[] result = new String[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i].toString();
        }
        return result;
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
    public Action chooseAction(List<Class<? extends Action>> validActions) {

        if (isRegressionPossible) {
            try {

                if (maxBid == null)
                    maxBid = currentBid;

                if (allBidsList.indexOf(currentBid) < allBidsList.indexOf(maxBid))
                    maxBid = currentBid;

                if (currentBid == null || !validActions.contains(Accept.class)) {
                    return new Offer(getPartyId(), getUtilitySpace().getMaxUtilityBid());
                }

                if (timeline.getCurrentTime() == 998) {
                    return new Offer(getPartyId(), getUtilitySpace().getMaxUtilityBid());
                } else if (timeline.getCurrentTime() == 999) {
                    return new Offer(getPartyId(), getUtilitySpace().getMaxUtilityBid());
                } else if (timeline.getCurrentTime() == 1000) {
                    if (0.84 <= utils.predict(regression, utils.encodeBid(maxBid, countAll, allIssues))) {
                        return new Offer(getPartyId(), maxBid);
                    } else {
                        try {
                            Random r = new Random();
                            Bid lastBid = utils.asBid(info_.getUserModel().getDomain(), toStringArray(optimalBids.get(r.nextInt(optimalBids.size()))));
                            if (0.70 <= utils.predict(regression, utils.encodeBid(lastBid, countAll, allIssues))) {
                                return new Offer(getPartyId(), lastBid);
                            } else {
                                return new Offer(getPartyId(), getUtilitySpace().getMaxUtilityBid());
                            }
                        } catch (Exception e) {
                            return new Offer(getPartyId(), getUtilitySpace().getMaxUtilityBid());
                        }

                    }
                }

                if (0.92 <= utils.predict(regression, utils.encodeBid(currentBid, countAll, allIssues))) {
                    System.out.println("accepted");
                    return new Accept(getPartyId(), currentBid);
                } else {
                    try {
                        if (timeline.getCurrentTime() >= 200) {
                            Random r = new Random();
                            return new Offer(getPartyId(), utils.asBid(info_.getUserModel().getDomain(), toStringArray(optimalBids.get(r.nextInt(optimalBids.size())))));
                        }
                    } catch (Exception e) {
                        System.out.println("freq failed");
                        return new Offer(getPartyId(), getUtilitySpace().getMaxUtilityBid());
                    }
                    return new Offer(getPartyId(), getUtilitySpace().getMaxUtilityBid());
                }

            } catch (Exception e) {
                e.printStackTrace();
                return new Offer(getPartyId(), generateRandomBid());
            } finally {
                if (currentBid != null) {
                    beforeBid = currentBid;
                }
            }
        } else {
            if (timeline.getCurrentTime() == 995) {
                return new Offer(getPartyId(), bidOrder.get(bidOrder.size() - 1));
            }else{
                return new Offer(getPartyId(), bidOrder.get(bidOrder.size() - 1));
            }
        }
    }

    private void updateWeights() {
        for (int i = 0; i < numberOfIssues; i++) {
            ValueDiscrete currentBidValue = (ValueDiscrete) (currentBid.getValue(i + 1));
            int currentBidValueIndex = utils.getIndexOfValueInIssue(i, currentBidValue.getValue(), currentBid);

            ValueDiscrete beforeBidValue = (ValueDiscrete) (beforeBid.getValue(i + 1));
            int beforeBidValueIndex = utils.getIndexOfValueInIssue(i, beforeBidValue.getValue(), currentBid);

            if (currentBidValueIndex == beforeBidValueIndex) {
                opponentModelIssue[i] += ((double) 1 / numberOfIssues);
            }
            opponentModelValue[i][currentBidValueIndex] += 1;
        }
    }

    // If we hold 2 bids, update the weights with it!
    @Override
    public void receiveMessage(AgentID sender, Action action) {

        try {
            super.receiveMessage(sender, action);
            if (action instanceof Offer) {

                currentBid = ((Offer) action).getBid();
                opponentBids.add(currentBid);
                if (!currentBid.equals(beforeBid) && beforeBid != null) {
                    updateWeights();
                    updateOMValues();

                    if (bidCount % 50 == 0) {
                        //updateOpponentModel();
                        estimateUtilities = estimateOpUtil(allPossibleBids, omValueNormalized);
                        opponentUtilityModel = utils.frequencyModelling(opponentBids, allIssuesAsString, opponentModelValue);

                        //optimal bids to offer are stored below
                        optimalBids = utils.getOptimalBids(allPossibleBids, utils.mostWanted, regression, info_.getUserModel().getDomain(), countAll, allIssues);
                    }
                }
                bidCount++;
            }
        } catch (Exception e) {
            //e.printStackTrace();
        }
    }

    private String[] toStringArray(List<String> input) {
        String[] result = new String[input.size()];
        for (int i = 0; i < input.size(); i++) {
            result[i] = input.get(i);
        }
        return result;
    }

    private void updateOMValues() {

        double sum = 0;
        for (int i = 0; i < opponentModelValue[0].length; i++) {
            sum += opponentModelValue[0][i];
        }

        omValueNormalized = new double[opponentModelValue.length][opponentModelValue[0].length];
        for (int i = 0; i < omValueNormalized.length; i++)
            omValueNormalized[i] = Arrays.copyOf(opponentModelValue[i], opponentModelValue[i].length);

        for (int i = 0; i < opponentModelValue.length; i++) {
            for (int j = 0; j < opponentModelValue[i].length; j++) {
                omValueNormalized[i][j] = opponentModelValue[i][j] / sum;
            }
        }

    }

    private double[] estimateOpUtil(List<List<String>> allPossibleBids, double[][] omValueNormalized) {
        double[] result = new double[allPossibleBids.size()];
        for (int i = 0; i < allPossibleBids.size(); i++) {
            for (int j = 0; j < allIssuesAsString.size(); j++) {
                for (int k = 0; k < allIssuesAsString.get(j).size(); k++) {
                    if (allPossibleBids.get(i).get(j).equals(allIssuesAsString.get(j).get(k))) {
                        result[i] += omValueNormalized[j][k];
                    }
                }
            }
        }
        return result;
    }

    @Override
    public String getDescription() {
        return "mezgit_soft";
    }

}
