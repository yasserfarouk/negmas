/*
Team members:
Eden Hartman
Shalom Hassid
Gal Lev
Tidhar Suchard
Yoav Wizhendler
 */

package agents.anac.y2019.eagent;

import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.uncertainty.AdditiveUtilitySpaceFactory;
import genius.core.utility.AbstractUtilitySpace;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.lang.Math;

public class EAgent extends AbstractNegotiationParty {
    public EAgent() {
    }

    public Action chooseAction(List<Class<? extends Action>> var1) {
        if (this.getLastReceivedAction() instanceof Offer && this.hasPreferenceUncertainty()) {
            Bid offerBid = ((Offer) this.getLastReceivedAction()).getBid();
            if (this.utilitySpace.getUtility(offerBid) > 0.8) {
                return new Accept(this.getPartyId(), offerBid);
            }
        }

        return new Offer(this.getPartyId(), this.generateSmartBid());
    }

    private Bid generateSmartBid() {
        try {
            Bid maxBid = this.utilitySpace.getMaxUtilityBid();
            return maxBid;
        } catch (Exception e) {
            e.printStackTrace();
            return new Bid(utilitySpace.getDomain());
        }
    }

    private float calculateStdev(List<Integer> intList) {
        // calculate average
        int sum = 0;
        Iterator intListIterator = intList.iterator();
        while (intListIterator.hasNext()) {
            sum += (int) intListIterator.next();
        }
        float avg = sum / intList.size();
        // calculate and return standard deviation
        float variance = 0;
        intListIterator = intList.iterator();
        while (intListIterator.hasNext()) {
            variance += Math.pow(((int) intListIterator.next()) - avg, 2);
        }
        variance /= intList.size();
        return (float)Math.sqrt(variance);
    }

    @Override
    public AbstractUtilitySpace estimateUtilitySpace() {
        AdditiveUtilitySpaceFactory utilitySpaceFactory = new AdditiveUtilitySpaceFactory(this.getDomain());
        List issuesList = utilitySpaceFactory.getIssues();
        Iterator issuesIterator = issuesList.iterator();
        // number of bids in the bid ranking
        int bidRankingSize = 0;
        // utility map: the utility for each value of each issue is stored in this map
        Map<String,Map<String, Float>> issuesValueToUtility = new HashMap<>();
        // bid count map: the number of bids each value of each issue appears in the bid ranking is stored in this map
        Map<String,Map<String, Integer>> issuesValueToBidCount = new HashMap<>();
        // rank map: the ranks (scores) for each value of each issue is stored in this map
        Map<String,Map<String, List<Integer>>> issuesValueToRanks = new HashMap<>();

        // initialize utility and bid count maps - iterate issues
        while (issuesIterator.hasNext()) {
            IssueDiscrete issueDiscrete = (IssueDiscrete) issuesIterator.next();
            Iterator issueDiscreteValuesIterator = issueDiscrete.getValues().iterator();
            Map<String, Float> valueToUtility = new HashMap<>();
            Map<String, Integer> valueToBidCount = new HashMap<>();
            Map<String, List<Integer>> valueToRanks = new HashMap<>();
            // iterate values of issue
            while (issueDiscreteValuesIterator.hasNext()) {
                ValueDiscrete valueDiscrete = (ValueDiscrete) issueDiscreteValuesIterator.next();
                valueToUtility.put(valueDiscrete.getValue(), (float) 0);
                valueToBidCount.put(valueDiscrete.getValue(), 0);
                List<Integer> rankList = new ArrayList<Integer>();
                valueToRanks.put(valueDiscrete.getValue(), rankList);
            }
            issuesValueToUtility.put(issueDiscrete.getName(), valueToUtility);
            issuesValueToBidCount.put(issueDiscrete.getName(), valueToBidCount);
            issuesValueToRanks.put(issueDiscrete.getName(), valueToRanks);
        }

        // calculate scores for the values (basic sum, without average or normalize), count bids for each value and fill rank (score) list for each value
        Iterator bidOrderIterator = this.userModel.getBidRanking().getBidOrder().iterator();
        int bidScore = 1;
        int totalBidScores = 0;
        // iterate bid order (preferences in ascending order - best is last)
        while (bidOrderIterator.hasNext()) {
            Bid bid = (Bid) bidOrderIterator.next();
            // iterate issues
            issuesIterator = issuesList.iterator();
            while (issuesIterator.hasNext()) {
                IssueDiscrete issueDiscrete = (IssueDiscrete) issuesIterator.next();
                String valueStr = bid.getValue(issueDiscrete).toString();
                // update utility of value
                Map<String, Float> valueToUtility = issuesValueToUtility.get(issueDiscrete.getName());
                float currScore = valueToUtility.get(valueStr);
                valueToUtility.put(valueStr, currScore + bidScore);
                // update bid count of value
                Map<String, Integer> valueToBidCount = issuesValueToBidCount.get(issueDiscrete.getName());
                int currBidCount = valueToBidCount.get(valueStr);
                valueToBidCount.put(valueStr, currBidCount + 1);
                // update rank list of value
                Map<String, List<Integer>> valueToRanks = issuesValueToRanks.get(issueDiscrete.getName());
                List<Integer> rankList = valueToRanks.get(valueStr);
                rankList.add(bidScore);
            }
            totalBidScores += bidScore;
            bidScore++;
        }
        bidRankingSize = bidScore - 1;

        // calculate average scores for the values, normalize them and set them in utility space
        issuesIterator = issuesList.iterator();
        while (issuesIterator.hasNext()) {
            IssueDiscrete issueDiscrete = (IssueDiscrete) issuesIterator.next();
            Map<String, Float> valueToUtility = issuesValueToUtility.get(issueDiscrete.getName());
            Map<String, Integer> valueToBidCount = issuesValueToBidCount.get(issueDiscrete.getName());
            float averageSumScore = totalBidScores / valueToUtility.size();
            float bidRankingSizeDividedValuesSize = bidRankingSize / valueToUtility.size();
            float maxNewSumScore = 0;
            // iterate values of issue - calculate average scores for the values
            Iterator issueDiscreteValuesIterator = issueDiscrete.getValues().iterator();
            while (issueDiscreteValuesIterator.hasNext()) {
                ValueDiscrete valueDiscrete = (ValueDiscrete) issueDiscreteValuesIterator.next();
                int bidCount = valueToBidCount.get(valueDiscrete.getValue());
                float newSumScore = 0;
                if (bidCount > 0) {
                    float sumScore = valueToUtility.get(valueDiscrete.getValue());
                    if (bidCount > bidRankingSizeDividedValuesSize) {
                        newSumScore = sumScore / bidCount;
                    } else {
                        newSumScore = sumScore / bidRankingSizeDividedValuesSize;
                    }
                } else {
                    newSumScore = averageSumScore / bidRankingSizeDividedValuesSize;
                }
                // update (average) utility of value
                valueToUtility.put(valueDiscrete.getValue(), newSumScore);
                if (newSumScore > maxNewSumScore) {
                    maxNewSumScore = newSumScore;
                }
            }
            // iterate values of issue - normalize scores and set them in utility space
            Iterator issueDiscreteValuesIterator1 = issueDiscrete.getValues().iterator();
            while (issueDiscreteValuesIterator1.hasNext()) {
                ValueDiscrete valueDiscrete = (ValueDiscrete) issueDiscreteValuesIterator1.next();
                float scoreNotNormalized = valueToUtility.get(valueDiscrete.getValue());
                float scoreNormalized = scoreNotNormalized / maxNewSumScore;
                // update (normalize) utility of value
                valueToUtility.put(valueDiscrete.getValue(), scoreNormalized);
                // set final score in utility space
                utilitySpaceFactory.setUtility(issueDiscrete, valueDiscrete, scoreNormalized);
            }
        }

        // calculate the weights of the issues and set them in utility space
        Map<String, Float> issueToStdevAvgInverse = new HashMap<>();
        float sumStdevAvgInverse = 0;
        issuesIterator = issuesList.iterator();
        while (issuesIterator.hasNext()) {
            IssueDiscrete issueDiscrete = (IssueDiscrete) issuesIterator.next();
            Map<String, List<Integer>> valueToRanks = issuesValueToRanks.get(issueDiscrete.getName());
            float stdevSum = 0;
            // iterate values of issue - calculate stdev sum for the values
            Iterator issueDiscreteValuesIterator = issueDiscrete.getValues().iterator();
            while (issueDiscreteValuesIterator.hasNext()) {
                ValueDiscrete valueDiscrete = (ValueDiscrete) issueDiscreteValuesIterator.next();
                List<Integer> rankList = valueToRanks.get(valueDiscrete.getValue());
                if (rankList.size() > 0) {
                    stdevSum += calculateStdev(rankList);
                }
            }
            float stdevAvg = stdevSum / valueToRanks.size();
            double stdevAvgInverse = Math.pow(stdevAvg, -1);
            issueToStdevAvgInverse.put(issueDiscrete.getName(), (float) stdevAvgInverse);
            sumStdevAvgInverse += stdevAvgInverse;
        }
        issuesIterator = issuesList.iterator();
        while (issuesIterator.hasNext()) {
            IssueDiscrete issueDiscrete = (IssueDiscrete) issuesIterator.next();
            float stdevAvgInverse = issueToStdevAvgInverse.get(issueDiscrete.getName());
            // set final weight of issue in utility space
            utilitySpaceFactory.setWeight(issueDiscrete, stdevAvgInverse / sumStdevAvgInverse);
        }

        utilitySpaceFactory.scaleAllValuesFrom0To1();
        return utilitySpaceFactory.getUtilitySpace();
    }

    public String getDescription() {
        return "Our negotiation agent for uncertain preferences";
    }
}
