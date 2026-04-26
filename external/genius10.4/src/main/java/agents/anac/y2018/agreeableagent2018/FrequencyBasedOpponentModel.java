package agents.anac.y2018.agreeableagent2018;


import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import java.io.Serializable;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;

/**
 * Created by Sahar Mirzayi
 * University of Tehran
 * Agent Lab.
 * Sahar.Mirzayi @ gmail.com
 */

public class FrequencyBasedOpponentModel implements Serializable {

    private List<Issue> domainIssues;
    private List<Map<String, Integer>> issueValueFrequency = new ArrayList<>();
    private int totalNumberOfBids;
    private int issueCount;
    private double[] issueWeight;

    public void init(List<Issue> issues) {
        domainIssues = issues;
        issueCount = issues.size();
        issueWeight = new double[issueCount];
        for (int i = 0; i < issueCount; i++) {
            issueWeight[i] = 0;
        }
        for (int i = 0; i < issueCount; i++) {
            int numberOfValues = ((IssueDiscrete) domainIssues.get(i)).getNumberOfValues();
            Map<String, Integer> x = new HashMap<>();
            for (int j = 0; j < numberOfValues; j++) {
                String s = ((IssueDiscrete) domainIssues.get(i)).getValue(j).toString();
                x.put(s, 0);
            }
            issueValueFrequency.add(x);
        }
    }

    public void updateModel(Bid bid, int numberOfBids) {
        if (bid == null) return;
        totalNumberOfBids = numberOfBids;
        for (int i = 0; i < domainIssues.size(); i++) {
            String key = bid.getValue(i + 1).toString();
            Integer currentValue = issueValueFrequency.get(i).get(key);
            currentValue++;
            issueValueFrequency.get(i).put(key, currentValue);
            updateIssueWeight();
        }
    }

    private void updateIssueWeight() {
        for (int i = 0; i < issueCount; i++) {
            Map<String, Integer> issue = issueValueFrequency.get(i);
            issueWeight[i] = calculateStandardDeviation(issue);
        }
    }

    private double calculateStandardDeviation(Map<String, Integer> issue) {
        double sum = 0, mean;
        double standardDeviation;
        int size = issue.size();
        for (Object o : issue.entrySet()) {
            Map.Entry pair = (Map.Entry) o;
            sum += (Integer) pair.getValue();
        }
        mean = sum / size;
        double sum2 = 0;

        for (Object o : issue.entrySet()) {
            Map.Entry pair2 = (Map.Entry) o;
            sum2 += Math.pow((mean - (Integer) pair2.getValue()), 2);
        }
        if (sum2 != 0)
            standardDeviation = Math.sqrt(sum2 / size);
        else
            standardDeviation = 0;
        return standardDeviation;
    }


    public double getUtility(Bid bids) {
        if (totalNumberOfBids == 0) return 0;
        double sumOfEachIssueUtility = 0;
        double sumOfIssueWeight=0;
        for (double anIssueWeight : issueWeight) {
            sumOfIssueWeight += anIssueWeight;
        }
        for (int i = 1; i <= domainIssues.size(); i++) {
            String value = bids.getValue(i).toString();
            Integer numberOfPreOffers = issueValueFrequency.get(i - 1).get(value);
            sumOfEachIssueUtility += ((double) numberOfPreOffers
                    / (double) totalNumberOfBids) * (issueWeight[i - 1]/sumOfIssueWeight);
        }
        return sumOfEachIssueUtility / issueCount;
    }
}
