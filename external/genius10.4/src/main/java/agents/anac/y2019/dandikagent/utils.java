package agents.anac.y2019.dandikagent;

import agents.org.apache.commons.math.stat.regression.OLSMultipleLinearRegression;
import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import java.util.*;

public class utils {

    public static List<List<String>> generateAllPossibleBids(List<List<String>> input, int i) {

        if (i == input.size()) {
            List<List<String>> result = new ArrayList<List<String>>();
            result.add(new ArrayList<String>());
            return result;
        }

        List<List<String>> result = new ArrayList<List<String>>();
        List<List<String>> recursive = generateAllPossibleBids(input, i + 1); // recursive call

        for (int j = 0; j < input.get(i).size(); j++) {
            for (int k = 0; k < recursive.size(); k++) {
                List<String> newList = new ArrayList<String>(recursive.get(k));
                newList.add(input.get(i).get(j));
                result.add(newList);
            }
        }
        return result;
    }

    public static double predict(OLSMultipleLinearRegression regression, double[] x) {
        if (regression == null) {
            throw new IllegalArgumentException("regression must not be null.");
        }
        double[] beta = regression.estimateRegressionParameters();

        // intercept at beta[0]
        double prediction = beta[0];
        for (int i = 1; i < beta.length; i++) {
            prediction += beta[i] * x[i - 1];
        }
        //
        return prediction;
    }

    public static double scale(double x, double lowerBound, double max) {
        return ((x - lowerBound) / (max - lowerBound)) * (1 - lowerBound) + lowerBound;
    }

    public static int getIssueCount(List<List<ValueDiscrete>> allIssues) {
        int countAll = 0;
        ArrayList<String> allIssuesAsArray = new ArrayList<>();


        for (int i = 0; i < allIssues.size(); i++) {
            for (int j = 0; j < allIssues.get(i).size(); j++) {
                allIssuesAsArray.add(allIssues.get(i).get(j).toString());
                countAll++;
            }
        }
        return countAll;
    }

    public static void getIssueDiscrete(List<Issue> issues, List<List<ValueDiscrete>> allIssues) {
        for (Issue x : issues) {
            IssueDiscrete is = (IssueDiscrete) x;
            allIssues.add(is.getValues());
        }
    }

    public static double[][] encodeBids(List<Bid> bidOrder, int countAll, List<List<ValueDiscrete>> allIssues) {
        double[][] oneHotEncoded = new double[bidOrder.size()][countAll];
        int count = 0;
        for (int i = 0; i < oneHotEncoded.length; i++) {
            for (int j = 0; j < oneHotEncoded[0].length; j++) {
                for (int k = 0; k < bidOrder.get(i).getValues().values().size(); k++) {
                    for (int l = 0; l < allIssues.get(k).size(); l++) {
                        if (bidOrder.get(i).getValues().values().toArray()[k].toString().equals(allIssues.get(k).get(l).toString())) {
                            oneHotEncoded[i][count] = 1.0;
                        } else {
                            oneHotEncoded[i][count] = 0.0;
                        }
                        count++;
                    }
                }
                count = 0;
            }
        }
        return oneHotEncoded;
    }

    public static Bid asBid(Domain domain, String[] asString) {
        HashMap<Integer, Value> values = new HashMap();
        for (int i = 1; i <= asString.length; i++) {
            Value val = new ValueDiscrete(asString[i - 1]);
            values.put(i, val);
        }
        return new Bid(domain, values);
    }

    public static List<String> bidToListOfString(Bid input) {
        List<String> result = new ArrayList<>();

        for(int i = 0;i<input.getValues().size();i++){
            result.add(input.getValues().values().toArray()[i].toString());
        }
        return result;
    }

    public static double[][] encodeListOfStrings(List<List<String>> bidOrder, int countAll, List<List<ValueDiscrete>> allIssues) {
        double[][] oneHotEncoded = new double[bidOrder.size()][countAll];
        int count = 0;
        for (int i = 0; i < oneHotEncoded.length; i++) {
            for (int j = 0; j < oneHotEncoded[0].length; j++) {
                for (int k = 0; k < bidOrder.get(i).size(); k++) {
                    for (int l = 0; l < allIssues.get(k).size(); l++) {
                        if (bidOrder.get(i).get(k).equals(allIssues.get(k).get(l).toString())) {
                            oneHotEncoded[i][count] = 1.0;
                        } else {
                            oneHotEncoded[i][count] = 0.0;
                        }
                        count++;
                    }
                }
                count = 0;
            }
        }
        return oneHotEncoded;
    }
}
