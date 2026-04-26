package agents.anac.y2019.garavelagent;

import agents.org.apache.commons.math.stat.regression.OLSMultipleLinearRegression;
import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.misc.Pair;

import java.util.*;

public class utils {

    private static HashMap<String, Integer> issueFrequency;
    public static List<String> mostWanted;

    public static HashMap<String, Integer> getFrequency() {
        return issueFrequency;
    }

    private static <K, V> K getKey(Map<K, V> map, V value) {
        for (Map.Entry<K, V> entry : map.entrySet()) {
            if (entry.getValue().equals(value)) {
                return entry.getKey();
            }
        }
        return null;
    }

    public static Double sum(Double[] arr) {
        Double sum = 0.0; // initialize sum
        int i;
        // Iterate through all elements and add them to sum
        for (i = 0; i < arr.length; i++)
            sum += arr[i];

        return sum;
    }

    public static int getIndexOfValueInIssue(int issueIndex, String value, Bid currentBid) {
        IssueDiscrete is = (IssueDiscrete) currentBid.getIssues().get(issueIndex);
        return is.getValueIndex(value);
    }

    // i is used for recursion, for the initial call this should be 0
    public static List<List<String>> generateAllPossibleBids(List<List<String>> input, int i) {

        // stop condition
        if (i == input.size()) {
            // return a list with an empty list
            List<List<String>> result = new ArrayList<List<String>>();
            result.add(new ArrayList<String>());
            return result;
        }

        List<List<String>> result = new ArrayList<List<String>>();
        List<List<String>> recursive = generateAllPossibleBids(input, i + 1); // recursive call

        // for each element of the first list of input
        for (int j = 0; j < input.get(i).size(); j++) {
            // add the element to all combinations obtained for the rest of the lists
            for (int k = 0; k < recursive.size(); k++) {
                // copy a combination from recursive
                List<String> newList = new ArrayList<String>(recursive.get(k));
                // add element of the first list
                newList.add(input.get(i).get(j));
                // add new combination to result
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

        double prediction = beta[0];
        for (int i = 1; i < beta.length; i++) {
            prediction += beta[i] * x[i - 1];
        }
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

    public static double[] encodeBid(Bid bid, int countAll, List<List<ValueDiscrete>> allIssues) {
        double[] oneHotEncoded = new double[countAll];
        int count = 0;

        for (int j = 0; j < oneHotEncoded.length; j++) {
            for (int k = 0; k < bid.getValues().values().size(); k++) {
                for (int l = 0; l < allIssues.get(k).size(); l++) {
                    if (bid.getValues().values().toArray()[k].toString().equals(allIssues.get(k).get(l).toString())) {
                        oneHotEncoded[count] = 1.0;
                    } else {
                        oneHotEncoded[count] = 0.0;
                    }
                    count++;
                }
            }
            count = 0;
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

        for (int i = 0; i < input.getValues().size(); i++) {
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

    private static double[] encodeListOfString(List<String> bidOrder, int countAll, List<List<ValueDiscrete>> allIssues) {
        double[] oneHotEncoded = new double[countAll];
        int count = 0;

        for (int j = 0; j < oneHotEncoded.length; j++) {
            for (int k = 0; k < bidOrder.size(); k++) {
                for (int l = 0; l < allIssues.get(k).size(); l++) {
                    if (bidOrder.get(k).equals(allIssues.get(k).get(l).toString())) {
                        oneHotEncoded[count] = 1.0;
                    } else {
                        oneHotEncoded[count] = 0.0;
                    }
                    count++;
                }
            }
            count = 0;
        }

        return oneHotEncoded;
    }


    public static List<Pair<List<String>, Double>> frequencyModelling(List<Bid> allOpponentBids, List<List<String>> allIssuesAsString, double[][] opponentModelValue) {

        HashMap<String, Integer> frequency = getIssueFrequency(allOpponentBids);
        List<String> keys = new ArrayList<>(frequency.keySet());
        List<Integer> values = new ArrayList<>(frequency.values());

        int[] indexes = indexesOfTopElements(values.stream().mapToInt(i -> i).toArray(), allIssuesAsString.size() / 2 - 1);

        mostWanted = new ArrayList<>();
        for (int i = 0; i < indexes.length; i++) {
            mostWanted.add(keys.get(indexes[i]));
        }

        double[][] normalized = normalize(allOpponentBids.get(0), allOpponentBids.size(), allIssuesAsString, frequency, opponentModelValue);
        double[][] model = normalized.clone();
        for (int i = 0; i < model.length; i++) {
            for (int j = 0; j < model[i].length; j++) {
                model[i][j] = (model[i][j] + (1.0 / model[i].length)) / 2.0;
            }
        }
        return generateOpponentBidSpace(model, allIssuesAsString);
    }

    public static List<List<String>> getOptimalBids(List<List<String>> allPossibleBids, List<String> mostWanted, OLSMultipleLinearRegression regression, Domain domain, int countAll, List<List<ValueDiscrete>> allIssues) {
        List<List<String>> result = clone(allPossibleBids);
        List<String> wanted = new ArrayList<>();
        for (int i = 0; i < mostWanted.size(); i++) {
            wanted.add(mostWanted.get(i).split("_")[1]);
        }

        for (int i = 0; i < wanted.size(); i++) {
            for (int j = 0; j < result.size(); j++) {
                if (!result.get(j).contains(wanted.get(i))) {
                    result.remove(j);
                    j--;
                }
            }
        }

        for (int i = 0; i < result.size(); i++) {
            if (predict(regression, encodeListOfString(result.get(i), countAll, allIssues)) < 0.9) {
                result.remove(i);
                i--;
            }
        }

        return result;
    }

    private static HashMap<String, Integer> getIssueFrequency(List<Bid> Bids) {
        issueFrequency = new HashMap<>();

        for (int i = 0; i < Bids.size(); i++) {
            for (int j = 0; j < Bids.get(i).getValues().values().size(); j++) {
                String currentIssue = Bids.get(i).getIssues().get(j).toString() + "_" + Bids.get(i).getValues().values().toArray()[j].toString();
                if (issueFrequency.containsKey(currentIssue)) {
                    issueFrequency.put(currentIssue, issueFrequency.get(currentIssue) + 1);
                } else {
                    issueFrequency.put(currentIssue, 1);
                }
            }
        }
        return issueFrequency;
    }

    private static double[][] normalize(Bid sampleBid, int bidCount, List<List<String>> allIssuesAsString, HashMap<String, Integer> frequency, double[][] opponentModelValue) {
        double[][] result = Arrays.copyOf(opponentModelValue, opponentModelValue.length);

        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                result[i][j] = frequency.getOrDefault(sampleBid.getIssues().get(i).toString() + "_" + allIssuesAsString.get(i).get(j), 0) / (double) bidCount;
            }
        }
        return result;
    }

    private static List<Pair<List<String>, Double>> generateOpponentBidSpace(double[][] model, List<List<String>> allIssuesAsString) {
        HashMap<List<String>, Double> opponentBidSpace = new HashMap<>();

        List<List<String>> allBids = generateAllPossibleBids(allIssuesAsString, 0);
        reverse(allBids);

        for (int i = 0; i < allBids.size(); i++) {
            double util = 0;
            for (int j = 0; j < allBids.get(i).size(); j++) {
                util += model[j][allIssuesAsString.get(j).indexOf(allBids.get(i).get(j))];
            }
            opponentBidSpace.put(allBids.get(i), util);
        }

        return sortByValue(opponentBidSpace);

    }

    public static void reverse(List<List<String>> allPossibleBids) {
        for (List<String> sublist : allPossibleBids)
            Collections.reverse(sublist);
    }

    private static List<Pair<List<String>, Double>> sortByValue(HashMap<List<String>, Double> input) {
        List<Pair<List<String>, Double>> result = new ArrayList<>();
        ArrayList<Double> valueList = new ArrayList<>(input.values());
        Collections.sort(valueList);
        double min = valueList.get(0);
        double max = valueList.get(valueList.size() - 1);
        double scaleFactor = max - min;
        for (int i = 0; i < valueList.size(); i++) {
            result.add(new Pair<>(getKey(input, valueList.get(i)), ((valueList.get(i) - min) / scaleFactor)));
        }

        return result;
    }

    private static int[] indexesOfTopElements(int[] orig, int nummax) {
        try {
            int[] copy = Arrays.copyOf(orig, orig.length);
            Arrays.sort(copy);
            int[] honey = Arrays.copyOfRange(copy, copy.length - nummax, copy.length);
            int[] result = new int[nummax];
            int resultPos = 0;
            for (int i = 0; i < orig.length; i++) {
                int onTrial = orig[i];
                int index = Arrays.binarySearch(honey, onTrial);
                if (index < 0) continue;
                result[resultPos++] = i;
            }
            return result;
        } catch (Exception e) {
            return new int[]{0};
        }

    }

    private static List<List<String>> clone(final List<List<String>> src) {
        List<List<String>> dest = new ArrayList<>();
        for (List<String> sublist : src) {
            List<String> temp = new ArrayList<>();
            for (String val : sublist) {
                temp.add(val);
            }
            dest.add(temp);
        }
        return dest;
    }

}
