package agents.anac.y2019.solveragent;

import agents.org.apache.commons.math.stat.regression.OLSMultipleLinearRegression;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.boaframework.OutcomeSpace;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.uncertainty.AdditiveUtilitySpaceFactory;
import genius.core.utility.AbstractUtilitySpace;

import java.util.*;
import java.util.Map.Entry;
import java.util.stream.Collectors;


public class SolverAgent extends AbstractNegotiationParty {
    private List<List<IssueValuePair>> comparisonPerIssue;
    private List<Map<Value, IssueValuePair>> pairMapList;
    private Bid lastReceivedBid = null;
    private double firstPhaseLow = 0;
    private OutcomeSpace outcomespace;
    private List<Bid> phase2Bids;
    private boolean isConcession = false;
    private boolean isFirst = false;
    private boolean nashFound = false;
    private Bid previousBid;
    private List<Bid> previousBids;
    private List<List<ValueDiscrete>> values;
    private Map<Bid, Double> reversedBidMap;
    private List<List<Double>> expectedWeightsPerValueOrdering;
    private Bid maxBid;
    private Map<Bid, Double> previousOpponentBidsMap;
    private int issuesSize;
    private Map<Double, Bid> nashBids;
    private Bid nashBid;
    private List<Bid> bidsAfterNash;
    private Random rand;
    private Map<Double, List<Bid>> utilBidMap;
    private ArrayList<Double> utilBidMapKeys;
    private int valuesCount;
    private Set<Value> opponentValues;
    private Bid ourMinBid;
    private List<Bid> sortedBids;
    private List<Bid> phase1Bids;
    private int phase2Index = 0;
    private int bidGivenCount = 0;
    double phase1Bound;

    private double[][] encodeListOfStrings(List<Bid> bidOrder, int countAll) {
        double[][] oneHotEncoded = new double[bidOrder.size()][countAll];
        int count = 0;
        for (int i = 0; i < oneHotEncoded.length; i++) {
            for (int j = 0; j < oneHotEncoded[0].length; j++) {
                for (int k = 0; k < issuesSize; k++) {
                    for (int l = 0; l < values.get(k).size(); l++) {
                        if (bidOrder.get(i).getValue(k + 1).equals(values.get(k).get(l))) {
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

    private static <T> List<List<T>> powerSet(List<T> originalSet) {
        List<List<T>> sets = new ArrayList<>();
        if (originalSet.isEmpty()) {
            sets.add(new ArrayList<>());
            return sets;
        }
        List<T> list = new ArrayList<>(originalSet);
        T head = list.get(0);
        List<T> rest = new ArrayList<>(list.subList(1, list.size()));
        for (List<T> set : powerSet(rest)) {
            List<T> newSet = new ArrayList<>();
            newSet.add(head);
            newSet.addAll(set);
            sets.add(newSet);
            sets.add(set);
        }
        return sets;
    }

    private static <T> List<T> reverseList(List<T> list) {
        return list.stream()
                .collect(Collectors.collectingAndThen(
                        Collectors.toCollection(ArrayList::new), lst -> {
                            Collections.reverse(lst);
                            return lst.stream();
                        }
                )).collect(Collectors.toCollection(ArrayList::new));
    }

    class ComparisonObject {
        private List<IssueValuePair> items;
        private List<Integer> issues;
        int weight;


        ComparisonObject(List<IssueValuePair> items, List<Integer> issues, int weight) {
            this.issues = new ArrayList<>();
            this.issues.addAll(issues);
            this.items = new ArrayList<>(Collections.nCopies(pairMapList.size(), new IssueValuePair(-1, -1)));
            int i = 0;
            for (Integer iss : issues) {
                this.items.set(iss, items.get(i++));
            }
            this.weight = weight;
        }

        IssueValuePair getValueAtIssueNo(int i) {
            return items.get(i - 1);
        }

        void setWeight(int weight) {
            this.weight = weight;
        }

        boolean isComparable(ComparisonObject other) {
            for (IssueValuePair pair : other.getItems()) {
                if (this.items.contains(pair))
                    return false;
            }
            return true;
        }

        int getValueIndex(int iss) {
            return comparisonPerIssue.get(issues.get(iss)).indexOf(items.get(issues.get(iss)));
        }

        double getValueWeightOfIssue(int iss) {
            return expectedWeightsPerValueOrdering.get(iss).get(comparisonPerIssue.get(iss).indexOf(items.get(iss)));
        }

        List<IssueValuePair> getItems() {
            return items;
        }

        public List<Integer> getIssues() {
            return issues;
        }

        int getIssueSize() {
            return issues.size();
        }

        public String toString() {
            StringBuilder string = new StringBuilder();
            string.append("[").append(issues.get(0)).append(":").append(items.get(issues.get(0)));
            for (int i = 1; i < issues.size(); i++) {
                string.append(", ").append(issues.get(i)).append(":").append(items.get(issues.get(i)));
            }
            string.append("]");
            return string.toString();
        }

    }

    class IssueValuePair {
        int first;
        int second;

        IssueValuePair(int first, int second) {
            this.first = first;
            this.second = second;
        }

        public String toString() {
            if (first == -1) return "None";
            return getKeysByValue(pairMapList.get(first), this).toString();
        }

        private Value getKeysByValue(Map<Value, IssueValuePair> map, IssueValuePair value) {
            for (Value val : map.keySet()) {
                if (map.get(val) == value) return val;
            }
            return new ValueDiscrete("0");
        }

        public boolean isNone() {
            return first == -1 && second == -1;
        }
    }

    @Override
    public void init(NegotiationInfo info) {
        super.init(info);
        List<Bid> bidOrder = info.getUserModel().getBidRanking().getBidOrder();
        previousBids = new ArrayList<>();
        rand = new Random();
        valuesCount = 0;

        outcomespace = new OutcomeSpace(utilitySpace);

        List<Issue> issues = info.getUserModel().getDomain().getIssues();
        issuesSize = issues.size();
        previousOpponentBidsMap = new HashMap<>();
        nashBids = new HashMap<>();
        bidsAfterNash = new ArrayList<>();
        opponentValues = new HashSet<>();
        isFirst = false;



        values = new ArrayList<>();

        for (Issue x : issues) {
            values.add(((IssueDiscrete) x).getValues());
            valuesCount += ((IssueDiscrete) x).getValues().size();
        }

        pairMapList = new ArrayList<>();

        for (int i = 0; i < values.size(); i++) {
            Map<Value, IssueValuePair> valuesMap = new HashMap<>();
            for (int j = 0; j < values.get(i).size(); j++) {
                valuesMap.put(values.get(i).get(j), new IssueValuePair(i, j));
            }
            pairMapList.add(valuesMap);
        }


        List<List<IssueValuePair>> bidsList = new ArrayList<>();
        for (Bid bid : bidOrder) {
            List<Value> ignored = new ArrayList<>(bid.getValues().values());
            List<IssueValuePair> finalList = new ArrayList<>();
            for (int i = 0; i < ignored.size(); i++) {
                IssueValuePair finalListVal = pairMapList.get(i).get(ignored.get(i));
                finalList.add(finalListVal);
            }
            bidsList.add(finalList);
        }

        int bidListSize = bidsList.size();

        HashMap<List<IssueValuePair>, List<ComparisonObject>> valuesPerFixedValues = new LinkedHashMap<>();
        comparisonPerIssue = new ArrayList<>(Collections.nCopies(issuesSize, new ArrayList<>()));

        for (int i = 0; i < bidListSize; i++) {
            List<IssueValuePair> lops = new ArrayList<>(bidsList.get(i));
            List<List<IssueValuePair>> allSets = powerSet(new ArrayList<>(lops));
            for (List<IssueValuePair> set : allSets) {
                List<IssueValuePair> valueList = new ArrayList<>(lops);
                List<IssueValuePair> key = new ArrayList<>(set);
                valueList.removeAll(key);

                List<Integer> valueIssueList = new ArrayList<>();

                for (IssueValuePair index : valueList)
                    valueIssueList.add(index.first);

                if (lops.size() == 0 || valueList.size() <= 0) continue;
                ComparisonObject valObject = new ComparisonObject(valueList, valueIssueList, i);

                if (valuesPerFixedValues.containsKey(key)) {
                    boolean allComparable = true;
                    for (ComparisonObject comp : valuesPerFixedValues.get(key))
                        if (!comp.isComparable(valObject))
                            allComparable = false;
                    if (allComparable)
                        valuesPerFixedValues.get(key).add(valObject);
                } else {
                    List<ComparisonObject> objectList = new ArrayList<>();
                    objectList.add(valObject);
                    valuesPerFixedValues.put(key, objectList);
                }
            }
        }

        valuesPerFixedValues.entrySet().removeIf(e -> e.getValue().size() < 2);

        for (Entry<List<IssueValuePair>, List<ComparisonObject>> entry : valuesPerFixedValues.entrySet()) {
            int issueNo = entry.getValue().get(0).getIssues().get(0);
            if (entry.getKey().size() == issuesSize - 1 && values.get(issueNo).size() == entry.getValue().size()) {
                List<IssueValuePair> tempList = new ArrayList<>();
                for (ComparisonObject comp : entry.getValue()) {
                    tempList.add(comp.getValueAtIssueNo(issueNo + 1));
                    comparisonPerIssue.set(issueNo, tempList);
                }
            }
        }

        int p = 0;
        List<Integer> issuesBroken = new ArrayList<>();
        for (List<IssueValuePair> entry : comparisonPerIssue) {
            if (entry.isEmpty()) {
                issuesBroken.add(p);
            }
            p++;
        }


        Map<Integer, Map<List<IssueValuePair>, List<ComparisonObject>>> orderings = new LinkedHashMap<>();
        for (int i = issuesSize - 1; i > 0; i--) {
            HashMap<List<IssueValuePair>, List<ComparisonObject>> valuesPerFixedValuesTmp = new LinkedHashMap<>();
            for (Entry<List<IssueValuePair>, List<ComparisonObject>> x : valuesPerFixedValues.entrySet()) {
                if (x.getKey().size() == i) {
                    if (valuesPerFixedValuesTmp.containsKey(x.getKey())) {
                        valuesPerFixedValuesTmp.get(x.getKey()).addAll(x.getValue());
                    } else {
                        valuesPerFixedValuesTmp.put(x.getKey(), x.getValue());
                    }
                }
            }

            orderings.put(i, valuesPerFixedValuesTmp);
        }

        List<List<IssueValuePair>> allConflictComparisons = new ArrayList<>();

        for (Entry<List<IssueValuePair>, List<ComparisonObject>> entry : orderings.get(issuesSize - 1).entrySet()) {
            for (Integer x : entry.getValue().get(0).getIssues()) {
                List<IssueValuePair> tempSet = new ArrayList<>();
                for (ComparisonObject comp : entry.getValue()) {
                    tempSet.add(comp.getValueAtIssueNo(x + 1));
                }
                allConflictComparisons.addAll(new ArrayList<>(powerSet(tempSet)));
                allConflictComparisons.removeIf(e -> e.size() != 2);
            }
        }

        while (true) {
            int prevSize = allConflictComparisons.size();
            for (int i1 = issuesSize - 2; i1 > 0; i1--) {
                for (Entry<List<IssueValuePair>, List<ComparisonObject>> entry : orderings.get(i1).entrySet()) {
                    int issueCount = entry.getValue().get(0).issues.size();

                    for (int i11 = 0; i11 < entry.getValue().size(); i11++) {
                        for (int i12 = i11 + 1; i12 < entry.getValue().size(); i12++) {
                            List<Integer> conflicts = new ArrayList<>();
                            ComparisonObject comp1 = entry.getValue().get(i11);
                            ComparisonObject comp2 = entry.getValue().get(i12);
                            for (int iss = 0; iss < issueCount; iss++) {
                                List<IssueValuePair> tmp = new ArrayList<>();
                                tmp.add(comp1.getValueAtIssueNo(comp1.issues.get(iss) + 1));
                                tmp.add(comp2.getValueAtIssueNo(comp1.issues.get(iss) + 1));
                                Collections.reverse(tmp);
                                if (allConflictComparisons.contains(tmp)) {
                                    conflicts.add(comp1.issues.get(iss));
                                }
                            }
                            if (conflicts.size() == issuesSize - i1 - 1) {
                                List<Integer> finalIssues = new ArrayList<>(comp1.issues);
                                finalIssues.removeAll(conflicts);
                                List<IssueValuePair> finalList = new ArrayList<>();
                                finalList.add(comp1.getValueAtIssueNo(finalIssues.get(0) + 1));
                                finalList.add(comp2.getValueAtIssueNo(finalIssues.get(0) + 1));
                                if (allConflictComparisons.contains(finalList)) continue;
                                allConflictComparisons.add(finalList);
                            }
                        }
                    }
                }
            }

            if (allConflictComparisons.size() == prevSize) {
                break;
            }
        }

        for (int i = issuesSize - 1; i > 0; i--) {
            for (Integer iss : issuesBroken) {
                List<IssueValuePair> orderingsComparison = new ArrayList<>();
                for (List<ComparisonObject> entry : orderings.get(i).values()) {
                    for (ComparisonObject comp : entry) {
                        IssueValuePair pair = comp.getValueAtIssueNo(iss + 1);
                        if (pair.isNone()) continue;
                        if (orderingsComparison.isEmpty()) orderingsComparison.add(pair);
                        else if (orderingsComparison.contains(pair)) {
                            orderingsComparison.remove(pair);
                            orderingsComparison.add(pair);
                        } else orderingsComparison.add(pair);
                    }
                }
                if (comparisonPerIssue.isEmpty() || orderingsComparison.size() > comparisonPerIssue.get(iss).size())
                    comparisonPerIssue.set(iss, orderingsComparison);
            }
        }


        for (List<IssueValuePair> comparison : allConflictComparisons) {
            int iss = comparison.get(0).first;
            List<IssueValuePair> issueList = comparisonPerIssue.get(iss);

            int index1 = issueList.indexOf(comparison.get(0));
            int index2 = issueList.indexOf(comparison.get(1));
            if (index1 > index2) {
                Collections.swap(issueList, index1, index2);
                comparisonPerIssue.set(iss, issueList);
            }
        }

        for (int i = 0; i < issuesSize; i++) {
            List<IssueValuePair> tmp = new ArrayList<>();
            for (ValueDiscrete x : values.get(i)) {
                tmp.add(pairMapList.get(i).get(x));
            }

            if (!comparisonPerIssue.containsAll(tmp)) {
                List<IssueValuePair> excludedVals = new ArrayList<>(tmp);
                excludedVals.removeAll(comparisonPerIssue.get(i));

                comparisonPerIssue.get(i).addAll(comparisonPerIssue.get(i).size() / 2, excludedVals);

                for (int l = 0; l < excludedVals.size(); l++) {
                    for (int l1 = l + 1; l1 < excludedVals.size(); l1++) {
                        for (int j = 0; j < 2; j++) {
                            for (int k = bidsList.size() - 1; k > bidsList.size() - 3; k--) {
                                if ((bidsList.get(j).contains(excludedVals.get(l)) && bidsList.get(k).contains(excludedVals.get(l1)) &&
                                        comparisonPerIssue.get(i).indexOf(excludedVals.get(l)) < comparisonPerIssue.get(i).indexOf(excludedVals.get(l1)))) {
                                    Collections.swap(comparisonPerIssue.get(i), comparisonPerIssue.get(i).indexOf(excludedVals.get(l)),
                                            comparisonPerIssue.get(i).indexOf(excludedVals.get(l1)));
                                }
                            }
                        }
                    }
                }

            }
        }

        expectedWeightsPerValueOrdering = new ArrayList<>();
        for (int i = 0; i < issuesSize; i++) {
            int valueAmount = values.get(i).size();
            List<Double> expected = new ArrayList<>();
            for (int j = 0; j < valueAmount; j++) {
                expected.add(((double) j + 0.75) / valueAmount);
            }
            expectedWeightsPerValueOrdering.add(expected);
        }


        List<Issue> issueRanking = new ArrayList<>(issues);
        List<List<Issue>> issueRankingList = new ArrayList<>();
        System.out.println(orderings);

        if (orderings.keySet().contains(issuesSize - 2)) {
            for (Entry<List<IssueValuePair>, List<ComparisonObject>> entry : orderings.get(issuesSize - 2).entrySet()) {
                for (int i = 0; i < entry.getValue().size(); i++) {
                    for (int j = i + 1; j < entry.getValue().size(); j++) {
                        ComparisonObject comp1 = entry.getValue().get(i);
                        ComparisonObject comp2 = entry.getValue().get(j);
                        if (comp1.isComparable(comp2)) {
                            int lowerIssue = 0;
                            int greaterIssue = 0;
                            List<Issue> rank = new ArrayList<>();
                            if (comp1.getValueIndex(1) < comp2.getValueIndex(1) &&
                                    comp1.getValueIndex(0) < comp2.getValueIndex(0)
                            ) continue;
                            else if (comp1.getValueIndex(1) > comp2.getValueIndex(1)) {
                                rank.add(issues.get(comp1.issues.get(1)));
                                rank.add(issues.get(comp2.issues.get(0)));
                                greaterIssue = comp1.issues.get(0);
                                lowerIssue = comp1.issues.get(1);
                            } else if (comp1.getValueIndex(0) > comp2.getValueIndex(0)) {
                                rank.add(issues.get(comp1.issues.get(0)));
                                rank.add(issues.get(comp2.issues.get(1)));
                                greaterIssue = comp1.issues.get(1);
                                lowerIssue = comp1.issues.get(0);
                            }
                            if (rank.size() == 2) {
                                List<Issue> reversedRank = reverseList(rank);

                                if (issueRankingList.contains(reversedRank)) {
                                    if ((Math.abs(comp1.getValueWeightOfIssue(greaterIssue) - comp2.getValueWeightOfIssue(greaterIssue)
                                            / Math.abs(comp1.getValueWeightOfIssue(lowerIssue) - comp2.getValueWeightOfIssue(lowerIssue)))) < 1) {
                                        issueRankingList.remove(reversedRank);
                                        issueRankingList.add(rank);
                                    }
                                } else if (!issueRankingList.contains(rank)) issueRankingList.add(rank);
                            }
                        }
                    }
                }
            }

            for (List<Issue> comparison : issueRankingList) {
                int index1 = issueRanking.indexOf(comparison.get(0));
                int index2 = issueRanking.indexOf(comparison.get(1));

                if (index1 > index2) {
                    Collections.swap(issueRanking, index1, index2);
                }
            }
        }

        List<Double> expectedIssueWeight = new ArrayList<>(Collections.nCopies(issuesSize, 0d));

        int issueMid = issuesSize / 2;
        if (issuesSize % 2 == 0) issueMid--;
        double difference = 1 / (double) issuesSize;
        expectedIssueWeight.set(issueMid, difference);
        for (int i = 1; i <= issuesSize / 2; i++) {
            if (issueMid + i < issuesSize) {
                difference = (expectedIssueWeight.get(issueMid + i - 1) + expectedIssueWeight.get(issueMid + i - 1) / (double) issuesSize);
                expectedIssueWeight.set(issueMid + i, difference);
            }
            if (issueMid - i >= 0) {
                difference = (expectedIssueWeight.get(issueMid - i + 1) - expectedIssueWeight.get(issueMid - i + 1) / (double) issuesSize);
                expectedIssueWeight.set(issueMid - i, difference);
            }
        }


        double[] utilitiesPerBid = new double[bidOrder.size()];
        double[][] encodedBids = encodeBids(bidOrder, valuesCount, values);

        HashSet<Bid> allBidsSet = new HashSet<>(outcomespace.getAllBidsWithoutUtilities());
        List<Bid> allBids = new ArrayList<>(allBidsSet);

        double[][] allBidsEncoded = encodeListOfStrings(allBids, valuesCount);

        double highestUtil = info.getUserModel().getBidRanking().getHighUtility();
        double lowestUtil = info.getUserModel().getBidRanking().getLowUtility();

        utilitiesPerBid[0] = lowestUtil;
        utilitiesPerBid[utilitiesPerBid.length - 1] = highestUtil;

        double delta = highestUtil - lowestUtil;
        double decrementAmount = delta / (utilitiesPerBid.length - 1);

        for (int i = 1; i < utilitiesPerBid.length - 1; i++) {
            utilitiesPerBid[i] = utilitiesPerBid[i - 1] + decrementAmount;
        }


        Map<Bid, Double> allBidsRegressionReversed = new HashMap<>();

        if (bidOrder.size() > 23) {
            OLSMultipleLinearRegression regression = new OLSMultipleLinearRegression();
            regression.newSampleData(utilitiesPerBid, encodedBids);

            double[] allBidPredictions = new double[allBids.size()];

            for (int z = 0; z < allBids.size(); z++) {
                allBidPredictions[z] = predict(regression, allBidsEncoded[z]);
            }

            double boostAmount = 0;
            Arrays.sort(allBidPredictions);
            if (allBidPredictions[0] < 0) {
                boostAmount = Math.abs(allBidPredictions[0]);
            }
            double maxWeight = allBidPredictions[allBidPredictions.length-1] + boostAmount;

            Map<Double, List<Bid>> allBidsRegression = new HashMap<>();
            allBidsRegressionReversed = new HashMap<>();
            int z = 0;
            for (Bid x : allBids) {
                List<Bid> tmp = new ArrayList<>();
                tmp.add(x);
                double val = (predict(regression, allBidsEncoded[z++]) + boostAmount) / maxWeight;
                if (allBidsRegression.containsKey(val)) allBidsRegression.get(val).add(x);
                else allBidsRegression.put(val, tmp);
                allBidsRegressionReversed.put(x, val);
            }
        }


        Set<Bid> bidSet = new HashSet<>(outcomespace.getAllBidsWithoutUtilities());
        List<Bid> bidList5 = new ArrayList<>(bidSet);
        utilBidMap = new HashMap<>();
        reversedBidMap = new HashMap<>();

        System.out.println("issues:" + issueRanking);
        System.out.println("vals: " + comparisonPerIssue);

        for (Bid x : bidList5) {
            List<Value> ignored = new ArrayList<>(x.getValues().values());
            List<IssueValuePair> bidList = new ArrayList<>();
            for (int i = 0; i < ignored.size(); i++) {
                IssueValuePair finalListVal = pairMapList.get(i).get(ignored.get(i));
                bidList.add(finalListVal);
            }
            double result = 0;
            for (IssueValuePair item : bidList) {
                result += (expectedIssueWeight.get(issueRanking.indexOf(issues.get(item.first))))
                        * ((expectedWeightsPerValueOrdering.get(item.first).get(comparisonPerIssue.get(item.first).indexOf(item))));
            }

            if (bidOrder.size() > 23) {
                result += allBidsRegressionReversed.get(x);
                result = result / 2;
            }

            if (utilBidMap.containsKey(result))
                utilBidMap.get(result).add(x);
            else {
                List<Bid> tmp = new ArrayList<>();
                tmp.add(x);
                utilBidMap.put(result, tmp);
            }

            reversedBidMap.put(x, result);
        }

        utilBidMapKeys = new ArrayList<>(utilBidMap.keySet());
        Collections.sort(utilBidMapKeys);

        List<Double> keyPrime = new ArrayList<>();

        for (int i = utilBidMapKeys.size() - 1; i > 0; i--) {
            if (utilBidMapKeys.get(i) < 0.9d) break;
            keyPrime.add(utilBidMapKeys.get(i));
        }

        phase2Bids = new ArrayList<>();

        int interval = keyPrime.size() / 20;

        if (keyPrime.size() < 100) {
            interval = 1;
            isConcession = true;
        }

        for (int i = 0; i < keyPrime.size(); i += interval) {
            phase2Bids.add(utilBidMap.get(keyPrime.get(i)).get(0));
        }

        Collections.sort(utilBidMapKeys);

        sortedBids = new ArrayList<>();
        for (Double x: utilBidMapKeys) {
            sortedBids.addAll(utilBidMap.get(x));
        }

        double x1 = (double) bidOrder.size();
        double x2 = (double) allBids.size();
        double ratio =  x1 / x2;
        System.out.println("rat: " + ratio + " bidordersize: " + bidOrder.size() + " allbid size: " + allBids.size());
        phase1Bound = 0.55 + (0.1d - ratio) * 3.64d;

        if (phase1Bound < 0.75d)
            phase1Bound = 0.75d;

        System.out.println(phase1Bound);

        phase1Bids = new ArrayList<>();
        for (Double x: utilBidMapKeys) {
            if (x > phase1Bound) {
                phase1Bids.addAll(utilBidMap.get(x));
            }
        }

        phase2Bids = new ArrayList<>();
        for (Double x: utilBidMapKeys) {
            if (x > phase1Bound) {
                phase2Bids.addAll(utilBidMap.get(x));
            }
        }

        phase2Index = phase1Bids.size() - 1;

        System.out.println(sortedBids);

    }

    @Override
    public Action chooseAction(List<Class<? extends Action>> possibleActions) {
        if (getLastReceivedAction() instanceof Offer) {
            lastReceivedBid = ((Offer) getLastReceivedAction()).getBid();
        }

        if (previousBids == null || !possibleActions.contains(Accept.class)) {
            isFirst = true;
            return new Offer(getPartyId(), utilBidMap.get(utilBidMapKeys.get(utilBidMapKeys.size() - 1)).get(0));
        }

        if (lastReceivedBid != null && timeline.getCurrentTime() <= (timeline.getTotalTime() - 1) * 0.95) {
            if (previousOpponentBidsMap.containsKey(lastReceivedBid))
                previousOpponentBidsMap.put(lastReceivedBid, previousOpponentBidsMap.get(lastReceivedBid) + 1);
            else previousOpponentBidsMap.put(lastReceivedBid, 1d);

            opponentValues.addAll(lastReceivedBid.getValues().values());
        }
        if (timeline.getCurrentTime() == (timeline.getTotalTime() - 1) * 0.98) {
            double maxOpponentWeight = Collections.max(previousOpponentBidsMap.entrySet(),
                    Comparator.comparing(Entry::getValue)).getValue();
            for (Bid x : previousOpponentBidsMap.keySet()) {
                previousOpponentBidsMap.replace(x, previousOpponentBidsMap.get(x) / maxOpponentWeight);
            }

            if (previousOpponentBidsMap.size() >= valuesCount && opponentValues.size() >= valuesCount) {
                double[] utilitiesPerBid = new double[previousOpponentBidsMap.values().size()];
                int i = 0;
                for (double v : previousOpponentBidsMap.values()) {
                    utilitiesPerBid[i++] = v;
                }

                Set<Bid> bidSet = new HashSet<>(outcomespace.getAllBidsWithoutUtilities());
                List<Bid> allBidList = new ArrayList<>(bidSet);

                double[][] encodedBids = encodeBids(new ArrayList<>(previousOpponentBidsMap.keySet()), valuesCount, values);
                double[][] oneHotEncodedAll = encodeListOfStrings(allBidList, valuesCount);

                OLSMultipleLinearRegression regression = new OLSMultipleLinearRegression();
                regression.newSampleData(utilitiesPerBid, encodedBids);

                Map<Bid, Double> allOpponentBids = new HashMap<>();

                int j = 0;
                for (Bid x : allBidList) {
                    allOpponentBids.put(x, predict(regression, oneHotEncodedAll[j++]));
                }


                for (Entry<Bid, Double> entry : allOpponentBids.entrySet()) {
                    double nashVal = entry.getValue() * reversedBidMap.get(entry.getKey());
                    nashBids.put(nashVal, entry.getKey());
                }

                nashBid = nashBids.get(Collections.max(nashBids.keySet()));
                if (nashBid != null) nashFound = true;
                for (Entry<Double, Bid> entry : nashBids.entrySet()) {
                    if (reversedBidMap.get(entry.getValue()) > 0.90d) {
                        bidsAfterNash.add(entry.getValue());
                    }
                }
            }

            else {
                for (Entry<Bid, Double> entry : previousOpponentBidsMap.entrySet()) {
                    double nashVal = entry.getValue() * reversedBidMap.get(entry.getKey());
                    nashBids.put(nashVal, entry.getKey());
                }

                nashBid = nashBids.get(Collections.max(nashBids.keySet()));
                for (Entry<Double, Bid> entry : nashBids.entrySet()) {
                    if (reversedBidMap.get(entry.getValue()) > 0.70d) {
                        bidsAfterNash.add(entry.getValue());
                    }
                }
            }
        }

        previousBid = getBid();

        if (ourMinBid == null || reversedBidMap.get(previousBid) < reversedBidMap.get(ourMinBid))
            ourMinBid = previousBid;

        if (reversedBidMap.get(previousBid) <= reversedBidMap.get(lastReceivedBid))
            return new Accept(getPartyId(), lastReceivedBid);

        if (reversedBidMap.get(lastReceivedBid) >= phase1Bound)
            return new Accept(getPartyId(), lastReceivedBid);

        if (isFirst && timeline.getCurrentTime() >= timeline.getTotalTime() - 2) {
            if (reversedBidMap.get(maxBid) >= phase1Bound && !nashFound && reversedBidMap.get(maxBid) > reversedBidMap.get(nashBid))
                return new Offer(getPartyId(), maxBid);
            else if (nashFound && reversedBidMap.get(nashBid) >= 0.70d)
                return new Offer(getPartyId(), nashBid);
            else return new Offer(getPartyId(), getBid());
        }


        double acceptableUtil = 0.80;


        if (maxBid == null || reversedBidMap.get(lastReceivedBid) > reversedBidMap.get(maxBid))
            maxBid = lastReceivedBid;


        if (reversedBidMap.get(maxBid) >= 0.80 && !isFirst && timeline.getCurrentTime() == timeline.getTotalTime() - 2) {
            if (!nashFound && reversedBidMap.get(nashBid) < 0.70d)
                return new Offer(getPartyId(), maxBid);
            else return new Offer(getPartyId(), nashBid);
        }
        if (!isFirst && timeline.getCurrentTime() == timeline.getTotalTime() - 1 && acceptableUtil <= reversedBidMap.get(lastReceivedBid)) {
            return new Accept(getPartyId(), lastReceivedBid);
        } else {
            return new Offer(getPartyId(), previousBid);
        }
    }

    private Bid getBid() {
        double time = timeline.getCurrentTime();
        double firstPhase = timeline.getTotalTime() * 0.3;
        double secondPhase = timeline.getTotalTime() - 3;

        if (time <= firstPhase) {
            int bidIndex = rand.nextInt(phase1Bids.size());
            return phase1Bids.get(bidIndex);

        } else if (time > firstPhase && secondPhase >= time) {
            bidGivenCount++;
            phase2Index++;

            if (phase2Bids.size() == phase2Index) {
                phase2Index = 0;
            }

            return phase2Bids.get(phase2Index);

        } else if (time >= secondPhase) {
            return phase1Bids.get(0);
        }

        return sortedBids.get(sortedBids.size()-1);
    }

    /**
     * With this method, you can override the default estimate of the utility
     * space given uncertain preferences specified by the user model. This
     * example sets every value to zero.
     */
    @Override
    public AbstractUtilitySpace estimateUtilitySpace() {
        return new AdditiveUtilitySpaceFactory(getDomain()).getUtilitySpace();
    }

    @Override
    public String getDescription() {
        return "An agent that solves uncertainty with a special solver";
    }

    private double[][] encodeBids(List<Bid> bidOrder, int valueCount, List<List<ValueDiscrete>> allIssues) {
        double[][] oneHotEncoded = new double[bidOrder.size()][valueCount];
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

    private double predict(OLSMultipleLinearRegression regression, double[] x) {
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
}

