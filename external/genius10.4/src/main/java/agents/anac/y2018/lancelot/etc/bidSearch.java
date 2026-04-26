package agents.anac.y2018.lancelot.etc;

import java.util.List;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
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
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AbstractUtilitySpace;


public class bidSearch {
    private boolean DEBUG = true;
    private AbstractUtilitySpace utilitySpace;
    private TimeLineInfo timeLine;
    private ArrayList<HashMap<Value, Integer>> bidTables;
    private int cnt = 0;
    private int issue_num = -1;

    public bidSearch(AbstractUtilitySpace utilitySpace, TimeLineInfo timeLine){
        this.utilitySpace = utilitySpace;
        this.timeLine = timeLine;
        List<Issue> issues = utilitySpace.getDomain().getIssues();
        this.issue_num = issues.size();
        bidTables = new ArrayList<>();
        createBidTables();
    }

    public Bid maxBid(){
        try {
            Bid max_bid = utilitySpace.getMaxUtilityBid();
            System.out.println("max_bid = " + max_bid);
            return max_bid;
        } catch (Exception e){
            System.out.println("最大効用値のbidを探索できませんでした．");
            return getRandomBid((double) 0.8);
        }
    }

    public Bid offerOppositeBid(){
        Bid offer_bid = getRandomBid(0);
        Random rand = new Random();
        for(int i=0; i<issue_num; i++){
            HashMap<Value,Integer> bidTable = bidTables.get(i);
            ArrayList<Value> sorted_list = sortTable(bidTable,0);
            int list_num = sorted_list.size();
            try {
                int idx = i+1;
                offer_bid = offer_bid.putValue(idx, sorted_list.get(rand.nextInt((int)(list_num/2))));
            } catch (Exception e){
                System.out.println("offer_bidを更新できませんでした．");

            }
        }
//        System.out.println("offer_opposite_bid = " + offer_bid);
        return offer_bid;
    }

    public Bid offerPositiveBid(double min_util){
        Bid offer_bid = null;
        try {
            offer_bid = getRandomBid(min_util);
        } catch(Exception e){
            System.out.println("min_utilの値以上のBidがありません．");
            offer_bid = getRandomBid(0);
        }
        Bid print_bid = offer_bid;
        Random rand = new Random();
        for(int i=0; i<issue_num; i++) {
            if (rand.nextInt(5) % 2 == 0) {
                Bid replaced_bid = offer_bid;
                HashMap<Value, Integer> bitTable = bidTables.get(i);
                int sort_order = 1;
                ArrayList<Value> sorted_list = sortTable(bitTable, sort_order);
                for (int j = 0; j < sorted_list.size(); j++) {
                    Value value = sorted_list.get(j);
                    try {
//                        System.out.println("Value: value = " + value);
                        int idx = i + 1;
//                        System.out.println("issueId = " + idx);
//                        System.out.println("replaced_bid = " + replaced_bid);
                        replaced_bid = replaced_bid.putValue(idx, value);
//                        System.out.println("replaced_bid = " + replaced_bid);
                    } catch (Exception e) {
                        System.out.println("replaced_bidを変更できませんでした．");
                    }
//                    double util = utilitySpace.getUtility(replaced_bid);
                    double util = utilitySpace.getUtilityWithDiscount(replaced_bid,timeLine.getTime());
//                    double origin_util = utilitySpace.getUtility(offer_bid);
                    double origin_util = utilitySpace.getUtilityWithDiscount(offer_bid,timeLine.getTime());
//                    System.out.println("replaced_bid = " + replaced_bid);
//                    System.out.println(("offer_bid = " + offer_bid));
                    if (util - origin_util > 0) {
                        offer_bid = replaced_bid;
                        break;
                    }
                }
            }
        }
        return offer_bid;
    }

    private ArrayList<Value> sortTable(HashMap<Value,Integer> bidTable,int sort_order){
        List<Map.Entry<Value,Integer>> list_entries = new ArrayList<Map.Entry<Value, Integer>>(bidTable.entrySet());
        ArrayList<Value> value_list = new ArrayList<>();
        if(sort_order == 0) {
            Collections.sort(list_entries, new Comparator<Map.Entry<Value, Integer>>() {
                @Override
                public int compare(Map.Entry<Value, Integer> o1, Map.Entry<Value, Integer> o2) {
                    return o1.getValue().compareTo(o2.getValue());
                }
            });
            Random rand = new Random();
            for (Map.Entry<Value, Integer> entry : list_entries) {
                value_list.add(entry.getKey());
            }
        } else{
            Collections.sort(list_entries, new Comparator<Map.Entry<Value, Integer>>() {
                @Override
                public int compare(Map.Entry<Value, Integer> o1, Map.Entry<Value, Integer> o2) {
                    return o2.getValue().compareTo(o1.getValue());
                }
            });
            Random rand = new Random();
            for (Map.Entry<Value, Integer> entry : list_entries) {
                value_list.add(entry.getKey());
            }
        }
        return value_list;
    }

    private void createBidTables(){
        List<Issue> issues = utilitySpace.getDomain().getIssues();
        for(Issue issue : issues){
            HashMap<Value,Integer> bidTable = new HashMap<>();
            IssueDiscrete issueDiscrete = (IssueDiscrete) issue;
            List<ValueDiscrete> value_list = issueDiscrete.getValues();
            for (Value value : value_list){
                bidTable.put(value,0);
            }
            bidTables.add(bidTable);
        }
    }

    private void printBidTables(){
        for(int i=0; i<issue_num; i++){
            HashMap<Value,Integer> bidTable = bidTables.get(i);

            System.out.println("bidTable = " + bidTable);
        }
    }

    public void updateBidTable(Bid recievedBid){
        for (int i = 0; i < issue_num; i++) {
            HashMap<Value,Integer> bidTable = bidTables.get(i);
            Value key = recievedBid.getValue(i + 1);
            if (bidTable.get(key) != null) {
                int value_num = (int) bidTable.get(key);
                bidTable.put(key, value_num + 1);
            } else {
                bidTable.put(key, 1);
            }
        }
        cnt++;
        if(cnt % 50 == 0) {
            printBidTables();
        }
    }

    public Bid getRandomBid(double minUtil){
        double max_util = 0;
        try{
            max_util = utilitySpace.getUtilityWithDiscount(utilitySpace.getMaxUtilityBid(), timeLine.getTime());
        }catch (Exception e){
            System.out.println("max_utilを得られませんでした．");
        }
        minUtil = Math.min(minUtil,max_util);
        if(minUtil == max_util){
            minUtil -= 0.1;
        }
        HashMap<Integer, Value> values = new HashMap<Integer, Value>(); // pairs
        // <issuenumber,chosen
        // value
        // string>
        List<Issue> issues = utilitySpace.getDomain().getIssues();
        Random randomnr = new Random();

        // create a random bid with utility>MINIMUM_BID_UTIL.
        // note that this may never succeed if you set MINIMUM too high!!!
        // in that case we will search for a bid till the time is up (3 minutes)
        // but this is just a simple agent.
        Bid bid = null;
        do {
            for (Issue lIssue : issues) {
                switch (lIssue.getType()) {
                    case DISCRETE:
                        IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
                        int optionIndex = randomnr.nextInt(lIssueDiscrete.getNumberOfValues());
                        values.put(lIssue.getNumber(), lIssueDiscrete.getValue(optionIndex));
                        break;
                    case REAL:
                        IssueReal lIssueReal = (IssueReal) lIssue;
                        int optionInd = randomnr.nextInt(lIssueReal.getNumberOfDiscretizationSteps() - 1);
                        values.put(lIssueReal.getNumber(),
                                new ValueReal(lIssueReal.getLowerBound()
                                        + (lIssueReal.getUpperBound() - lIssueReal.getLowerBound()) * (double) (optionInd)
                                        / (double) (lIssueReal.getNumberOfDiscretizationSteps())));
                        break;
                    case INTEGER:
                        IssueInteger lIssueInteger = (IssueInteger) lIssue;
                        int optionIndex2 = lIssueInteger.getLowerBound()
                                + randomnr.nextInt(lIssueInteger.getUpperBound() - lIssueInteger.getLowerBound());
                        values.put(lIssueInteger.getNumber(), new ValueInteger(optionIndex2));
                        break;
                }
            }
            bid = new Bid(utilitySpace.getDomain(), values);
//            System.out.println("aaa  " + minUtil + "   " + max_util);
        } while (utilitySpace.getUtilityWithDiscount(bid, timeLine.getTime()) < minUtil);

        return bid;
    }



//    public Bid proposition1(){
//        return ;
//    }
}
