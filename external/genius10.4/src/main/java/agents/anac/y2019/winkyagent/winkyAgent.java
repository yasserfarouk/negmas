package agents.anac.y2019.winkyagent;

import java.util.*;

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
import genius.core.timeline.DiscreteTimeline;
import genius.core.uncertainty.BidRanking;

public class winkyAgent extends AbstractNegotiationParty {

    private Bid lastReceivedBid = null;
    private Map<Bid, Double> receiveBids = new HashMap<Bid, Double>();
    private List<Bid> bidOrder = null;
    int utilitySize = 0;
    int ranklistSize = 0;
    double receivehighestUtility = 0.0;//接收过的最高出价的效用
    List<Issue> issueList = null;//issue列表
    int issueSize = 0;//issue个数
    int valueSum = 0;//value个数
    double initUtility = 0.0;//value初始效用
    Map<ValueDiscrete, Double> valueCorrespond = new HashMap<ValueDiscrete, Double>();//value和对应效用
    ValueDiscrete[] values = null;//value数组
    double learningRate;
    List<Map.Entry<Bid, Double>> list = new ArrayList<>();//对receiveBids按照效用进行排序后得到的list
    boolean listSort = true;
    boolean lastBidTag = true;

    @Override
    public void init(NegotiationInfo info) {

        super.init(info);
        utilitySize = (int) utilitySpace.getDomain().getNumberOfPossibleBids();//一共可能有多少种出价
        bidOrder = userModel.getBidRanking().getBidOrder();
        ranklistSize = bidOrder.size();
        issueList = utilitySpace.getDomain().getIssues();
        issueSize = issueList.size();
        double[] results = new double[ranklistSize];//给定bid效用数组
        for (int i = 0; i < ranklistSize; i++) {    //results数组初始化赋值
            results[i] = getBidOrderUtility(bidOrder.get(i));
        }

        int[] valueSize = new int[issueSize];//第i个问题有j种选择
        for (int i = 0; i < issueSize; i++) {
            Issue issue = issueList.get(i);
            IssueDiscrete issued = (IssueDiscrete) issue;//某个issue的value
            valueSize[i] = issued.getNumberOfValues();
            valueSum += valueSize[i];
        }
        initUtility = 1.0 / valueSum; //value初始化的值
        learningRate = initUtility / 10.0;

        values = new ValueDiscrete[valueSum];//value数组
        int valuesIndexCnt = 0;
        while (valuesIndexCnt < valueSum) {     //初始化values数组和map valueCorrespond
            for (int i = 0; i < issueSize; i++) {
                Issue issue = issueList.get(i);
                IssueDiscrete issued = (IssueDiscrete) issue;//某个issue的value
                for (int j = 0; j < issued.getNumberOfValues(); j++) {
                    values[valuesIndexCnt] = issued.getValue(j);      //初始化values数组
                    valueCorrespond.put(values[valuesIndexCnt], initUtility);    //初始化map valueCorrespond
                    valuesIndexCnt++;
                }
            }
        }

        int[][] features = new int[ranklistSize][valueSum];//bidOrder训练集
        for (int i = 0; i < ranklistSize; i++) {
            HashMap<Integer, Value> valueHashMap = bidOrder.get(i).getValues();
            int vhmSize = valueHashMap.size();
//            for(int z=1;z<=vhmSize;z++){
//                log(z+" "+valueHashMap.get(z));
//            }
            int p = 1;
            for (int j = 0; j < valueSum; j++) {
                Value valueTemp = values[j];
                Value valueOfbidOrder = valueHashMap.get(p);
                if (valueTemp.equals(valueOfbidOrder) && p <= vhmSize) {
                    features[i][j] = 1;
                    p++;
                } else {
                    features[i][j] = 0;
                }
            }
//            log("\n");
        }

        double[] parameters = new double[valueSum];//训练得到的value值
        for (int i = 0; i < valueSum; i++) {
            parameters[i] = initUtility;
        }

        for (int i = 0; i < ranklistSize * valueSum; i++) {       //训练
            BGD(features, results, learningRate, parameters);
        }
    }


    private void BGD(int[][] features, double[] results, double learningRate, double[] parameters) {
        for (int t = 0; t < valueSum; t++) {
            double sum = 0.0;
            double parametersSum = 0.0;
            for (int j = 0; j < results.length; j++) {
                for (int i = 0; i < valueSum; i++) {
                    parametersSum += parameters[i] * features[j][i];
                }
                parametersSum = parametersSum - results[j];
                parametersSum = parametersSum * features[j][t];
                sum += parametersSum;
            }
            double updateValue = 2 * learningRate * sum / results.length;
            parameters[t] = parameters[t] - updateValue;
            valueCorrespond.put(values[t], parameters[t]);

        }
//        double totalLoss = 0;
//        for (int j = 0; j < results.length; j++) {
//            totalLoss = totalLoss + Math.pow((parameters[0] * features[j][0] + parameters[1] * features[j][1]
//                    + parameters[2] * features[j][2] + parameters[3] - results[j]), 2);
//        }
//        System.out.println(parameters[0] + " " + parameters[1] + " " + parameters[2] + " " + parameters[3]);
//        System.out.println("totalLoss:" + totalLoss);
    }

    private double linearEstUtility(Bid bid) {
        double linearUtility = 0.0;
        HashMap<Integer, Value> valueHashMap = bid.getValues();
        int vhmSize = valueHashMap.size();
        int p = 1;
        for (int j = 0; j < valueSum; j++) {
            Value valueTemp = values[j];
            Value valueOfbidOrder = valueHashMap.get(p);
            if (valueTemp.equals(valueOfbidOrder) && p <= vhmSize) {
                linearUtility += valueCorrespond.get(valueTemp);
                p++;
            }
        }
        return linearUtility;
    }

    private double getBidOrderUtility(Bid bid)  //估计已知出价效用，等分
    {
        BidRanking bidRanking = getUserModel().getBidRanking();
        Double min = bidRanking.getLowUtility();
        double max = bidRanking.getHighUtility();

        int i = bidOrder.indexOf(bid);

        // index:0 has utility min, index n-1 has utility max
        return min + i * (max - min) / (double) (ranklistSize - 1);
    }


    @Override
    public Action chooseAction(List<Class<? extends Action>> validActions) {

        int round = ((DiscreteTimeline) timeline).getRound();
        int tround = ((DiscreteTimeline) timeline).getTotalRounds();
        double receiveBidUtility = 0.0;
        double bidOrderMax = userModel.getBidRanking().getHighUtility();
        Bid bid;
        if (round < tround * 0.7) {
            if (round > 10 && receiveBids.size() < 7) {
                int temp = (int) Math.ceil(ranklistSize * 0.1);
                int randz = rand.nextInt(temp);
                bid = bidOrder.get(ranklistSize - 1 - randz);
                log("receiveBid<7,bidOrder: " + getBidOrderUtility(bid));
                return new Offer(getPartyId(), bid);
            }
            bid = generateBid(7, bidOrderMax);
            return new Offer(getPartyId(), bid);
        } else if (round < tround * 0.98) {
            if (receiveBids.size() < 10) {
                int temp = (int) Math.ceil(ranklistSize * 0.15);
                int randz = rand.nextInt(temp);
                bid = bidOrder.get(ranklistSize - 1 - randz);
                log("receiveBid<10,bidOrder: " + getBidOrderUtility(bid));
                return new Offer(getPartyId(), bid);
            }
            bid = generateBid(9, bidOrderMax);
            return new Offer(getPartyId(), bid);
        } else if (round < tround * 0.99) {
            receiveBidUtility = linearEstUtility(lastReceivedBid);
            if (listSort) {
                sortReceive();
                listSort = false;
                for (Map.Entry<Bid, Double> entry : list) {
                    System.out.println(entry);
                }
                log(receivehighestUtility + "\n");
            }
            if (receiveBidUtility > (receivehighestUtility - 0.03)) {
                return new Accept(getPartyId(), lastReceivedBid);
            }
            bid = generateReceiveBid();
            log("receive bid Utility: " + linearEstUtility(lastReceivedBid) + " accept阈值: " + (receivehighestUtility - 0.07) + "\n");
            return new Offer(getPartyId(), bid);
        } else if (round < tround * 0.995) {
            receiveBidUtility = linearEstUtility(lastReceivedBid);
            if (receiveBidUtility > (receivehighestUtility - 0.07)) {
                return new Accept(getPartyId(), lastReceivedBid);
            }
            bid = generateReceiveBid();
            log("receive bid Utility: " + linearEstUtility(lastReceivedBid) + " accept阈值: " + (receivehighestUtility - 0.11) + "\n");
            return new Offer(getPartyId(), bid);
        } else if (round == (tround-1)) {
            return new Accept(getPartyId(), lastReceivedBid);

        } else {
            receiveBidUtility = linearEstUtility(lastReceivedBid);
            if (receiveBidUtility > (receivehighestUtility - 0.1)) {
                return new Accept(getPartyId(), lastReceivedBid);
            }
            bid = generateReceiveBid();
            log("receive bid Utility: " + linearEstUtility(lastReceivedBid) + " accept阈值: " + (receivehighestUtility - 0.15) + "\n");
            return new Offer(getPartyId(), bid);
        }
    }


    public Bid generateBid(int zcnt, double bidOrderMax) {
        Bid randomBid = null;
        if (lastReceivedBid == null) {
            randomBid = userModel.getBidRanking().getMaximalBid();
        } else if (zcnt == 7) {
            if (bidOrderMax > 0.9) {
                do {
                    randomBid = generateRandomBid();
                } while (linearEstUtility(randomBid) < 0.82);
            } else if (bidOrderMax > 0.8) {
                do {
                    randomBid = generateRandomBid();
                } while (linearEstUtility(randomBid) < 0.75);
            } else {
                do {
                    randomBid = generateRandomBid();
                } while (linearEstUtility(randomBid) < 0.7);
            }
        } else if (zcnt == 9) {
            if (bidOrderMax > 0.9) {
                do {
                    randomBid = generateRandomBid();
                } while (linearEstUtility(randomBid) < 0.8);
            } else if (bidOrderMax > 0.8) {
                do {
                    randomBid = generateRandomBid();
                } while (linearEstUtility(randomBid) < 0.7);
            } else {
                do {
                    randomBid = generateRandomBid();
                } while (linearEstUtility(randomBid) < 0.68);
            }
        }

        log(((DiscreteTimeline) timeline).getRound() + "generateBid: " + linearEstUtility(randomBid) + "\n");
        return randomBid;
    }

    private Bid generateReceiveBid() {
        Bid bid;

        int listSelectUtility = (int) Math.ceil(list.size() * 0.03);
        double temp = list.get(listSelectUtility - 1).getValue();

        if (temp < 0.7) {
            temp = 0.7;
            do {
                bid = generateRandomBid();
            } while (linearEstUtility(bid) < temp);
            log(((DiscreteTimeline) timeline).getRound() + " generateRandomBid: " + linearEstUtility(bid) + " temp:" + temp);
            return bid;
        } else {
            if (lastBidTag) {
                int rand1 = rand.nextInt(listSelectUtility);
                bid = list.get(rand1).getKey();
                lastBidTag = false;
                log(((DiscreteTimeline) timeline).getRound() + " generateReceiveBid: " + linearEstUtility(bid) + " temp:" + temp);
                return bid;
            } else {
                do {
                    bid = generateRandomBid();
                } while (linearEstUtility(bid) < temp);
                lastBidTag = true;
                log(((DiscreteTimeline) timeline).getRound() + " generateRandomBid: " + linearEstUtility(bid) + " temp:" + temp);
                return bid;
            }
        }

    }

    @Override
    public void receiveMessage(AgentID sender, Action action) {
        super.receiveMessage(sender, action);
        if (action instanceof Offer) {
            lastReceivedBid = ((Offer) action).getBid();
            double lastReceivedBidUtility = linearEstUtility(lastReceivedBid);
            receiveBids.put(lastReceivedBid, lastReceivedBidUtility);
            if (lastReceivedBidUtility > receivehighestUtility) {
                receivehighestUtility = lastReceivedBidUtility;
            }
        }
    }

    private void sortReceive() { //将收到的出价进行排序
        for (Map.Entry<Bid, Double> entry : receiveBids.entrySet()) {
            list.add(entry); //将map中的元素放入list中
        }

        list.sort(new Comparator<Map.Entry<Bid, Double>>() {
            @Override
            public int compare(Map.Entry<Bid, Double> o1, Map.Entry<Bid, Double> o2) {
                double result = o2.getValue() - o1.getValue();
                if (result > 0)
                    return 1;
                else if (result == 0)
                    return 0;
                else
                    return -1;
            }
            //逆序（从大到小）排列，正序为“return o1.getValue()-o2.getValue”
        });
    }

    @Override
    public String getDescription() {
        return "ANAC2019";
    }

    private static void log(String s) {
        System.out.println(s);
    }

}

