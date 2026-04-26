package agents.anac.y2018.meng_wan;

import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AdditiveUtilitySpace;

public class Agent36 extends AbstractNegotiationParty {
    //可变的target utility
    Double changeTargetUtility=0.8D;
    private TimeLineInfo TimeLineInfo = null;
    //我的方法给offer调用的次数
    int DamonOfferTime=0;
    //时间格式化
    private double t1 = 0.0D;
    private double u2 = 1.0D;
    private final String description = "Example Agent";

    Bid lastBid;
    Action lastAction;
    String oppAName;
    String oppBName;
    int round;
    boolean Imfirst = false;
    Boolean withDiscount = null;
    boolean fornullAgent = false;
    ArrayList<BidUtility> opponentAB = new ArrayList();
    //保存大于0。7的offer
    ArrayList<BidUtility> opponentABC = new ArrayList();
    OpponentPreferences oppAPreferences = new OpponentPreferences();
    OpponentPreferences oppBPreferences = new OpponentPreferences();


    //初始化
    public void init(NegotiationInfo info) {
        super.init(info);
        this.TimeLineInfo = info.getTimeline();
    }

    //选择策略，当进入自己turn的时候，进入此方法；
    public Action chooseAction(List<Class<? extends Action>> validActions) {
        try {
            if (this.lastBid == null) {
                this.Imfirst = true;
                Bid b = getMybestBid(this.utilitySpace.getMaxUtilityBid()); //algorithm 2
                return new Offer(getPartyId(), b);
            }
            if(this.TimeLineInfo.getTime()>=0.95D){
                this.changeTargetUtility=0.75D;
            }
            if(this.TimeLineInfo.getTime()>0.98D){
                this.changeTargetUtility=0.7D;
            }
           System.out.println("MyUtiityValueMax:-"+getMyutility());
            //如果来了一个utility大于MyUtility的offer
            if (this.utilitySpace.getUtility(this.lastBid) >= getMyutility()) {
                //在前150秒不接受任何offer，这种缺点就是万一有一个sbagent在之前都接受这个offer，但是后来又不接受了，但一般不会
                if(this.TimeLineInfo.getTime()<0.9D){
                    //只有这个协议是其他俩都接受的时候，才来改变targetAction
                    if(this.lastAction instanceof  Accept) {
                        this.changeTargetUtility = this.utilitySpace.getUtility(this.lastBid);
                    }
                }
                //最后30秒开始接受offer
                else{
                    return new Accept(getPartyId(), this.lastBid);
                }
            }

            //如果时间0.8以上了，就增加一个策略：
            //看commonAccept是否空，不空就
            //从common里面挑选，有没有大于等于myutility的offer，有的话，抛出去
            if(this.TimeLineInfo.getTime()>= 0.9D) {
                if (opponentABC != null && opponentABC.size() != 0) {
                    Double maxU = 0D;
                    int indexI = 0;
                    for (int i = 0; i < opponentABC.size(); i++) {
                        if (opponentABC.get(i).utility > maxU) {
                            maxU = opponentABC.get(i).utility;
                            indexI = i;
                        }
                    }
                    Bid b;
                    //如果这个列表中的最高utility 大于 当前的MyUtility
                    if (opponentABC.get(indexI).utility >=getMyutility()) {
                        //一旦把此offer发出去，就从列表中删除
                        b = opponentABC.get(indexI).bid;
                        opponentABC.remove(opponentABC.get(indexI));
                        DamonOfferTime += 1;
                        System.out.println("MyMethodGiveOfferTime:-" + DamonOfferTime);
                    } else {
                        b = offerMyNewBid(false);
                    }

                    return new Offer(getPartyId(), b);
                }
            }
            //前150s 或者 来的offer的utility 小于 MyUtility，都执行下面代码
              Bid b = offerMyNewBid(false);
//            if (this.utilitySpace.getUtility(b) < getMyutility()) {
//                return new Offer(getPartyId(), getMybestBid(this.utilitySpace.getMaxUtilityBid()));
//            }
            return new Offer(getPartyId(), b);
        } catch (Exception e) {
            System.out.println("Error Occured " + e.getMessage());

            Bid mb = null;
            try {
                mb = this.utilitySpace.getMaxUtilityBid();
                return new Offer(getPartyId(), getMybestBid(mb));
            } catch (Exception e1) {
                try {
                    return new Offer(getPartyId(), mb);
                } catch (Exception localException1) {
                }
            }
        }
        return new Accept(getPartyId(), this.lastBid);
    }

    public void receiveMessage(AgentID sender, Action action) {
        super.receiveMessage(sender, action); //this.lastBid = ((Offer) arguments).getBid();
        String agentName = sender == null ? "null" : sender.toString();

        this.fornullAgent = (!this.fornullAgent); //好像没用？
        if ((action != null) && ((action instanceof Offer))) { //如果读到一个offer
            Bid newBid = ((Offer) action).getBid();
            try {
                //根据时间变化的utility 没用
                BidUtility opBid = new BidUtility(newBid, this.utilitySpace.getUtility(newBid),
                        System.currentTimeMillis()); //BidUtility为私有类，用来储存bid、该bid对Damon的utility、时间， 这个时间是干嘛的？

                //这下面将bid加入arraylist的操作是否有用？ 暂时没发现作用
                if ((this.oppAName != null) && (this.oppAName.equals(agentName))) {
                    addBidToList(this.oppAPreferences.getOpponentBids(), opBid); //如果是A对手给的offer，则添加到A对手的OpponentPreference类中的arraylist里
                } else if ((this.oppBName != null) && (this.oppBName.equals(agentName))) {
                    addBidToList(this.oppBPreferences.getOpponentBids(), opBid); //如果是B对手给的offer，则添加到B对手的OpponentPreference类中的arraylist里
                } else if (this.oppAName == null) { //如果没有A名称，则添加A名称，并添加bid
                    this.oppAName = agentName;
                    this.oppAPreferences.getOpponentBids().add(opBid);
                } else if (this.oppBName == null) {//如果没有B名称，则添加B名称，并添加bid
                    this.oppBName = agentName;
                    this.oppBPreferences.getOpponentBids().add(opBid);
                }


                calculateParamForOpponent(this.oppAName.equals(agentName) ? this.oppAPreferences : this.oppBPreferences,
                        newBid); //针对A或B执行calculateParamForOpponent().这个function是干嘛的？
                System.out.println("opp placed bid:" + newBid);
                this.lastBid = newBid;
            } catch (Exception e) {
                e.printStackTrace();
            }
            //如果读到一个accept
        } else if ((action != null) && ((action instanceof Accept))) {
            BidUtility opBid = null;
            try {
                opBid = new BidUtility(this.lastBid, this.utilitySpace.getUtility(this.lastBid),
                        System.currentTimeMillis());
            } catch (Exception e) {
                System.out.println("Exception  44" + e.getMessage());
            }
            addBidToList(this.opponentAB,opBid); //如果读到一个accept，则添加这个bid到AB共同bidList。
            //对于我大于0。7且 他们都接受的offer
            if(opBid.utility > 0.7D){
                opponentABC.add(opBid);
                System.out.println("OpponentABC-SIZE:-"+opponentABC.size());
            }
            //但是收到accept有可能是某个对手accept Damon的offer。需考虑怎么辨别是否是AB共同的。需辨别这个bid是否是Damon发出去的，如果不是，则AB同时同意这个offer，需加入list。
            //可以用Bid.equal(pBid)来比较
        }
        this.lastAction = action;
    }

    public int MyBestValue(int issueindex) {
        List<Issue> dissues = this.utilitySpace.getDomain().getIssues();
        Issue isu = (Issue) dissues.get(issueindex);
        HashMap<Integer, Value> map = new HashMap();
        double maxutil = 0.0D;
        int maxvalIndex = 0;
        try {
            map = this.utilitySpace.getMaxUtilityBid().getValues();
        } catch (Exception e) {
            e.printStackTrace();
        }
        if ((isu instanceof IssueDiscrete)) {
            IssueDiscrete is = (IssueDiscrete) isu;

            int num = 0;
            if (num < is.getNumberOfValues()) {
                map.put(new Integer(issueindex + 1), is.getValue(num));

                double u = 0.0D;
                try {
                    Bid temp = new Bid(this.utilitySpace.getDomain(), map);
                    u = this.utilitySpace.getUtility(temp);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                if (u > maxutil) {
                    maxutil = u;
                    maxvalIndex = num;
                }
            }
        } else if (((isu instanceof IssueInteger)) && (map != null)) {
            return ((ValueInteger) map.get(Integer.valueOf(issueindex + 1))).getValue();
        }
        return maxvalIndex;
    }

    //true开启50%策略， false不开启
    public Bid offerMyNewBid(boolean strategy) {
        Bid bidNN = null;
        if ((this.opponentAB != null) && (this.opponentAB.size() != 0)) {
            bidNN = getNNBid(this.opponentAB);
        }
        try {
            if ((bidNN == null) || (this.utilitySpace.getUtility(bidNN) < getMyutility())) {
                List<List<Object>> mutualIssues = getMutualIssues(strategy);
                //第一次map
                HashMap<Integer, Value> map1 = new HashMap();
                //第二次可能出现 X Y  Y X形式的map
                HashMap<Integer, Value> map2 = new HashMap();

                List<Issue> issues = this.utilitySpace.getDomain().getIssues();
                for (int i = 0; i < mutualIssues.size(); i++) {
                    //issueFrequency
                    List<Object> issueFrequency = (List) mutualIssues.get(i);

                    Issue issue = (Issue) issues.get(i);
                    if ((issue instanceof IssueDiscrete)) {
                        if (issueFrequency != null) {
                            IssueDiscrete discrete = (IssueDiscrete) issues.get(i);
                            //目的在于map里面只能放 issue.valuel类型，而不能直接放string ， 放入map1
                            for (int num = 0; num < discrete.getNumberOfValues(); num++) {
                                    if(issueFrequency.size()>4) {
                                        if (discrete.getValue(num).toString().equals(issueFrequency.get(4).toString())) {
                                            map2.put(new Integer(i + 1), discrete.getValue(num));
                                        }
                                    }
                                    if (discrete.getValue(num).toString().equals(issueFrequency.get(0).toString())) {
                                    map1.put(new Integer(i + 1), discrete.getValue(num));
                                    if(map2.get(new Integer(i+1))!=null) {
                                        map2.put(new Integer(i + 1), discrete.getValue(num));
                                    }
                                }
                            }
                            //X Y Y X形式 ,放入map 2
                            if(issueFrequency.size()>4){
                                for (int num = 0; num < discrete.getNumberOfValues(); num++) {
                                    if (discrete.getValue(num).toString().equals(issueFrequency.get(4).toString())) {
                                        map2.put(new Integer(i + 1), discrete.getValue(num));
                                        break;
                                    }
                                }

                            }
                        } else {
                            IssueDiscrete is = (IssueDiscrete) issues.get(i);
                            map1.put(new Integer(i + 1), is.getValue(MyBestValue(i)));
                            map2.put(new Integer(i + 1), is.getValue(MyBestValue(i)));
                        }
                    } else {
                        throw new IllegalStateException("not supported issue " + issue);
                    }
                }
                try {
                    Bid bid1 = new Bid(this.utilitySpace.getDomain(), (HashMap) map1.clone());
                    Bid bid2 = new Bid(this.utilitySpace.getDomain(), (HashMap) map2.clone());
                    if (this.utilitySpace.getUtility(bid1) >= getMyutility()) {
                        return bid1;
                    }else if(this.utilitySpace.getUtility(bid2) >= getMyutility()){
                        return bid2;
                    }
                    if(!strategy){
                        offerMyNewBid(true);
                    }

                    return getMybestBid(this.utilitySpace.getMaxUtilityBid());
                } catch (Exception e) {
                    System.out.println("Exception 55 " + e.getMessage());
                }
            }
            return bidNN;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    //获取其他agent共同偏爱的issue ,strategy为true就开启50%停止策略
    public List<List<Object>> getMutualIssues(boolean strategy) {
        //对手agent共同偏爱的list
        List<List<Object>> mutualList = new ArrayList();
        //该party对应的issue列表
        List<Issue> dissues = this.utilitySpace.getDomain().getIssues();
        //
        int onlyFirstFrequency=2;
        while(onlyFirstFrequency>0) {
            mutualList = new ArrayList();
            System.out.println("ISSUES SIZE:"+dissues.size());
            int count=0;
            for (int i = 0; i < dissues.size(); i++) {
                //防止第一次注入对手频率大的就超过50% 导致我们utility不够
                if(strategy) {
                    if (updateMutualList(mutualList, dissues, i, onlyFirstFrequency)) {
                        count += 1;
                    }
                    //超过50%就停止
                    if (count * 2 >= dissues.size()) {
                        break;
                    }
                }
                //一样？
                //如果对手agent的issue 的频率map为空，则进行下一次循环
                //也就是说这个是判断是不是对手还没给过offer？？
                if ((this.oppAPreferences.getRepeatedissue().get(dissues.get(i).getName()).size() == 0)
                        || (this.oppBPreferences.getRepeatedissue().get(dissues.get(i).getName())).size() == 0) {
                    return null;                                                
                }
            }
                         System.out.println("MutualList SIZE:"+mutualList.size());
            if (this.opponentAB.size() == 0) {
                float nullval = 0.0F;
                for (int i = 0; i < mutualList.size(); i++) {
                    if (mutualList.get(i) != null) {
                        nullval += 1.0F;
                    }
                }
                nullval /= mutualList.size();
                if (nullval >= 0.5D) {
                    break;
                }
            }
            onlyFirstFrequency--;
        }

        return mutualList;
    }

    //更新其他agent共同偏爱的List
    private boolean updateMutualList(List<List<Object>> mutualList, List<Issue> dissues,int i ,int onlyFirstFrequency) {
        //如果这个issue A之前抛出过
        if (this.oppAPreferences.getRepeatedissue().get(dissues.get(i).getName()) != null) {
            //此issue A抛出过的值的集合
            HashMap<Value, Integer> valsA = this.oppAPreferences.getRepeatedissue()
                    .get(dissues.get(i).getName());
            //此issue B抛出过的值的集合
            HashMap<Value, Integer> valsB =this.oppBPreferences.getRepeatedissue()
                    .get(dissues.get(i).getName());

            //A集合的key转换成数组
            Object[] keys = valsA.keySet().toArray();
            //最大值数组
            int[] maxA = new int[2];
            //最大值对应的key的数组
            Object[] maxkeyA = new Object[2];
            //对于此isuue 遍历A抛出过的值的数组
            for (int j = 0; j < keys.length; j++) {
                //temp临时储存 A的这个值抛出的次数
                Integer temp = (Integer) valsA.get(keys[j]);
                //即找最大的次数以及所对应的key
                if (temp.intValue() > maxA[0]) {
                    maxA[0] = temp.intValue();
                    maxkeyA[0] = keys[j];
                    //找第二大的次数以及所对应的key
                } else if (temp.intValue() > maxA[1]) {
                    maxA[1] = temp.intValue();
                    maxkeyA[1] = keys[j];
                }
            }
            //如果B 在此issue上 之前抛出过offer
            if (valsB != null) {
                //B集合的key转换成数组
                Object[] keysB = valsB.keySet().toArray();
                //同理
                int[] maxB = new int[2];
                Object[] maxkeyB = new Object[2];
                for (int j = 0; j < keysB.length; j++) {
                    Integer temp = (Integer) valsB.get(keysB[j]);
                    if (temp.intValue() > maxB[0]) {
                        maxB[0] = temp.intValue();
                        maxkeyB[0] = keysB[j];
                    } else if (temp.intValue() > maxB[1]) {
                        maxB[1] = temp.intValue();
                        maxkeyB[1] = keysB[j];
                    }
                }
                //如果是第一次循环
                if (onlyFirstFrequency == 2) {
                    //如果A，B俩的最高频率的值是一样的
                    if ((maxkeyA[0] != null) && (maxkeyB[0] != null) && (maxkeyA[0].equals(maxkeyB[0]))) {
                        ArrayList<Object> l = new ArrayList();
                        l.add(maxkeyA[0]);
                        l.add(Integer.valueOf(maxB[0]));
                        l.add(Integer.valueOf(maxA[0]));
                        //把第i个issue的共同列表中添加这个l结果
                        mutualList.add(i, l);
                        return true;
                    } else {
                        mutualList.add(i, null);
                        return false;
                    }
                    //如果是第二次循环
                } else {
                    boolean insideloop = true;
                    for (int m = 0; (insideloop) && (m < 2); m++) {
                        for (int n = 0; (insideloop) && (n < 2); n++) {
                            if ((maxkeyA[m] != null) && (maxkeyB[n] != null) && (maxkeyA[m].equals(maxkeyB[n]))) {
                                ArrayList<Object> l = new ArrayList();
                                l.add(maxkeyA[m]);
                                l.add(Integer.valueOf(maxB[n]));
                                l.add(Integer.valueOf(maxA[m]));
                                //感觉可以改进一下从B开始, 防止A：X Y  B： Y X 情况发生，而取不到Y
                                if ((maxkeyA[0] != null) && (maxkeyA[1] != null) && (maxkeyB[0] != null) && (maxkeyB[1] != null) && (maxkeyA[0].equals(maxkeyB[1])) && (maxkeyA[1].equals(maxkeyB[0]))) {
                                    l.add(maxkeyB[0]);
                                    l.add(Integer.valueOf(maxB[0]));
                                    l.add(Integer.valueOf(maxA[1]));
                                }
                                mutualList.add(i, l);
                                insideloop = false;
                            }
                        }
                    }
                    if (insideloop) {
                        mutualList.add(i, null);
                        return false;
                    }
                    return true;
                }

            } else {
                mutualList.add(i, null);
                this.oppBPreferences.getRepeatedissue().put(((Issue) dissues.get(i)).getName(), new HashMap());
                return false;
            }
        } else {

            this.oppAPreferences.getRepeatedissue().put(((Issue) dissues.get(i)).getName(), new HashMap());
            mutualList.add(i, null);
            return false;
        }
    }

    public Bid getNNBid(ArrayList<BidUtility> oppAB) {
        List<Issue> dissues = this.utilitySpace.getDomain().getIssues();
        Bid maxBid = null;
        double maxutility = 0.0D;
        int size = 0;
        int exloop = 0;
        while (exloop < dissues.size()) {
            int bi = chooseBestIssue();
            size = 0;
            while ((oppAB != null) && (oppAB.size() > size)) {
                Bid b = ((BidUtility) oppAB.get(size)).getBid();
                Bid newBid = new Bid(b);
                try {
                    HashMap vals = b.getValues();
                    vals.put(Integer.valueOf(bi), getRandomValue((Issue) dissues.get(bi - 1)));
                    newBid = new Bid(this.utilitySpace.getDomain(), vals);
                    if ((this.utilitySpace.getUtility(newBid) > getMyutility())
                            && (this.utilitySpace.getUtility(newBid) > maxutility)) {
                        maxBid = new Bid(newBid);
                        maxutility = this.utilitySpace.getUtility(maxBid);
                    }
                } catch (Exception e) {
                    System.out.println("Exception 66 " + e.getMessage());
                }
                size++;
            }
            exloop++;
        }
        return maxBid;
    }

    public int chooseBestIssue() {
        double random = Math.random();
        double sumWeight = 0.0D;
        AdditiveUtilitySpace utilitySpace1 = (AdditiveUtilitySpace) this.utilitySpace;
        for (int i = utilitySpace1.getDomain().getIssues().size(); i > 0; i--) {
            sumWeight += utilitySpace1.getWeight(i);
            if (sumWeight > random) {
                return i;
            }
        }
        return 0;
    }

    public int chooseWorstIssue() {
        double random = Math.random() * 100.0D;
        double sumWeight = 0.0D;
        int minin = 1;
        double min = 1.0D;
        AdditiveUtilitySpace utilitySpace1 = (AdditiveUtilitySpace) this.utilitySpace;
        for (int i = utilitySpace1.getDomain().getIssues().size(); i > 0; i--) {
            sumWeight += 1.0D / utilitySpace1.getWeight(i);
            if (utilitySpace1.getWeight(i) < min) {
                min = utilitySpace1.getWeight(i);
                minin = i;
            }
            if (sumWeight > random) {
                return i;
            }
        }
        return minin;
    }

    public Bid getMybestBid(Bid sugestBid) { //time没用到啊？
        List<Issue> dissues = this.utilitySpace.getDomain().getIssues();
        Bid newBid = new Bid(sugestBid);
        int index = chooseWorstIssue();
        boolean loop = true;
        long bidTime = System.currentTimeMillis();
        while (loop) {
            if ((System.currentTimeMillis() - bidTime) * 1000L > 3L) {
                break;
            }
            newBid = new Bid(sugestBid);
            try {
                HashMap map = newBid.getValues();
                map.put(Integer.valueOf(index), getRandomValue((Issue) dissues.get(index - 1)));
                newBid = new Bid(this.utilitySpace.getDomain(), map);
                if (this.utilitySpace.getUtility(newBid) > getMyutility()) {
                    return newBid;
                }
            } catch (Exception e) {
                loop = false;
            }
        }
        return newBid;
    }


    /*
     * 在原来的list中不存在相同的bid，则把新的bid加入list中。
     * ？没有弄懂为什么要做下面的第一个if？ 如果list中有utility list： 0:0.7； 1:0.5， 一个新的bid，u=0.8， 则这个list会变成 0:0.7； 1:0.8； 2:0.5
     * 是不是弄错了？ 大概是希望得到一个utility降序的list：0:0.8； 1:0.7； 2:0.5 大概吧？
     *
     */
    public void addBidToList(ArrayList<BidUtility> mybids, BidUtility newbid) {
        int index = mybids.size();
        for (int i = 0; i < mybids.size(); i++) {
            if (((BidUtility) mybids.get(i)).getUtility() <= newbid.getUtility()) { //新的offer让parsagnet更满意
                if (!((BidUtility) mybids.get(i)).getBid().equals(newbid.getBid())) { //且新的offer和之前的不一样
                    index = i;
                } else {
                    return;
                }
            } //所以else就是新的offer比原来的offer更差
        }
        mybids.add(index, newbid);
    }

    /*
     * 这个function在拿到一个对手的bid后，根据这个bid所有issue的value值，更新OpoponentPreference的repeeatedIssue中value的frequency值。
     */
    public void calculateParamForOpponent(OpponentPreferences op, Bid bid) {
        List<Issue> dissues = this.utilitySpace.getDomain().getIssues(); //向系统读取所有issues
        HashMap<Integer, Value> bidVal = bid.getValues(); //获取这个bid的所有issues的value值。 hashmap的key应该是issue的index值
        Integer[] keys = new Integer[0];
        keys = (Integer[]) bidVal.keySet().toArray(keys); //将issue的index值放入Integer[]。 需测试放入的顺序是否和原来一致？
        for (int i = 0; i < dissues.size(); i++) {
            if (op.getRepeatedissue().get(((Issue) dissues.get(i)).getName()) != null) { //如果op的repeatedIssue已经存在这个issue了
                HashMap<Value, Integer> vals = (HashMap) op.getRepeatedissue().get(((Issue) dissues.get(i)).getName()); //读取这个issue的value
                try {
                    if (vals.get(bidVal.get(keys[i])) != null) { //如果这个value原来有freq值，则+1
                        Integer repet = (Integer) vals.get(bidVal.get(keys[i]));
                        repet = Integer.valueOf(repet.intValue() + 1);
                        vals.put((Value) bidVal.get(keys[i]), repet);
                    } else { //以前没有则创建1
                        vals.put((Value) bidVal.get(keys[i]), new Integer(1));
                    }
                } catch (Exception localException) {
                }
            } else { //如果op的repeatedIssue还不存在这个issue
                HashMap<Value, Integer> h = new HashMap();
                try {
                    h.put((Value) bidVal.get(keys[i]), new Integer(1)); //对应的value freq值为1
                } catch (Exception localException1) {
                }
                op.getRepeatedissue().put(((Issue) dissues.get(i)).getName(), h);
            }
        }
    }



    //获取我当前的Utility
    public double getMyutility() {
        double changeU=this.changeTargetUtility;
        double systemU= 1-Math.pow(this.TimeLineInfo.getTime(),5);
        if(changeU>systemU){
            return changeU;
        }else {
            return systemU;
        }
    }


    public double getE() {
        if (this.withDiscount.booleanValue()) {
            return 0.2D;
        }
        return 0.15D;
    }

    //对手偏好类
    private class OpponentPreferences {
        //String是 issue 的name ，HashMap<Value,Integer>是 issue的各种选项所对应的次数
        private HashMap<String, HashMap<Value, Integer>> repeatedissue = new HashMap();
        private ArrayList selectedValues;
        ArrayList<Agent36.BidUtility> opponentBids = new ArrayList();

        private OpponentPreferences() {
        }

        public void setRepeatedissue(HashMap<String, HashMap<Value, Integer>> repeatedissue) {
            this.repeatedissue = repeatedissue;
        }

        public HashMap<String, HashMap<Value, Integer>> getRepeatedissue() {
            return this.repeatedissue;
        }

        public void setSelectedValues(ArrayList selectedValues) {
            this.selectedValues = selectedValues;
        }

        public ArrayList getSelectedValues() {
            return this.selectedValues;
        }

        public void setOpponentBids(ArrayList<Agent36.BidUtility> opponentBids) {
            this.opponentBids = opponentBids;
        }

        public ArrayList<Agent36.BidUtility> getOpponentBids() {
            return this.opponentBids;
        }
    }

    private class BidUtility {
        private Bid bid;
        private double utility;
        private long time;

        BidUtility(Bid b, double u, long t) {
            this.bid = b;
            this.utility = u;
            this.time = t;
        }

        BidUtility(BidUtility newbid) {
            this.bid = newbid.getBid();
            this.utility = newbid.getUtility();
            this.time = newbid.getTime();
        }

        public void setBid(Bid bid) {
            this.bid = bid;
        }

        public Bid getBid() {
            return this.bid;
        }

        public void setUtility(double utility) {
            this.utility = utility;
        }

        public double getUtility() {
            return this.utility;
        }

        public void setTime(long time) {
            this.time = time;
        }

        public long getTime() {
            return this.time;
        }
    }

    @Override
    public String getDescription() {
        return "ANAC2018";
    }
}
