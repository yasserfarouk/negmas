package agents.anac.y2019.agentgg;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.gui.session.SessionPanel;

import javax.swing.*;
import java.awt.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.lang.Math.max;


/**
 *
 */
public class AgentGG extends AbstractNegotiationParty {

    private ImpMap impMap;
    private ImpMap opponentImpMap;
    private double offerLowerRatio = 1.0;
    private double offerHigherRatio = 1.1;
    private double MAX_IMPORTANCE;
    private double MIN_IMPORTANCE;
    private double MEDIAN_IMPORTANCE;
    private Bid MAX_IMPORTANCE_BID;
    private Bid MIN_IMPORTANCE_BID;
    private double OPPONENT_MAX_IMPORTANCE;
    private double OPPONENT_MIN_IMPORTANCE;
    private Bid receivedBid;
    private Bid initialOpponentBid = null;
    private double lastBidValue;
    private double reservationImportanceRatio;
    private boolean offerRandomly = true;

    private double startTime;
    private boolean maxOppoBidImpForMeGot = false;
    private double maxOppoBidImpForMe;
    private double estimatedNashPoint;
    private Bid lastReceivedBid;
    private boolean initialTimePass = false;


    @Override
    public void init(NegotiationInfo info) {
        super.init(info);

        // 创建 空的 我的importance map 以及对手的 value map
        this.impMap = new ImpMap(userModel);
        this.opponentImpMap = new ImpMap(userModel);


        // 更新我的importance map
        this.impMap.self_update(userModel);

        // 获取最大、最小、中位数bid
        this.getMaxAndMinBid();
        this.getMedianBid();

        // 获取reservation value，折算为importance的百分比
        this.reservationImportanceRatio = this.getReservationRatio();

        System.out.println("reservation ratio: " + this.reservationImportanceRatio);
        System.out.println("my max importance bid: " + this.MAX_IMPORTANCE_BID);
        System.out.println("my max importance: " + this.MAX_IMPORTANCE);
        System.out.println("my min importance bid: " + this.MIN_IMPORTANCE_BID);
        System.out.println("my min importance: " + this.MIN_IMPORTANCE);
        System.out.println("my median importance: " + this.MEDIAN_IMPORTANCE);
        System.out.println("Agent " + this.getPartyId() + " has finished initialization");
    }

    @Override
    public Action chooseAction(List<Class<? extends Action>> list) {
        double time = getTimeLine().getTime();

        // 开始比赛
        if (!(this.getLastReceivedAction() instanceof Offer)) return new Offer(getPartyId(), this.MAX_IMPORTANCE_BID);

        // 对方报价对我来说的importance ratio
        double impRatioForMe = (this.impMap.getImportance(this.receivedBid) - this.MIN_IMPORTANCE) / (this.MAX_IMPORTANCE - this.MIN_IMPORTANCE);

        // 接受报价的条件，即高于我的threshold
        if (impRatioForMe >= this.offerLowerRatio) {
            System.out.println("\n\naccepted agent: Agent" + this.getPartyId());
            System.out.println("last bid: " + this.receivedBid);
            System.out.println("\ncurrent threshold: " + this.offerLowerRatio);
            System.out.println("\n\n");
            return new Accept(this.getPartyId(), this.receivedBid);
        }

        // 对手importance为1.0左右时，己方大致可以拿到多少。即寻找Pareto边界的端点
        if (!maxOppoBidImpForMeGot) this.getMaxOppoBidImpForMe(time, 3.0 / 1000.0);

        // 更新对手importance表
        if (time < 0.3) this.opponentImpMap.opponent_update(this.receivedBid);

        // 策略
        this.getThreshold(time);

        // 最后一轮
        if (time >= 0.9989) {
            double ratio = (this.impMap.getImportance(this.receivedBid) - this.MIN_IMPORTANCE) / (this.MAX_IMPORTANCE - this.MIN_IMPORTANCE);
            if (ratio > this.reservationImportanceRatio + 0.2) {
                return new Accept(getPartyId(), receivedBid);
            }
        }

        System.out.println("high threshold: " + this.offerHigherRatio);
        System.out.println("low threshold: " + this.offerLowerRatio);
        System.out.println("estimated nash: " + this.estimatedNashPoint);
        System.out.println("reservation: " + this.reservationImportanceRatio);
        System.out.println();

        Bid bid = getNeededRandomBid(this.offerLowerRatio, this.offerHigherRatio);
        this.lastReceivedBid = this.receivedBid;
        return new Offer(getPartyId(), bid);
    }

    @Override
    public void receiveMessage(AgentID sender, Action act) {
        super.receiveMessage(sender, act);
        if (act instanceof Offer) {
            Offer offer = (Offer) act;
            this.receivedBid = offer.getBid();
        }
    }

    @Override
    public String getDescription() {
        return "Well Played";
    }

    /**
     * 获取对方utility为1.0左右时我方的最优值（Pareto最优边界点）
     * 对方可能先报几次最高的相同bid，忽视，当不同时候开始计时。在持续时间内（如20轮），选择对我来说最高importance的bid。
     * 由于此时对方的bid对对方而言importance一定很高，因此可以满足我方要求。
     */
    private void getMaxOppoBidImpForMe(double time, double timeLast) {
        double thisBidImp = this.impMap.getImportance(this.receivedBid);
        if (thisBidImp > this.maxOppoBidImpForMe) this.maxOppoBidImpForMe = thisBidImp;

        if (this.initialTimePass) {
            if (time - this.startTime > timeLast) {
                double maxOppoBidRatioForMe = (this.maxOppoBidImpForMe - this.MIN_IMPORTANCE) / (this.MAX_IMPORTANCE - this.MIN_IMPORTANCE);
                this.estimatedNashPoint = (1 - maxOppoBidRatioForMe) / 1.7 + maxOppoBidRatioForMe; // 1.414 是圆，2是直线
                this.maxOppoBidImpForMeGot = true;
            }
        } else {
            if (this.lastReceivedBid != this.receivedBid) {
                this.initialTimePass = true;
                this.startTime = time;
            }
        }
    }

    /**
     * 根据时间获取阈值上下限
     */
    private void getThreshold(double time) {
        if (time < 0.01) {
            // 前10轮报0.9999，为了适应部分特殊的域
            this.offerLowerRatio = 0.9999;
        } else if (time < 0.02) {
            // 10~20轮报0.99，为了适应部分特殊的域
            this.offerLowerRatio = 0.99;
        } else if (time < 0.2) {
            // 20~200轮报高价，降至0.9
            this.offerLowerRatio = 0.99 - 0.5 * (time - 0.02);
        } else if (time < 0.5) {
            this.offerRandomly = false;
            // 200~500轮逐步降低阈值，降至距估计的Nash点0.5
            double p2 = 0.3 * (1 - this.estimatedNashPoint) + this.estimatedNashPoint;
            this.offerLowerRatio = 0.9 - (0.9 - p2) / (0.5 - 0.2) * (time - 0.2);
        } else if (time < 0.9) {
            // 500~900轮快速降低阈值，降至距估计的Nash点0.2
            double p1 = 0.3 * (1 - this.estimatedNashPoint) + this.estimatedNashPoint;
            double p2 = 0.15 * (1 - this.estimatedNashPoint) + this.estimatedNashPoint;
            this.offerLowerRatio = p1 - (p1 - p2) / (0.9 - 0.5) * (time - 0.5);
        } else if (time < 0.98) {
            // 妥协1
            double p1 = 0.15 * (1 - this.estimatedNashPoint) + this.estimatedNashPoint;
            double p2 = 0.05 * (1 - this.estimatedNashPoint) + this.estimatedNashPoint;
            double possibleRatio = p1 - (p1 - p2) / (0.98 - 0.9) * (time - 0.9);
            this.offerLowerRatio = max(possibleRatio, this.reservationImportanceRatio + 0.3);
        } else if (time < 0.995) {
            // 妥协2 980~995轮
            double p1 = 0.05 * (1 - this.estimatedNashPoint) + this.estimatedNashPoint;
            double p2 = 0.0 * (1 - this.estimatedNashPoint) + this.estimatedNashPoint;
            double possibleRatio = p1 - (p1 - p2) / (0.995 - 0.98) * (time - 0.98);
            this.offerLowerRatio = max(possibleRatio, this.reservationImportanceRatio + 0.25);
        } else if (time < 0.999) {
            // 妥协3 995~999轮
            double p1 = 0.0 * (1 - this.estimatedNashPoint) + this.estimatedNashPoint;
            double p2 = -0.35 * (1 - this.estimatedNashPoint) + this.estimatedNashPoint;
            double possibleRatio = p1 - (p1 - p2) / (0.9989 - 0.995) * (time - 0.995);
            this.offerLowerRatio = max(possibleRatio, this.reservationImportanceRatio + 0.25);
        } else {
            double possibleRatio = -0.4 * (1 - this.estimatedNashPoint) + this.estimatedNashPoint;
            this.offerLowerRatio = max(possibleRatio, this.reservationImportanceRatio + 0.2);
        }
        this.offerHigherRatio = this.offerLowerRatio + 0.1;
    }

    /**
     * 获取reservation value对应到importance matrix中的比例
     */
    private double getReservationRatio() {
        double medianBidRatio = (this.MEDIAN_IMPORTANCE - this.MIN_IMPORTANCE) / (this.MAX_IMPORTANCE - this.MIN_IMPORTANCE);
        return this.utilitySpace.getReservationValue() * medianBidRatio / 0.5;
    }

    /**
     * 获取最大、最小importance的值及对应offer
     */
    private void getMaxAndMinBid() {
        HashMap<Integer, Value> lValues1 = new HashMap<>();
        HashMap<Integer, Value> lValues2 = new HashMap<>();
        for (Map.Entry<Issue, List<impUnit>> entry : this.impMap.entrySet()) {
            Value value1 = entry.getValue().get(0).valueOfIssue;
            Value value2 = entry.getValue().get(entry.getValue().size() - 1).valueOfIssue;
            int issueNumber = entry.getKey().getNumber();
            lValues1.put(issueNumber, value1);
            lValues2.put(issueNumber, value2);
        }
        this.MAX_IMPORTANCE_BID = new Bid(this.getDomain(), lValues1);
        this.MIN_IMPORTANCE_BID = new Bid(this.getDomain(), lValues2);
        this.MAX_IMPORTANCE = this.impMap.getImportance(this.MAX_IMPORTANCE_BID);
        this.MIN_IMPORTANCE = this.impMap.getImportance(this.MIN_IMPORTANCE_BID);
    }


    /**
     * 获取bid ranking 中的中位数bid对应的importance值
     */
    private void getMedianBid() {
        int median = (this.userModel.getBidRanking().getSize() - 1) / 2;
        int median2 = -1;
        if (this.userModel.getBidRanking().getSize() % 2 == 0) {
            median2 = median + 1;
        }
        int current = 0;
        for (Bid bid : this.userModel.getBidRanking()) {
            current += 1;
            if (current == median) {
                this.MEDIAN_IMPORTANCE = this.impMap.getImportance(bid);
                if (median2 == -1) break;
            }
            if (current == median2) {
                this.MEDIAN_IMPORTANCE += this.impMap.getImportance(bid);
                break;
            }
        }
        if (median2 != -1) this.MEDIAN_IMPORTANCE /= 2;
    }

//    /**
//     * 更新对手的最大及最小Importance的值及对应OFFER
//     */
//    private void getOpponentMaxAndMinBid() {
//        HashMap<Integer, Value> lValues1 = new HashMap<>();
//        HashMap<Integer, Value> lValues2 = new HashMap<>();
//        for (Map.Entry<Issue, List<impUnit>> entry : this.opponentImpMap.entrySet()) {
//            Value value1 = entry.getValue().get(0).valueOfIssue;
//            Value value2 = entry.getValue().get(entry.getValue().size() - 1).valueOfIssue;
//            int issueNumber = entry.getKey().getNumber();
//            lValues1.put(issueNumber, value1);
//            lValues2.put(issueNumber, value2);
//        }
//        Bid OPPONENT_MAX_IMPORTANCE_BID = new Bid(this.getDomain(), lValues1);
//        Bid OPPONENT_MIN_IMPORTANCE_BID = new Bid(this.getDomain(), lValues2);
//        this.OPPONENT_MAX_IMPORTANCE = this.opponentImpMap.getImportance(OPPONENT_MAX_IMPORTANCE_BID);
//        this.OPPONENT_MIN_IMPORTANCE = this.opponentImpMap.getImportance(OPPONENT_MIN_IMPORTANCE_BID);
//    }


    /**
     * 获取符合条件的随机bid。随机生成k个bid，选取其中在阈值范围内的bids，返回其中对手importance最高的一个bid。
     *
     * @param lowerRatio 生成随机bid的importance下限
     * @param upperRatio 生成随机bid的importance上限
     * @return Bid
     */
    private Bid getNeededRandomBid(double lowerRatio, double upperRatio) {
        double lowerThreshold = lowerRatio * (this.MAX_IMPORTANCE - this.MIN_IMPORTANCE) + this.MIN_IMPORTANCE;
        double upperThreshold = upperRatio * (this.MAX_IMPORTANCE - this.MIN_IMPORTANCE) + this.MIN_IMPORTANCE;
        for (int t = 0; t < 3; t++) {
            long k = 2*this.getDomain().getNumberOfPossibleBids();
            double highest_opponent_importance = 0.0;
            Bid returnedBid = null;
            for (int i = 0; i < k; i++) {
                Bid bid = this.generateRandomBid();
                double bidImportance = this.impMap.getImportance(bid);
                double bidOpponentImportance = this.opponentImpMap.getImportance(bid);
                if (bidImportance >= lowerThreshold && bidImportance <= upperThreshold) {
                    if (this.offerRandomly) return bid; // 前0.2时间随机给bid即可
                    if (bidOpponentImportance > highest_opponent_importance) {
                        highest_opponent_importance = bidOpponentImportance;
                        returnedBid = bid;
                    }
                }
            }
            if (returnedBid != null) {
                return returnedBid;
            }
        }
        // 如果出现了问题，没找到合适的bid，就随即返回一个高于下限的
        while (true) {
            Bid bid = generateRandomBid();
            if (this.impMap.getImportance(bid) >= lowerThreshold) {
                return bid;
            }
        }
    }

    public static void main(String[] args) {
        final JFrame gui = new JFrame();
        gui.setLayout(new BorderLayout());
        gui.getContentPane().add(new SessionPanel(), BorderLayout.CENTER);
        gui.pack();
        gui.setVisible(true);
    }

}
