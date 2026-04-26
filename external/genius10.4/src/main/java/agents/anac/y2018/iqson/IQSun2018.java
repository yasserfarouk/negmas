package agents.anac.y2018.iqson;

import java.util.List;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.Domain;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiationWithAnOffer;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.OutcomeSpace;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.UtilitySpace;


/*
*
* Crated by Seyed Mohammad Hussein Kazemi
*
* April 15th 2018
*
* University of Tehran
*
* Agent Lab.
*
* smhk1375@icloud.com
*
*
* */


public class IQSun2018 extends AbstractNegotiationParty {

    private Bid lastReceivedBid, myLastBid;
    private static double MIN_UTILITY = 0.5;
    private double concessionFactor;//e
    protected int k = 0;///** k \in [0, 1]. For k = 0 the agent starts with a bid of maximum utility */
    private BidHistory currSessOppBidHistory, currSessOurBidHistory;
    private TimeLineInfo TimeLineInfo = null;
    private OutcomeSpace outcomeSpace;
    private double[] weightsIfHelpingUtilIsLessThanOrEqToOne, weightsIfHelpingUtilIsMoreThanOne;

    private List<BidInfo> lBids;

    @Override
    public void init(NegotiationInfo info) {
        super.init(info);
        concessionFactor = 0.1;
        lastReceivedBid = null;
        outcomeSpace = new OutcomeSpace(utilitySpace);
        generateBidNearMyUtility(1);
        currSessOppBidHistory = new BidHistory();
        currSessOurBidHistory = new BidHistory();
        TimeLineInfo = info.getTimeline();
        lBids = new ArrayList<>(AgentTool.generateRandomBids(info.getUtilitySpace().getDomain(), 30000, this.rand, this.utilitySpace));
        lBids.sort(new BidInfoComp().reversed());
        determineWeights();

//        utilitySpace.getDiscountFactor(); <--------------------------  it gives us discount factor
    }

    private Bid generateHelpingBid(){
        double threshold_high = 1 - 0.15 * this.timeline.getTime() * Math.abs(Math.sin(this.timeline.getTime() * 20));
        double threshold_low = 1 - 0.21 * this.timeline.getTime() * Math.abs(Math.sin(this.timeline.getTime() * 20));
        Bid bid = null;
        while (bid == null) {
            bid = AgentTool.selectBidfromList(this.lBids, threshold_high, threshold_low);
            if (bid == null) {
                threshold_low -= 0.0001;
            }
        }
        return bid;
    }

    private void determineWeights() {
        double domainSize = utilitySpace.getDomain().getNumberOfPossibleBids();
        if(domainSize <= 1000){
            weightsIfHelpingUtilIsLessThanOrEqToOne = new double[]{0.75, 0.05, 0.2};
            weightsIfHelpingUtilIsMoreThanOne = new double[]{0.75, 0.25};
        }else if(domainSize > 1000 && domainSize <= 2000){
            weightsIfHelpingUtilIsLessThanOrEqToOne = new double[]{0.85, 0.05, 0.1};
            weightsIfHelpingUtilIsMoreThanOne = new double[]{0.85, 0.15};
        } else if(domainSize > 2000 && domainSize <= 100000) {
            concessionFactor = 0.3;
            weightsIfHelpingUtilIsLessThanOrEqToOne = new double[]{0.7, 0.05, 0.25};
            weightsIfHelpingUtilIsMoreThanOne = new double[]{0.7, 0.3};
        } else {
            weightsIfHelpingUtilIsLessThanOrEqToOne = new double[]{0.85, 0.05, 0.1};
            weightsIfHelpingUtilIsMoreThanOne = new double[]{0.85, 0.15};
        }
    }

    public Action chooseAction(List<Class<? extends Action>> validActions) {
        try {
            if (lastReceivedBid == null) {
                try {
                    return new Offer(getPartyId(), outcomeSpace.getBidNearUtility(1).getBid());
                } catch (Exception e) {
                    System.out.println("here in chooseAction: ");
                    System.out.println(e.getLocalizedMessage());
                    e.printStackTrace();
                }
            } else {
                try {
                    double offeredUtilFromOpponent = utilitySpace.getUtility(lastReceivedBid), time = timeline.getTime();
                    if (time < 0.17) {
                        if (offeredUtilFromOpponent >= 0.9)
                            return new Accept(getPartyId(), lastReceivedBid);
                        return (new Offer(getPartyId(), outcomeSpace.getBidNearUtility(1).getBid()));
                    }
                    if (isAcceptable(offeredUtilFromOpponent, time))
                        return new Accept(getPartyId(), lastReceivedBid);
                    double[] weights;
                    double[] values;
                    double helpingUtility = getUtility(generateHelpingBid());
                    if (helpingUtility <= 1) {
                        values = new double[]{getUtilityByTime(time), determineUtilOnlyByCurrSesHistory(), helpingUtility};
                        weights = weightsIfHelpingUtilIsLessThanOrEqToOne;
                    } else {
                        values = new double[]{getUtilityByTime(time), determineUtilOnlyByCurrSesHistory()};
                        weights = weightsIfHelpingUtilIsMoreThanOne;
                    }
                    generateBidNearMyUtility(average(values, weights));
                    return generateMyOfferAndAddItToMyCurrSessHistory();
                } catch (Exception e) {
                    System.out.println("HERE IT IS !!");
                    System.out.println(e.getMessage());
                }
            }
            generateBidNearMyUtility(1);
            return (new Offer(getPartyId(), myLastBid));
        }catch (Exception e){
            return (new Offer(getPartyId(), outcomeSpace.getBidNearUtility(1).getBid()));
        }
    }

    private Offer generateMyOfferAndAddItToMyCurrSessHistory(){
        BidDetails myLastBidDetails = new BidDetails(myLastBid, getUtility(myLastBid), TimeLineInfo.getTime());
        currSessOurBidHistory.add(myLastBidDetails);
        return (new Offer(getPartyId(), myLastBid));
    }



    private void generateBidNearMyUtility(double myUtility){
        if(myUtility > 1)
            myUtility = 0.99;
        if(myUtility < MIN_UTILITY)
            myUtility = MIN_UTILITY;
        myLastBid = outcomeSpace.getBidNearUtility(myUtility).getBid();
    }

    @Override
    public void receiveMessage(AgentID sender, Action action) {
        super.receiveMessage(sender, action);
        if (action instanceof Offer) {
            lastReceivedBid = ((Offer) action).getBid();
            try {
                BidDetails opponentBid = new BidDetails(lastReceivedBid, getUtility(lastReceivedBid),
                        TimeLineInfo.getTime());
                currSessOppBidHistory.add(opponentBid);
            } catch (Exception e) {
                System.out.println("here in receiveMessage: ");
                System.out.println(e.getLocalizedMessage());
                e.printStackTrace();
                new EndNegotiationWithAnOffer(this.getPartyId(),
                        outcomeSpace.getBidNearUtility(1).getBid());
            }
        }
    }

    @Override
    public String getDescription() {
        return "ANAC2018";
    }

    private double averageOfBestFiveBids(){
        if(currSessOppBidHistory.size() > 5){
            List<BidDetails> bestNBidsDetails = currSessOppBidHistory.getNBestBids(5);
            double retVal = 0;
            for(BidDetails bidDetails : bestNBidsDetails){
                retVal += getUtility(bidDetails.getBid());
            }
            return retVal / 5;
        }else {
            return currSessOppBidHistory.getAverageUtility();
        }
    }

    private double determineUtilOnlyByCurrSesHistory(){
        if(currSessOppBidHistory.getHistory().isEmpty()){
            return 1;
        }
        if(currSessOurBidHistory.getHistory().isEmpty())
            return averageOfBestFiveBids() * 0.05 + 0.95;

        return averageOfBestFiveBids() * 0.05 + currSessOurBidHistory.getAverageUtility() * 0.95;
    }


    private boolean isAcceptable(double offeredUtilFromOpponent, double time) {
        double myLastBidUtility = utilitySpace.getUtility(myLastBid);
        if(time >= 0.99 && partnerLastBidHasUtilMoreThanOrEqualToResVal())
            return true;

        if(time >= 0.95 && (MIN_UTILITY <= offeredUtilFromOpponent))
            return true;

        if(getUtilityByTime(time) > offeredUtilFromOpponent)
            return false;
        if(!partnerLastBidHasUtilMoreThanResVal()) {
            return false;
        }
        if(offeredUtilFromOpponent < currSessOppBidHistory.getAverageUtility())
            return false;
        return myLastBidUtility <= offeredUtilFromOpponent && MIN_UTILITY <= offeredUtilFromOpponent;
    }

    private double getUtilityByTime(double t) {
        try {
            double pMin = utilitySpace.getUtility(utilitySpace.getMinUtilityBid());
            double pMax = utilitySpace.getUtility(utilitySpace.getMaxUtilityBid());
            double u = pMin + (pMax - pMin) * (1 - f(t));
            return u >= MIN_UTILITY ? u : MIN_UTILITY;
        } catch (Exception e) {
            System.out.println("here in getUtilityByTime: ");
            System.out.println(e.getLocalizedMessage());
            e.printStackTrace();
            return 1;
        }
    }

    public double f(double t) throws Exception {
        if (concessionFactor == 0)
            return k;
        return k + (1 - k) * Math.pow(t, 1.0 / concessionFactor);
    }


    private double average(double[]values, double[]weights){
        double average = 0;
        double sumOfWeights = 0;
        for(int i = 0; i < values.length; i++){
            average += values[i] * weights[i];
            sumOfWeights += weights[i];
        }

        return average / sumOfWeights;
    }

    private boolean partnerLastBidHasUtilMoreThanResVal(){
        return utilitySpace.getUtility(lastReceivedBid) > utilitySpace.getReservationValue();
    }

    private boolean partnerLastBidHasUtilMoreThanOrEqualToResVal(){
        return utilitySpace.getUtility(lastReceivedBid) >= utilitySpace.getReservationValue();
    }

}

class AgentTool {

    private static Random random = new Random();

    static Bid selectBidfromList(List<BidInfo> bidInfoList, double higerutil, double lowwerutil) {
        List<BidInfo> bidInfos = new ArrayList<>();
        for (BidInfo bidInfo : bidInfoList) {
            if (bidInfo.getUtil() <= higerutil && bidInfo.getUtil() >= lowwerutil) {
                bidInfos.add(bidInfo);
            }
        }
        if (bidInfos.size() == 0) {
            return null;
        } else {
            return bidInfos.get(random.nextInt(bidInfos.size())).getBid();
        }
    }

    static Set<BidInfo> generateRandomBids(Domain d, int numberOfBids, Random random, UtilitySpace utilitySpace) {
        Set<BidInfo> randomBids = new HashSet<>();
        for (int i = 0; i < numberOfBids; i++) {
            Bid b = d.getRandomBid(random);
            randomBids.add(new BidInfo(b, utilitySpace.getUtility(b)));
        }
        return randomBids;
    }

}

class BidInfo {
    private Bid bid;
    private double util;

    BidInfo(Bid b, double u) {
        this.bid = b;
        util = u;
    }

    public Bid getBid() {
        return bid;
    }

    public double getUtil() {
        return util;
    }

    @Override
    public int hashCode() {
        return bid.hashCode();
    }

    public boolean equals(BidInfo bidInfo) {
        return bid.equals(bidInfo.getBid());
    }

    @Override
    public boolean equals(Object obj) {
        return obj != null && obj instanceof BidInfo && ((BidInfo) obj).getBid().equals(bid);
    }

}

final class BidInfoComp implements Comparator<BidInfo> {
    BidInfoComp() {
        super();
    }

    @Override
    public int compare(BidInfo o1, BidInfo o2) {
        return Double.compare(o1.getUtil(), o2.getUtil());
    }
}
