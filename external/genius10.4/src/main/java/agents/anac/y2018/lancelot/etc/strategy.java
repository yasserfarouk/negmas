package agents.anac.y2018.lancelot.etc;

import java.util.List;

import java.util.ArrayList;

import genius.core.Bid;
import genius.core.parties.NegotiationInfo;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AbstractUtilitySpace;

public class strategy {
    private boolean DEBUG = true;
    private AbstractUtilitySpace utilitySpace;
    private TimeLineInfo timeLine;
    private NegotiationInfo negotiationInfo;
    private double eval_opponent = 0.0;
    private int eval_cnt = 0;
    private double my_last_util_ave = 0;
    private double my_last_util_sum = 0;
    private int last_cnt = 0;
    private List<Double> my_util_list;

    public strategy(AbstractUtilitySpace utilitySpace, TimeLineInfo timeLine, NegotiationInfo negotiationInfo){
        this.utilitySpace = utilitySpace;
        this.timeLine = timeLine;
        this.negotiationInfo = negotiationInfo;
        my_util_list = new ArrayList<Double>();
    }

    // When my turn comes, the method is called and decide accept or offer.
    // return true: Accept
    // return false: Offer
    public Boolean decideAcceptOrOffer(Bid lastReceivedBid,double opponent_eval,double my_util){
        double util = 0.0;
        double time = 0.0;
        try {
            time = timeLine.getTime();
            util = utilitySpace.getUtilityWithDiscount(lastReceivedBid,time);

        } catch(Exception e){
            System.out.println("Utilityもしくはtimeを取得できませんでした");
        }
        if (util > getUtilThreshold2(time,opponent_eval,my_util)){
            return true;
        }
        return false;
    }

    //return my threshold of utility at current time
    public double getUtilThreshold2(double time, double opponent_value, double my_util){
        Bid max_bid = null;
        double max_util = 0;
        try {
            max_bid = utilitySpace.getMaxUtilityBid();
        } catch(Exception e){
            System.out.println("max_bidを得ることができませんでした．");
        }
        max_util = utilitySpace.getUtilityWithDiscount(max_bid,time);
//        System.out.println("opponent_value = " + opponent_value);
        double threshold = 1.0;
        double sep_point = 0.7;
        if(time < sep_point){
            threshold = (max_util - opponent_value) / (Math.pow(sep_point,2)) * Math.pow(time-sep_point,2) + opponent_value;
        } else if(time < 0.99){
            threshold = (opponent_value - max_util) / Math.pow(1-sep_point,2) * Math.pow(time-1,2) + max_util;
        }else{
            threshold = max_util * 0.75 * time;
        }

//        if(time > 0.99){
//            last_cnt ++;
//            my_last_util_sum += my_util;
//            threshold = Math.min(getMyUtilAve() - getStandardDeviation() ,1);
//            System.out.println("***************************************last threshold*************************************** = " + threshold);
//        }
//        System.out.println("lancelot's threshold = " + threshold + ": opponent_value = " + opponent_value);
        return threshold;
    }

    public double getUtilThreshold3(double time, double opponent_value, double my_util){
//        System.out.println("opponent_value = " + opponent_value);
        double threshold = 1.0;
        double sep_point = 0.7;
        if(time < sep_point){
            threshold = (1 - opponent_value) / (Math.pow(sep_point,2)) * Math.pow(time-sep_point,2) + opponent_value;
        } else if(time < 0.99){
            threshold = (opponent_value - 1) / Math.pow(1-sep_point,2) * Math.pow(time-1,2) + 1;
        }else{
            threshold = 0.75 * time;
        }

//        if(time > 0.99){
//            last_cnt ++;
//            my_last_util_sum += my_util;
//            threshold = Math.min(getMyUtilAve() + getStandardDeviation()*2 ,1);
//            System.out.println("***************************************last threshold*************************************** = " + threshold);
//        }
//        System.out.println("lancelot's threshold = " + threshold + ": opponent_value = " + opponent_value);
        return threshold;
    }

    public double getUtilThresholdForOffer(){
        return 0.90;
    }

    public double evaluateOpponent(Bid lastRecievedBid){
        eval_cnt++;
//        double my_util = utilitySpace.getUtility(lastRecievedBid);
        double my_util = utilitySpace.getUtilityWithDiscount(lastRecievedBid,timeLine.getTime());
        eval_opponent += my_util;
        my_util_list.add(my_util);
//        getStandardDeviation();
        return getEvalAve() + getStandardDeviation();
    }

    private double getEvalAve(){
        return eval_opponent / eval_cnt;
    }

    private double getMyUtilAve(){
        return my_last_util_sum / last_cnt;
    }

    private double getStandardDeviation(){
        double deviation_sum = 0;
        double ave = getEvalAve();
        for(double my_util : my_util_list){
            deviation_sum += Math.pow((my_util-ave),2);
        }
//        System.out.println("StandardDeviation : " + Math.sqrt(deviation_sum / eval_cnt));
        return Math.sqrt(deviation_sum / eval_cnt);
    }
}
