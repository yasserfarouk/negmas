package agents.anac.y2018.agent33.etc;

import java.util.ArrayList;

import genius.core.Bid;
import genius.core.parties.NegotiationInfo;


public class NegoStrategy {
    private NegotiationInfo info;
    private boolean isPrinting = false; // デバッグ用
    private boolean isPrinting_Strategy = true;

    private NegoStats negoStats; // 交渉情報
    private NegoHistory negoHistory;
    private double rv = 0.0;                // 留保価格
    private double df = 0.0;                // 割引効用


    public NegoStrategy(NegotiationInfo info, boolean isPrinting, NegoStats negoStats, NegoHistory negoHistory){
        this.info = info;
        this.isPrinting = isPrinting;

        this.negoStats = negoStats;
        this.negoHistory = negoHistory;
        rv = info.getUtilitySpace().getReservationValueUndiscounted();
        df = info.getUtilitySpace().getDiscountFactor();

        if(this.isPrinting){
            System.out.println("[isPrinting] NegoStrategy: success");
        }
        if(isPrinting_Strategy){
            System.out.println("[isPrint_Strategy] rv = " + rv);
            System.out.println("[isPrint_Strategy] df = " + df);
        }
    }

    // 受容判定
    public boolean selectAccept(Bid offeredBid, double time) {
        try {
            if(info.getUtilitySpace().getUtility(offeredBid) >= getThreshold(time)){
                return true;
            } else {
                return false;
            }
        } catch (Exception e) {
            System.out.println("[Exception_Strategy] 受容判定に失敗しました");
            e.printStackTrace();
            return false;
        }
    }

    // 交渉終了判定
    public boolean selectEndNegotiation(double time) {
        // 割引効用が設定されているかどうか (ANAC 2017では設定されない方針)
        if(1.0 - df < 1e-7){
            return false;
        } else {
            if (rv > getThreshold(time)){
                return true;
            } else {
                return false;
            }
        }

    }

    /**
     * 統計情報を元に相手の変位幅を推定 (二分割一様分布を利用)
     * @param m 平均効用値
     * @param sd 標準偏差
     * @return
     */
    public double calWidth(double m, double sd){
        if(m > 0.1 && m < 0.9){
            return Math.sqrt(3.0 / (m - m*m)) * sd;
        } else {
            return Math.sqrt(12) * sd;
        }
    }

    // 閾値を返す
    public double getThreshold(double time) {
        double threshold = 1.0;//相手からのbidを判断する譲歩関数
        //double alpha = 3.0;//-----3
        //double alpha = 3.5;
        //double alpha = 4.0;//
        //double alpha = 4.3;
        //double alpha = 4.9;//---2
        double alpha = 4.5;//---1
        //double alpha = 4.2;

        // 交渉相手全員に対してemaxを計算し，最小となるものを探す
        ArrayList<Object> rivals = negoStats.getRivals();
        double emax = 1.0;
        for(Object sender: rivals){
            double m    = negoStats.getRivalMean(sender);
            double sd   = negoStats.getRivalSD(sender);

            // emax = Math.min(emax, m + (1 - m)*calWidth(m, sd));
            // negoStats.getRivalMax(sender) より今sessionにおける最大効用値を採用
            emax = Math.min(emax, Math.max(negoStats.getRivalMax(sender), m + (1 - m)*calWidth(m, sd)));
            emax = Math.max(emax, rv); //　留保価格より小さい場合は，rvを採用する．
        }

        // 割引効用が設定されているかどうか (ANAC 2017では設定されない方針)
        if(1.0 - df < 1e-7){
            threshold = Math.min(threshold, 1 - (1 - emax) * Math.pow(time, alpha));
        } else {

        		//2018/5/18 Farma2016
        		//threshold = Math.max(1 - (1 - df) * Math.log(1 + (Math.E - 1) * time), emax);

        		//Farma2017
            //threshold = Math.max(threshold - time, emax);

        		//threshold1
        		//threshold = Math.max(1 - Math.E *  df * Math.log(1 + Math.E * time), emax);

        		//threshold2
        		//threshold = Math.max(1 - (1 - df) * Math.log(1 + Math.pow((Math.E - 1),alpha) * time), emax);

        		//threshold3------2.alpha 4.5
        		//threshold = Math.max(1 - (1 - df) * Math.log(Math.E - 1.9 + Math.pow((Math.E - 1),alpha) * time), emax);


        		//threshold4------1

        		System.out.println("df:" + df);
        		System.out.println("emax:" + emax);
        		if(Math.abs((df - (1 - df))) > 4.0){
        		if(df > 0.5) {
        		threshold = Math.max(df - (1 - df) * Math.log(Math.E - 1.9 + Math.pow((Math.E - 1.0),alpha) * time), emax);
        		}
        		if(df <= 0.5) {
        		threshold = Math.max((1 - df) - df * Math.log(Math.E - 1.9 +Math.pow((Math.E - 1.2),alpha) * time), emax);//1.9
        		}

        		}

        else {
        			threshold = Math.max(1 - (1 - df) * Math.log(Math.E - 1.9 + Math.pow((Math.E - 1),alpha) * time), emax);
        		}

        	//threshold5
        	/*
        	if(df > 0.5) {
        		threshold = df - (1 - df) * Math.log(Math.E - 1.9 + Math.pow((Math.E - 1.0),alpha) * time);
        		}
        		if(df <= 0.5) {
        		threshold = df - (1 - df) * alpha * Math.log(Math.E - 1.9 +Math.pow((Math.E - 1.0),alpha) * time);//1.9
        		}
        }
        */
        }
        // 交渉決裂寸前では、過去の提案で最大のものまで譲歩する
        if(time > 0.95){
            for(Object sender: rivals) {
                threshold = Math.min(threshold, negoStats.getRivalMax(sender));
            }
            threshold = Math.max(threshold, rv);
        }

        if(isPrinting_Strategy){
            System.out.println("[isPrint_Strategy] threshold = " + threshold
                    + " | time: " + time
                    + ", emax: " + emax);
        }

        return threshold;
    }


    /**
     * 2018/5/27
     * bid探索用閾値を返す
     */
    public double getThresholdForBidSearch(double time) {
        double threshold = 1.0;//相手からのbidを判断する譲歩関数
        double alpha = 3.0;

        // 交渉相手全員に対してemaxを計算し，最小となるものを探す
        ArrayList<Object> rivals = negoStats.getRivals();
        double emax = 1.0;
        for(Object sender: rivals){
            double m    = negoStats.getRivalMean(sender);
            double sd   = negoStats.getRivalSD(sender);

            // emax = Math.min(emax, m + (1 - m)*calWidth(m, sd));
            // negoStats.getRivalMax(sender) より今sessionにおける最大効用値を採用
            emax = Math.min(emax, Math.max(negoStats.getRivalMax(sender), m + (1 - m)*calWidth(m, sd)));
            emax = Math.max(emax, rv); //　留保価格より小さい場合は，rvを採用する．
        }


            threshold = 1 - (1 - emax) * Math.pow(time, alpha);

        		//2018/5/18 Farma2016
        		//threshold = Math.max(1 - (1 - df) * Math.log(1 + (Math.E - 1) * time), emax);

        		//Farma2017
            //threshold = Math.max(threshold - time, emax);

        		//threshold1
        		//threshold = Math.max(1 - Math.E *  df * Math.log(1 + Math.E * time), emax);

        		//threshold2




        // 交渉決裂寸前では、過去の提案で最大のものまで譲歩する
        if(time > 0.99){
            for(Object sender: rivals) {
                threshold = Math.min(threshold, negoStats.getRivalMax(sender));
            }
            threshold = Math.max(threshold, rv);
        }

        if(isPrinting_Strategy){
            System.out.println("[isPrint_Strategy] threshold = " + threshold
                    + " | time: " + time
                    + ", emax: " + emax);
        }

        return threshold;
    }
}
