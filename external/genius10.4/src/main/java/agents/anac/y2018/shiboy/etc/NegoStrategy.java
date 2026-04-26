package agents.anac.y2018.shiboy.etc;

import java.util.ArrayList;

import genius.core.Bid;
import genius.core.parties.NegotiationInfo;


/**
 * Created by shota_suzuki on 2018/05/16.
 */
public class NegoStrategy {
    private NegotiationInfo info;
    private boolean isPrinting = false; // デバッグ用
    private boolean isPrinting_Strategy = true;

    private NegoStats negoStats; // 交渉情報
    private NegoHistory negoHistory;
    private double rv = 0.0;                // 留保価格
    private double df = 0.0;                // 割引効用


    public NegoStrategy(NegotiationInfo info, boolean isPrinting, NegoStats negoStats, NegoHistory negoHistory) {
        this.info = info;
        this.isPrinting = isPrinting;

        this.negoStats = negoStats;
        this.negoHistory = negoHistory;
        rv = info.getUtilitySpace().getReservationValueUndiscounted();
        df = info.getUtilitySpace().getDiscountFactor();

        if (this.isPrinting) {
            System.out.println("[isPrinting] NegoStrategy: success");
        }
        if (isPrinting_Strategy) {
            System.out.println("[isPrint_Strategy] rv = " + rv);
            System.out.println("[isPrint_Strategy] df = " + df);
        }
    }

    // 受容判定
    public boolean selectAccept(Bid offeredBid, double time) {
        try {
            if (info.getUtilitySpace().getUtility(offeredBid) >= getThreshold(time)) {
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
        if (1.0 - df < 1e-7) { // 割引効用が設定されていない
            return false;
        } else {
            if (rv > getThreshold(time)) {
                return true;
            } else {
                return false;
            }
        }

    }

    /**
     * 統計情報を元に相手の変位幅を推定 (二分割一様分布を利用)
     *
     * @param m  平均効用値
     * @param sd 標準偏差
     * @return
     */
    public double calWidth(double m, double sd) {
        if (m > 0.1 && m < 0.9) {
            return Math.sqrt(3.0 / (m - m * m)) * sd;
        } else {
            return Math.sqrt(12) * sd;
        }
    }

    // 閾値を返す
    public double getThreshold(double time) {
        double threshold = 1.0;
        double alpha = 4.0;

        // 交渉相手全員に対してemaxを計算し，最小となるものを探す
        ArrayList<Object> rivals = negoStats.getRivals();
        double emax = 1.0;
        for (Object sender : rivals) {
            double m = negoStats.getRivalMean(sender);
            double sd = negoStats.getRivalSD(sender);

            // emax = Math.min(emax, m + (1 - m)*calWidth(m, sd));
            // negoStats.getRivalMax(sender) より今sessionにおける最大効用値を採用
            emax = Math.min(emax, Math.max(negoStats.getRivalMax(sender), m + (1 - m) * calWidth(m, sd)));
            emax = Math.max(emax, rv); //　留保価格より小さい場合は，rvを採用する．
        }

        if (1.0 - df < 1e-7) { // 割引効用が設定されていない
            threshold = Math.min(threshold, 1 - (1 - emax) * Math.pow(time, alpha));
        } else {
            threshold = Math.max(threshold - time, emax);
        }

        /*
     // 一旦、過去の提案で最大のものまで譲歩する
        if (0.80 < time && time < 0.82) {
            for (Object sender : rivals) {
                threshold = Math.min(threshold, negoStats.getRivalMax(sender));
            }
            threshold = Math.max(threshold, rv);
        }

        if (isPrinting_Strategy) {
            System.out.println("[isPrint_Strategy] threshold = " + threshold
                    + " | time: " + time
                    + ", emax: " + emax);
        }
        */
        
        // 交渉決裂寸前では、過去の提案で最大のものまで譲歩する
        if (time > 0.99) {
            for (Object sender : rivals) {
                threshold = Math.min(threshold, negoStats.getRivalMax(sender));
            }
            threshold = Math.max(threshold, rv);
        }

        if (isPrinting_Strategy) {
            System.out.println("[isPrint_Strategy] threshold = " + threshold
                    + " | time: " + time
                    + ", emax: " + emax);
        }

        return threshold;
    }
}
