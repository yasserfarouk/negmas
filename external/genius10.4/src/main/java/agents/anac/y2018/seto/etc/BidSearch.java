package agents.anac.y2018.seto.etc;

import java.util.List;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.parties.NegotiationInfo;


public class BidSearch {
    private NegotiationInfo info;
    private boolean isPrinting = false; // デバッグ用
    private boolean isPrinting_Search = false;

    private NegoStats negoStats; // 交渉情報
    private NegoHistory negoHistory;
    private Bid maxBid = null; // 最大効用値Bid

    public BidSearch(NegotiationInfo info, boolean isPrinting, NegoStats negoStats, NegoHistory negoHistory) throws Exception {
        this.info = info;
        this.isPrinting = isPrinting;

        this.negoStats = negoStats;
        this.negoHistory = negoHistory;

        initMaxBid(); // 最大効用値Bidの初期探索
        negoStats.setValueRelativeUtility(maxBid); // 相対効用値を導出する

        if(this.isPrinting){
            System.out.println("[isPrinting] BidSearch: success");
        }

    }

    /**
     * Bidを返す
     * @param baseBid
     * @param threshold
     * @return
     */
    public Bid getBid(Bid baseBid, double threshold) {
        try {
            Bid bid = getBidbyAppropriateSearch(baseBid, threshold); // 閾値以上の効用値を持つ合意案候補を探索

            // 探索によって得られたBidがthresholdよりも小さい場合，最大効用値Bidを基準とする
            if (info.getUtilitySpace().getUtility(bid) < threshold) {
                bid = new Bid(maxBid);
            }

            ArrayList<Object> rivals = negoStats.getRivals();
            Bid tempBid = new Bid(bid);
            for(int i = 0; i < 100; i++) {
                for (Object rival : rivals) {
                    tempBid = getReplacedBidByAR(rival, tempBid);
                }

                if (info.getUtilitySpace().getUtility(bid) >= threshold) {
                    break;
                }
            }

            // 探索によって得られたBidがthresholdよりも小さい場合
            if (info.getUtilitySpace().getUtility(tempBid) < threshold) {
                return bid;
            } else {
                return tempBid;
            }

        } catch (Exception e) {
            System.out.println("[Exception_Search] Bidの探索に失敗しました");
            e.printStackTrace();
            return baseBid;
        }
    }

    // Bidの探索
    private static int SA_ITERATION = 1;
    /**
     * Bidの探索
     * @param baseBid
     * @param threshold
     * @return
     */
    private Bid getBidbyAppropriateSearch(Bid baseBid, double threshold) {
        Bid bid = new Bid(baseBid);
        try {
            // 線形効用空間用の探索
            if(negoStats.isLinerUtilitySpace()){
                bid = relativeUtilitySearch(threshold);

                // 探索に失敗した場合，非線形効用空間用の探索に切り替える
                if(info.getUtilitySpace().getUtility(bid) < threshold){
                    negoStats.utilSpaceTypeisNonLiner();
                }
            }

            // 非線形効用空間用の探索
            if(!negoStats.isLinerUtilitySpace()){
                Bid currentBid = null;
                double currentBidUtil = 0;
                double min = 1.0;
                for (int i = 0; i < SA_ITERATION; i++) {
                    currentBid = SimulatedAnnealingSearch(bid, threshold);
                    currentBidUtil = info.getUtilitySpace().getUtility(currentBid);
                    if (currentBidUtil <= min && currentBidUtil >= threshold) {
                        bid = new Bid(currentBid);
                        min = currentBidUtil;
                    }
                }
            }
        } catch (Exception e) {
            System.out.println("[Exception_Search] SA探索に失敗しました");
            System.out.println("[Exception_Search] Problem with received bid(SA:last):" + e.getMessage() + ". cancelling bidding");
        }
        return bid;
    }



    /**
     * 最大効用値Bidの初期探索(最初は効用空間のタイプが不明であるため，SAを用いて探索する)
     * @throws Exception
     */
    private void initMaxBid() throws Exception{
        int tryNum = info.getUtilitySpace().getDomain().getIssues().size(); // 試行回数

        Random rnd = new Random(info.getRandomSeed()); //Randomクラスのインスタンス化
        //maxBid = info.getUtilitySpace().getDomain().getRandomBid(rnd);
        maxBid = info.getUtilitySpace().getMaxUtilityBid();
        for (int i = 0; i < tryNum; i++) {
            try {
                do{
                    SimulatedAnnealingSearch(maxBid, 1.0);
                } while (info.getUtilitySpace().getUtility(maxBid) < info.getUtilitySpace().getReservationValue());
                if(info.getUtilitySpace().getUtility(maxBid) == 1.0){
                    break;
                }
            } catch (Exception e) {
                System.out.println("[Exception_Search] 最大効用値Bidの初期探索に失敗しました");
                e.printStackTrace();
            }
        }

        System.out.println("[isPrinting_Search]:" + maxBid.toString() + " " + info.getUtilitySpace().getUtility(maxBid));
    }

    /**
     * 論点ごとに最適化を行う探索
     * @param threshold
     * @return
     * @throws Exception
     */
    private Bid relativeUtilitySearch(double threshold) throws Exception{
        Bid bid = new Bid(maxBid);
        double d = threshold - 1.0; // 最大効用値との差
        double concessionSum = 0.0; // 減らした効用値の和
        double relativeUtility = 0.0;
        HashMap<Issue, HashMap<Value, Double>> valueRelativeUtility = negoStats.getValueRelativeUtility();
        ArrayList<Issue> randomIssues = (ArrayList<Issue>) negoStats.getIssues();
        Collections.shuffle(randomIssues);
        ArrayList<Value> randomValues = null;

        // シャップルされた論点毎に
        for(Issue issue:randomIssues){
            randomValues = negoStats.getValues(issue);
            Collections.shuffle(randomValues);

            // シャップルされた選択肢毎に
            for(Value value:randomValues){
                relativeUtility = valueRelativeUtility.get(issue).get(value); // 最大効用値を基準とした相対効用値
                if(d <= concessionSum + relativeUtility){
                    bid = bid.putValue(issue.getNumber(), value);
                    concessionSum += relativeUtility;
                    break;
                }
            }
        }
        return bid;
    }



    // SA
    static double START_TEMPERATURE = 1.0; // 開始温度
    static double END_TEMPERATURE = 0.0001; // 終了温度
    static double COOL = 0.999; // 冷却度
    static int STEP = 1;// 変更する幅
    static int STEP_NUM = 1; // 変更する回数
    /**
     * SA
     * @param baseBid
     * @param threshold
     * @return
     * @throws Exception
     */
    private Bid SimulatedAnnealingSearch(Bid baseBid, double threshold) throws Exception {
        Bid currentBid = new Bid(baseBid); // 初期解の生成
        double currenBidUtil = info.getUtilitySpace().getUtility(baseBid);
        Bid nextBid = null; // 評価Bid
        double nextBidUtil = 0.0;
        ArrayList<Bid> targetBids = new ArrayList<Bid>(); // 最適効用値BidのArrayList
        double targetBidUtil = 0.0;
        double p; // 遷移確率
        Random randomnr = new Random(); // 乱数
        double currentTemperature = START_TEMPERATURE; // 現在の温度
        double newCost = 1.0;
        double currentCost = 1.0;
        List<Issue> issues = negoStats.getIssues();

        // 温度が十分下がるまでループ
        while (currentTemperature > END_TEMPERATURE) {
            nextBid = new Bid(currentBid); // next_bidを初期化

            // 近傍のBidを取得する
            for (int i = 0; i < STEP_NUM; i++) {
                int issueIndex = randomnr.nextInt(issues.size()); // 論点をランダムに指定
                Issue issue = issues.get(issueIndex); // 指定したindexのissue
                ArrayList<Value> values = negoStats.getValues(issue);
                int valueIndex = randomnr.nextInt(values.size()); // 取り得る値の範囲でランダムに指定
                nextBid = nextBid.putValue(issue.getNumber(), values.get(valueIndex));
                nextBidUtil = info.getUtilitySpace().getUtility(nextBid);

                // 最大効用値Bidの更新
                if (maxBid == null || nextBidUtil >= info.getUtilitySpace().getUtility(maxBid)) {
                    maxBid = new Bid(nextBid);
                }
            }

            newCost = Math.abs(threshold - nextBidUtil);
            currentCost = Math.abs(threshold - currenBidUtil);
            p = Math.exp(-Math.abs(newCost - currentCost) / currentTemperature);
            if (newCost < currentCost || p > randomnr.nextDouble()) {
                currentBid = new Bid(nextBid); // Bidの更新
                currenBidUtil = nextBidUtil;
            }

            // 更新
            if (currenBidUtil >= threshold){
                if(targetBids.size() == 0){
                    targetBids.add(new Bid(currentBid));
                    targetBidUtil = info.getUtilitySpace().getUtility(currentBid);
                } else{
                    if(currenBidUtil < targetBidUtil){
                        targetBids.clear(); // 初期化
                        targetBids.add(new Bid(currentBid)); // 要素を追加
                        targetBidUtil = info.getUtilitySpace().getUtility(currentBid);
                    } else if (currenBidUtil == targetBidUtil){
                        targetBids.add(new Bid(currentBid)); // 要素を追加
                    }
                }
            }
            currentTemperature = currentTemperature * COOL; // 温度を下げる
        }

        if (targetBids.size() == 0) {
            // 境界値より大きな効用値を持つBidが見つからなかったときは，baseBidを返す
            return new Bid(baseBid);
        } else {
            // 効用値が境界値付近となるBidを返す
            return new Bid(targetBids.get(randomnr.nextInt(targetBids.size())));
        }
    }

    /**
     * AR (Agree / Reject)
     * @param sender
     * @param bid
     * @return
     */
    Bid getReplacedBidByAR(Object sender, Bid bid){
        Random rnd = new Random(info.getRandomSeed()); //Randomクラスのインスタンス化

        List<Issue> issues = negoStats.getIssues();
        for(Issue issue : issues) {
            double r = Math.random();
            HashMap<Value, ArrayList<Double>> cpr = negoStats.getCPRejectedValue(sender, issue);

            // もし累積Reject率における範囲に入った場合は置換
            if(cpr.get(bid.getValue(issue.getNumber())).get(0) < r
                    && cpr.get(bid.getValue(issue.getNumber())).get(1) >= r){

                double a = Math.random();
                HashMap<Value, ArrayList<Double>> cpa = negoStats.getCPAgreedValue(sender, issue);

                // 各valueについて置換先を確率的に決める
                ArrayList<Value> values = negoStats.getValues(issue);
                for(Value value : values){
                    if (cpa.get(value).get(0) < a && cpa.get(value).get(1) >= a){
                        bid = bid.putValue(issue.getNumber(), value);
                    }
                    break; // １つのvalueに何回も置換しない
                }

            }
        }

        return bid;
    }


}
