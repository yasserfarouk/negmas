package agents.anac.y2018.agent33.etc;

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
    private boolean isPrinting_Search = true;

    private NegoStats negoStats; // 交渉情報
    /**
     * 2018/5/27
     * 交渉戦略
     */
    private NegoStrategy negoStrategy;

    private NegoHistory negoHistory;
    private Bid maxBid = null; // 最大効用値Bid
    /**
     * 2018/5/20
     * reject率を拡大するための係数
     */
    double pow = 3.0; //各issueの各valueのrejectとagreeの確率を計算する時使う平方数
    double judgeReject = 0.8; //reject率がこれ以上の場合変換する
    double judgeAgree = 0.6; //agree率がこれ以上の場合変換する


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

        		double time = info.getTimeline().getTime();
        		double thresholdChange = 0.0;
        		double judgeNum = 0.1;//thresholdとの差分
            Bid bid = getBidbyAppropriateSearch(baseBid, threshold); // 閾値以上の効用値を持つ合意案候補を探索
            		/**
                	 * 2018/5/29
                	 * thresholdと比較
                	 */
                	for(int i = 0; i < 50; i++) {

                		bid = relativeUtilitySearch1(threshold);
                		//System.out.println("線形効用空間:bid  " + bid);
                    //System.out.println("線形効用空間　bid value " + info.getUtilitySpace().getUtility(bid));
                    //System.out.println("threshold:" + threshold);

                        //if (info.getUtilitySpace().getUtility(bid) >= threshold) {
                    /**
                     * 2018/6/1
                     * thresholdとの差分で判断
                     */
                    if (Math.abs(info.getUtilitySpace().getUtility(bid) - threshold) <= judgeNum) {
                            break;
                        }
                    }

        		//}
            /*
            // 探索によって得られたBidがthresholdよりも小さい場合，最大効用値Bidを基準とする
            if (info.getUtilitySpace().getUtility(bid) < threshold) {
                bid = new Bid(maxBid);
            }
             */

            ArrayList<Object> rivals = negoStats.getRivals();
            Bid tempBid = bid;
            //System.out.println("tempBid:" + tempBid);
            //System.out.println("bid utility and threshold:" + info.getUtilitySpace().getUtility(bid) + " and " + threshold);

            /**
             * 2018/5/31
             * bid valueがthresholdより小さい場合
             */
            //if (info.getUtilitySpace().getUtility(bid) < threshold) {
            if (Math.abs(info.getUtilitySpace().getUtility(bid) - threshold) > judgeNum) {
            /**
             * 2018/6/1
             * time <= 0.8 && time > 0.4
             */

            	thresholdChange = threshold - info.getUtilitySpace().getUtility(bid);
          	for(int i = 0; i < 50; i++) {
        		//bid = relativeUtilitySearch1(threshold);
        		//bid = relativeUtilitySearch1(negoStrategy.getThreshold(info.getTimeline().getTime()));
        		bid = relativeUtilitySearch1(threshold);
        		//System.out.println("線形効用空間:bid  " + bid);
            //System.out.println("線形効用空間　bid value " + info.getUtilitySpace().getUtility(bid));
            //System.out.println("thresholdChange:" + (threshold - thresholdChange));

                //if (info.getUtilitySpace().getUtility(bid) >= threshold - thresholdChange) {
            if (Math.abs(info.getUtilitySpace().getUtility(bid) - threshold) < judgeNum) {
                    break;
                }
            }

            }
            System.out.println("bid utility and thresholdChange:" + info.getUtilitySpace().getUtility(bid) + " and " + (threshold- thresholdChange));


            /**
             * 2018/5/20
             * bidの中にreject率高いissueのvalueをagree率の方を変更
             * 2018/5/25
             * time>0.7の時実行
             */
            System.out.println("time:" + time);

            if(time > 0.80) {
            	//if (info.getUtilitySpace().getUtility(bid) < threshold) {
            	if (Math.abs(info.getUtilitySpace().getUtility(bid) - threshold) >= judgeNum) {
            for(int i = 0; i < 30; i++) {
            		for (Object rival : rivals) {
            			tempBid = getReplacedBidByAR(rival,bid);
            		}
                //if (info.getUtilitySpace().getUtility(tempBid) >= threshold) {
            		if (Math.abs(info.getUtilitySpace().getUtility(bid) - threshold) < judgeNum) {
                		//System.out.println("bid find!!!!!!!");
                		if(info.getUtilitySpace().getUtility(tempBid) > info.getUtilitySpace().getUtility(bid)) {
                			bid = tempBid;
                		}
                		break;
                }
            }
            }
            }

            /**
             * 2018/5/29
             * thresholdと比較しない
             */
            /*
            if (info.getUtilitySpace().getUtility(bid) < threshold) {
                bid = new Bid(maxBid);
            }
            */
            return bid;


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
            	/**
            	 * 2018/5/27
            	 * 重要issueを除き探索
            	 */
            	//System.out.println("線形効用空間でbid探索が始まる");
            //bid = relativeUtilitySearch1(threshold);
            	//System.out.println("線形効用空間:bid  " + bid);
           	//System.out.println("線形効用空間　bid value " + info.getUtilitySpace().getUtility(bid));

            	/**
            	 * 2018/5/29
            	 * thresholdと比較
            	 */
            	/*
            	for(int i = 0; i < 50; i++) {
            		//bid = relativeUtilitySearch1(threshold);
            		//bid = relativeUtilitySearch1(negoStrategy.getThreshold(info.getTimeline().getTime()));
            		bid = relativeUtilitySearch1(threshold);
            		System.out.println("線形効用空間:bid  " + bid);
                System.out.println("線形効用空間　bid value " + info.getUtilitySpace().getUtility(bid));
                System.out.println("threshold:" + threshold);

                    if (info.getUtilitySpace().getUtility(bid) >= threshold) {
                        break;
                    }
                }
            	*/
                // 探索に失敗した場合，非線形効用空間用の探索に切り替える
            	/*
                if(info.getUtilitySpace().getUtility(bid) < threshold){
                    negoStats.utilSpaceTypeisNonLiner();
                }
                */
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
        //System.out.println("get bid:" + bid);
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

        //ArrayList<Value> randomValues = null;
        ArrayList<Value> randomValues = new ArrayList<Value>();

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

    /**
     * 2018/5/25
     * 重視issue以外の論点ごとに最適化を行う探索
     */
    private Bid relativeUtilitySearch1(double threshold) throws Exception{
    			/*
    			double time = info.getTimeline().getCurrentTime();
    			double threshold = negoStrategy.getThresholdForBidSearch(time);//bid探索のための譲歩関数
    			System.out.println("!!!!!!!!threshold:" + threshold);
    			*/
    			Bid bidPrior = new Bid(maxBid);
    			//System.out.println("valuePriorAll:" + negoStats.getValuePriorAll());
    			//重視issueと重視valueをbidPriorに入れる
    			//if(negoStats.getValuePriorAll() !=  null) {
    			if(!negoStats.getValuePriorAll().isEmpty()) {
    			for(Issue issue: negoStats.getValuePriorAll().keySet()) {
    				//System.out.println("issue:" + issue);
    				//System.out.println("issue.getNumber():" + issue.getNumber());
    				//System.out.println("negoStats.getValuePriorAll().get(issue).size():" + negoStats.getValuePriorAll().get(issue).size());
    				Random rand = new Random();
    				int r = rand.nextInt(11) % (negoStats.getValuePriorAll().get(issue).size());//valuelistからランダムにvalueを選ぶ
    				//System.out.println("r:" + r);
    				//System.out.println("negoStats.getValuePriorAll().get(issue).get(r):" + negoStats.getValuePriorAll().get(issue).get(r));
    				bidPrior = bidPrior.putValue(issue.getNumber(), negoStats.getValuePriorAll().get(issue).get(r));
    				//System.out.println("bidPrior:" + bidPrior);
    			}
    			}
            double d = threshold - 1.0; // 最大効用値との差
            double concessionSum = 0.0; // 減らした効用値の和
            double relativeUtility = 0.0;
            HashMap<Issue, HashMap<Value, Double>> valueRelativeUtility = negoStats.getValueRelativeUtility();
            ArrayList<Issue> randomIssues = (ArrayList<Issue>) negoStats.getIssues();
            Collections.shuffle(randomIssues);

           //ArrayList<Value> randomValues = null;
            ArrayList<Value> randomValues = new ArrayList<Value>(); //errorを解決する
            // シャップルされた論点毎に
            for(Issue issue:randomIssues){
            		//System.out.println("issue:" + issue);
                randomValues = negoStats.getValues(issue);
                Collections.shuffle(randomValues);

                //重視issueかどうか確認
                boolean isPriorIssue = false;
                //if(negoStats.getValuePriorAll() != null) {
                if(!negoStats.getValuePriorAll().isEmpty()) {
                for(Issue issuePrior: negoStats.getValuePriorAll().keySet()) {
                		if(issue == issuePrior) {
                			isPriorIssue = true;
                		}
                }
                }
                // シャップルされた選択肢毎に
                for(Value value:randomValues){
                    relativeUtility = valueRelativeUtility.get(issue).get(value); // 最大効用値を基準とした相対効用値
                    //System.out.println("d and concessionSum and relativeUtility:" + d + " and "+ concessionSum + " and " + relativeUtility);
                    /**
                     * 2018/5/29
                     *
                     */
                    if(d <= concessionSum + relativeUtility && !isPriorIssue){//重視issueの場合交換しない
                    //if(!isPriorIssue) {
                    //if(d <= concessionSum + relativeUtility) {
                        bidPrior = bidPrior.putValue(issue.getNumber(), value);
                        concessionSum += relativeUtility;
                        //System.out.println("d and concessionSum and relativeUtility:" + d + " and "+ concessionSum + " and " + relativeUtility);
                        break;
                    }
                }
            }
            //System.out.println("bidPrior:" + bidPrior);
            return bidPrior;

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
            /**
             * 2018/5/18
             */
            //HashMap<Value, ArrayList<Double>> cpr = negoStats.getCPRejectedValue1(sender, issue, pow);

            // もし累積Reject率における範囲に入った場合は置換
            if(cpr.get(bid.getValue(issue.getNumber())).get(0) < r
                    && cpr.get(bid.getValue(issue.getNumber())).get(1) >= r){
            		//System.out.println("startpoint:" + cpr.get(bid.getValue(issue.getNumber())).get(0));
            		//System.out.println("endpoint:" + cpr.get(bid.getValue(issue.getNumber())).get(1));
            		//System.out.println("r:" + r);
                double a = Math.random();
                HashMap<Value, ArrayList<Double>> cpa = negoStats.getCPAgreedValue(sender, issue);
                /**
                 * 2018/5/18
                 */
                //HashMap<Value, ArrayList<Double>> cpa = negoStats.getCPAgreedValue1(sender, issue, pow);

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

    /**
     * 2018/5/20
     * AR (Agree / Reject)
     * @param sender
     * @param bid
     * @return
     */
    Bid getReplacedBidByAR1(Object sender, Bid bid){
        Random rnd = new Random(info.getRandomSeed()); //Randomクラスのインスタンス化

        List<Issue> issues = negoStats.getIssues();
        for(Issue issue : issues) {
            //double r = Math.random();
            //HashMap<Value, ArrayList<Double>> cpr = negoStats.getCPRejectedValue(sender, issue);
            /**
             * 2018/5/18
             */
            HashMap<Value, ArrayList<Double>> cpr = negoStats.getCPRejectedValue1(sender, issue, pow);

            // もし累積Reject率における範囲に入った場合は置換

            if((cpr.get(bid.getValue(issue.getNumber())).get(1) - cpr.get(bid.getValue(issue.getNumber())).get(0)) > judgeReject) {
                double a = Math.random();
                //HashMap<Value, ArrayList<Double>> cpa = negoStats.getCPAgreedValue(sender, issue);
                /**
                 * 2018/5/18
                 */
                HashMap<Value, ArrayList<Double>> cpa = negoStats.getCPAgreedValue1(sender, issue, pow);

                // 各valueについて置換先を確率的に決める
                ArrayList<Value> values = negoStats.getValues(issue);
                for(Value value : values){

                		if((cpa.get(value).get(1) - cpa.get(value).get(0)) > judgeAgree) {
                        bid = bid.putValue(issue.getNumber(), value);
                        break; // １つのvalueに何回も置換しない
                    }
                }

            }
        }

        return bid;
    }

}
