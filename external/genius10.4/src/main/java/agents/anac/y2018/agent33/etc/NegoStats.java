 package agents.anac.y2018.agent33.etc;

import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.parties.NegotiationInfo;


public class NegoStats {
    private NegotiationInfo info;
    private boolean isPrinting = false; // デバッグ用
    private boolean isPrinting_Stats = false;

    // 交渉における基本情報
    private List<Issue> issues;
    private ArrayList<Object> rivals;
    private int negotiatorNum = 0; // 交渉者数
    private boolean isLinerUtilitySpace = true; // 線形効用空間であるかどうか
    private HashMap<Issue, HashMap<Value, Double>> valueRelativeUtility = null; // 自身の効用空間における各論点値の相対効用値行列（線形効用空間用）
    /**
     * 2015/5/22　頻度を小数にする
     * 重みをつけるための係数
     */
    private HashMap<Object, HashMap<Issue, HashMap<Value, Double>>> agreedValueFrequency1   = null;
    private double weightFrequencyRate = 0.01;
    private double weightF = 1.0;
    /**
     * 2015/5/22-----2
     * agree valueのカウント個数を時間の長さで決める
     */
    private double countTime = 0.25;


    // current session の提案履歴
    private ArrayList<Bid> myBidHist = null;
    private HashMap<Object, ArrayList<Bid>> rivalsBidHist = null;
    private HashMap<Object, HashMap<Issue, HashMap<Value, Integer>>> agreedValueFrequency   = null;


    private HashMap<Object, HashMap<Issue, HashMap<Value, Integer>>> rejectedValueFrequency = null;


    // 交渉相手の統計情報
    private HashMap<Object, Double> rivalsMean      = null;
    private HashMap<Object, Double> rivalsVar       = null;
    private HashMap<Object, Double> rivalsSum       = null;
    private HashMap<Object, Double> rivalsPowSum    = null;
    private HashMap<Object, Double> rivalsSD        = null;

    private HashMap<Object, Double> rivalsMax       = null; // 今sessionにおける相手の提案に対する自身の最大効用値
    private HashMap<Object, Double> rivalsMin       = null; // 今sessionにおける相手の提案に対する自身の最低効用値

    /**
     * 2018/5/23
     * 相手の重視情報
     */
    private HashMap<Object, ArrayList<HashMap<Issue,Value>>> rivalsValuePrior = null;//今sessionにおける相手の提案に対する相手の重視value
    private HashMap<Object,ArrayList<HashMap<Issue,Value>>> valuePrior = null; //今session相手の重視issueとそのissueの最重視valueのlist
    private HashMap<Object,HashMap<Issue,Double>> valueAgreeRateSD = null;//今session相手の各issueのvalueAgreeRateの標準偏差
    //private HashMap<Issue,Value> issueValuePrior_tmp = null;

    /**
     * 2018/5/24
     */
    private HashMap<Issue,ArrayList<Value>> valuePriorAll = null;//今sessionにおける全相手が重視するissueとその重視value

    public NegoStats(NegotiationInfo info, boolean isPrinting){
        this.info = info;
        this.isPrinting = isPrinting;

        // 交渉における基本情報
        issues = info.getUtilitySpace().getDomain().getIssues();
        rivals = new ArrayList<Object>();
        valueRelativeUtility = new HashMap<Issue, HashMap<Value, Double>> ();

        // current session の提案履歴
        myBidHist       = new ArrayList<Bid>();
        rivalsBidHist   = new HashMap<Object, ArrayList<Bid>>();
        agreedValueFrequency   = new HashMap<Object, HashMap<Issue, HashMap<Value, Integer>>> (); // Value毎のAccept数
        rejectedValueFrequency = new HashMap<Object, HashMap<Issue, HashMap<Value, Integer>>> (); // Value毎のReject数

        /**
         * 2018/5/29
         * 相手の重視情報
         */
        rivalsValuePrior = new HashMap<Object, ArrayList<HashMap<Issue,Value>>>();
        valuePrior = new HashMap<Object,ArrayList<HashMap<Issue,Value>>>();
        valueAgreeRateSD = new HashMap<Object,HashMap<Issue,Double>>();

        try {
            initValueRelativeUtility();
        } catch (Exception e) {
            System.out.println("[Exception_Stats] 相対効用行列の初期化に失敗しました");
            e.printStackTrace();
        }

        // 交渉相手の統計情報
        rivalsMean      = new HashMap<Object, Double>();
        rivalsVar       = new HashMap<Object, Double>();
        rivalsSum       = new HashMap<Object, Double>();
        rivalsPowSum    = new HashMap<Object, Double>();
        rivalsSD        = new HashMap<Object, Double>();

        rivalsMax       = new HashMap<Object, Double>();
        rivalsMin       = new HashMap<Object, Double>();

        if(this.isPrinting){
            System.out.println("[isPrinting] NegoStats: success");
        }

    }

    public void initRivals(Object sender) {
    		System.out.println("initRivalsが始まる");
        initNegotiatingInfo(sender); // 交渉情報を初期化
        rivals.add(sender); // 交渉参加者にsenderを追加
    }

    public void updateInfo(Object sender, Bid offeredBid) {
        try {
        	//System.out.println("??????????????????????");
            updateNegoStats(sender, offeredBid); // 交渉情報の更新

        } catch (Exception e1) {
            System.out.println("[Exception_Stats] 交渉情報の更新に失敗しました");
            e1.printStackTrace();
        }
    }

    private void initNegotiatingInfo(Object sender) {
        rivalsBidHist.put(sender, new ArrayList<Bid>());
    		//rivalsBidHist.put(sender, null);
        //System.out.println("rivalsBidHist:" + rivalsBidHist);
        rivalsMean.put(sender, 0.0);
        rivalsVar.put(sender, 0.0);
        rivalsSum.put(sender, 0.0);
        rivalsPowSum.put(sender, 0.0);
        rivalsSD.put(sender, 0.0);

        rivalsMax.put(sender, 0.0);
        rivalsMin.put(sender, 1.0);

        /**
         * 2018/5/23
         * 重要valueの初期化
         */
        //System.out.println("initNegotiatingInfo");
        rivalsValuePrior.put(sender, new ArrayList<HashMap<Issue,Value>>());
        //System.out.println("rivalsValuePrior:" + rivalsValuePrior);
        valuePrior.put(sender, new ArrayList<HashMap<Issue,Value>>());
        //System.out.println("valuePrior:" + valuePrior);
    }

    /**
     * 交渉者数を更新する
     * @param num
     */
    public void updateNegotiatorsNum(int num) {
        negotiatorNum = num;
    }


    /**
     * 線形効用空間でない場合
     */
    public void utilSpaceTypeisNonLiner() {
        isLinerUtilitySpace = false;
    }

    /**
     * 相対効用行列の初期化
     * @throws Exception
     */
    private void initValueRelativeUtility() throws Exception{
        //ArrayList<Value> values = null;
        ArrayList<Object> rivals = getRivals();

        for(Issue issue:issues){
            valueRelativeUtility.put(issue, new HashMap<Value, Double>()); // 論点行の初期化

            // 論点行の要素の初期化
            ArrayList<Value> values = getValues(issue);
            for(Value value:values){
                valueRelativeUtility.get(issue).put(value, 0.0);
            }
        }
    }

    // 相対効用行列の導出
    public void setValueRelativeUtility(Bid maxBid) throws Exception {
        ArrayList<Value> values = null;
        Bid currentBid = null;
        for(Issue issue:issues){
            currentBid = new Bid(maxBid);
            values = getValues(issue);
            for(Value value:values){
                currentBid = currentBid.putValue(issue.getNumber(), value);
                valueRelativeUtility.get(issue).put(value,
                        info.getUtilitySpace().getUtility(currentBid) - info.getUtilitySpace().getUtility(maxBid));
            }
        }
        System.out.println("SetValueRelatility OK!");
    }

    /**
     * Agent senderが受け入れたValueの頻度を更新
     * @param sender
     * @param bid
     */
    public void updateAgreedValues(Object sender, Bid bid){
        // senderが過去に登場していない場合は初期化
        if(!agreedValueFrequency.containsKey(sender)){
            agreedValueFrequency.put(sender, new HashMap<Issue, HashMap<Value, Integer>> ());

            for(Issue issue : issues){
                agreedValueFrequency.get(sender).put(issue, new HashMap<Value, Integer>());

                ArrayList<Value> values = getValues(issue);
                for(Value value : values){
                    agreedValueFrequency.get(sender).get(issue).put(value, 0);
                }
            }
        }
        /**
         * 2018/5/22----2
         * 最初の30sの個数だけカウント
         */
        System.out.println("+++++++++time:" + info.getTimeline().getTime());
        //if(info.getTimeline().getTime() <= countTime) {
        // 各issue毎に個数をカウント
        for(Issue issue : issues) {
            Value value = bid.getValue(issue.getNumber());
              agreedValueFrequency.get(sender).get(issue).put(value, agreedValueFrequency.get(sender).get(issue).get(value) + 1);
        }
       // }

        if(isPrinting_Stats){
            System.out.println("[isPrint_Stats] (ACCEPT) " + sender.toString() + ":");
            for(Issue issue : issues){
                ArrayList<Value> values = getValues(issue);
                for(Value value : values){
                    System.out.print(agreedValueFrequency.get(sender).get(issue).get(value) + " ");
                }
                System.out.println();
            }


            getMostAgreedValues(sender);
        }
    }

    /**
     * 2018/5/23
     * 各agentに重視issueとその最重視valueを探す
     */
	public void valuePriorSearch(Object sender) {

    	double issuePriorSD1; //重要度１のissueの標準偏差
    	double issuePriorSD2; //重要度2のissueの標準偏差

    //	HashMap<Issue,Value> issuePrior = new HashMap<Issue,Value>();//重視issue
    	//System.out.println("issuePrior:" + issuePrior);
    	ArrayList<HashMap<Issue,Value>> issueValuePrior = new ArrayList<HashMap<Issue,Value>>();//各issueとその最重視valueのlist
    //	System.out.println("issueValuePrior:" + issueValuePrior);
    	valuePrior.put(sender, new ArrayList<HashMap<Issue,Value>>()); //今session相手の重視issueとそのissueの最重視valueのlistの初期化

    //	System.out.println("issues:" + issues);
    	//各issueとその最重視valueを探す
    	for(Issue issue: issues) {
    		ArrayList<Value> values = getValues(issue);

    		//issueごとに最重視valueを返す
    		double valueAgreeRate_tmp = 0.0;
    		HashMap<Issue, Value> issueValuePrior_tmp = new HashMap<Issue,Value>();

			for(Value value: values) {
    			//issueの最大valueを調べる
    			if(agreedValueFrequency.get(sender).get(issue).get(value) > valueAgreeRate_tmp) {
    				valueAgreeRate_tmp = agreedValueFrequency.get(sender).get(issue).get(value);
    				issueValuePrior_tmp.put(issue, value);
    			}
    		}
    		//issueとその最大valueのセットをlistに追加
    		issueValuePrior.add(issueValuePrior_tmp);
    	}

    //	System.out.println("issueValuePrior:" + issueValuePrior);
    	issuePriorSD1 = 	selectPriorIssueValue(sender, issueValuePrior, 0.0);
    	if(issues.size() > 2) {//issueが２個以上の場合
    	issuePriorSD2 = 	selectPriorIssueValue(sender, issueValuePrior, issuePriorSD1);
    	}
    }

    /**
     * 2018/5/23
     * 重視issueとその最重視valueをlistに追加
     */
	//public double selectPriorIssueValue(Object sender, HashMap<Issue,Value> issuePrior,
    	//	ArrayList<HashMap<Issue,Value>> issueValuePrior, double issuePriorSD1){
	public double selectPriorIssueValue(Object sender,
		ArrayList<HashMap<Issue,Value>> issueValuePrior, double issuePriorSD1){

    	double compareSD = 0.0;
    	/**2018/5/30
    	 *
    	 */
    	HashMap<Issue,Value> issuePrior = new HashMap<Issue,Value>();//重視issue
    //	issuePrior = null;
    	System.out.println("issuePrior:" + issuePrior);


    	for(Issue issue: issues) {
    		ArrayList<Value> values = getValues(issue);

    		double sumValueAgreeRate = 0.0; //issueごとに全てのvalueAgreeRateの和
    		double valueAgreeRateMean = 0.0; //issueごとに全てのvalueAgreeRateの平均
    		double valueAgreeRateSD_tmp = 0.0; //issueごとに全てのvalueAgreeRateの標準偏差
    		double issueSize = agreedValueFrequency.get(sender).get(issue).size();//issueの中のvalue個数
    		HashMap<Object,HashMap<Issue,Double>> valueAgreeRateSD = new HashMap<Object,HashMap<Issue,Double>>();

    		//issueごとに全てのvalueAgreeRateの和を求める
    		for(Value value: values) {
    			sumValueAgreeRate += agreedValueFrequency.get(sender).get(issue).get(value);
    			}

    		//issueごとにvalueAgreeRateの平均を求める
    		for(Value value: values) {
    			valueAgreeRateMean += agreedValueFrequency.get(sender).get(issue).get(value) / sumValueAgreeRate;
    		}
    		valueAgreeRateMean /=  issueSize;

    		//issueごとにvalueAgreeRateの標準偏差を求める
    		for(Value value: values) {
    			valueAgreeRateSD_tmp += Math.pow((agreedValueFrequency.get(sender).get(issue).get(value) - valueAgreeRateMean),2);
    		}
    		valueAgreeRateSD_tmp /= issueSize;
    		valueAgreeRateSD_tmp = Math.sqrt(valueAgreeRateSD_tmp);

    		//issueとその標準偏差をセット
    		System.out.println("valueAgreeRateSD.get(sender):" + valueAgreeRateSD.get(sender));
    		HashMap<Issue, Double> issueSD = new HashMap<Issue, Double>();
    		//System.out.println("issueSD:" + issueSD);
    		issueSD.put(issue,valueAgreeRateSD_tmp);
    		valueAgreeRateSD.put(sender, issueSD);
    		//System.out.println("valueAgreeRateSD.get(" + sender + "):" + valueAgreeRateSD.get(sender));
    		//System.out.println("valueAgreeRateSD_tmp:" + valueAgreeRateSD_tmp);
    		//System.out.println("compareSD:" + compareSD);
    		//System.out.println("issuePriorSD1:" + issuePriorSD1);

    		//標準偏差が最大のissueとその最重視valueをpick out
    		//System.out.println("issuePrior:" + issuePrior);
    		//System.out.println("issue:" + issue);
    		//System.out.println("valuePrior.get(sender):" + valuePrior.get(sender));
		if(valueAgreeRateSD_tmp > compareSD && issuePriorSD1 != valueAgreeRateSD_tmp) //1回目探索するとき
				 { //2番目のissueを探索するとき、１回目を回避
    		//if(valueAgreeRateSD_tmp > compareSD  && issuePriorSD1 != valueAgreeRateSD_tmp) {
    			compareSD = valueAgreeRateSD_tmp;

    			//issuePrior = issueValuePrior.get(issue.getNumber());
    			/**
    			 * 2018/5/30
    			 * issueの順番が違うため
    			 * issue.getNumber()使えない
    			 */
    			for (HashMap<Issue, Value> issueValuePriorOne: issueValuePrior) {
    				if(issueValuePriorOne.containsKey(issue)) {
    					issuePrior.clear();
    					issuePrior.put(issue, issueValuePriorOne.get(issue));
    					System.out.println("issuePrior:" + issuePrior);
    					break;
    				}
    				}
    		}
    	}

    	//重視issueとその最重視valueを保存
    	//System.out.println("valuePrior.get(sender):" + valuePrior.get(sender));
    ArrayList<HashMap<Issue,Value>> valuePriorGet = new ArrayList<HashMap<Issue,Value>>();
    	valuePriorGet = valuePrior.get(sender);
    	//System.out.println("valuePriorGet:" + valuePriorGet);
    	//System.out.println("issuePrior:" + issuePrior);
    	if(!issuePrior.isEmpty()) {
    	//System.out.println("&&&&&&&&&&&&&&&");
    	valuePriorGet.add(issuePrior);
    	}
    //	System.out.println("valuePriorGet:" + valuePriorGet);


    	valuePrior.put(sender, valuePriorGet);
    	//System.out.println("valuePrior:" + valuePrior);


    	return compareSD;
    }


    /**
     * 2018/5/22
     * Agent senderが受け入れたValueの頻度を更新
     * 頻度も時間とともに重み付きで減少、小数となる
     */
    public void updateAgreedValues1(Object sender, Bid bid){
        // senderが過去に登場していない場合は初期化
        if(!agreedValueFrequency1.containsKey(sender)){
            agreedValueFrequency1.put(sender, new HashMap<Issue, HashMap<Value, Double>> ());

            for(Issue issue : issues){
                agreedValueFrequency1.get(sender).put(issue, new HashMap<Value, Double>());

                ArrayList<Value> values = getValues(issue);
                for(Value value : values){
                    agreedValueFrequency1.get(sender).get(issue).put(value, 0.0);
                }
            }
        }

        // 各issue毎に個数をカウント
        for(Issue issue : issues) {
            Value value = bid.getValue(issue.getNumber());
              agreedValueFrequency1.get(sender).get(issue).put(value, agreedValueFrequency1.get(sender).get(issue).get(value) + weightF);
        }

        if(weightF > 0 ) {
        weightF = weightF - weightFrequencyRate; //頻度の重みを更新
        }

        if(isPrinting_Stats){
            System.out.println("[isPrint_Stats] (ACCEPT) " + sender.toString() + ":");
            for(Issue issue : issues){
                ArrayList<Value> values = getValues(issue);
                for(Value value : values){
                    System.out.print(agreedValueFrequency1.get(sender).get(issue).get(value) + " ");
                }
                System.out.println();
            }

            getMostAgreedValues(sender);
        }
    }



    /**
     * senderが拒絶したValueの頻度を更新
     * @param sender
     * @param bid
     */
    public void updateRejectedValues(Object sender, Bid bid){
        // senderが過去に登場していない場合は初期化
        if(!rejectedValueFrequency.containsKey(sender)){
            rejectedValueFrequency.put(sender, new HashMap<Issue, HashMap<Value, Integer>> ());

            for(Issue issue : issues){
                rejectedValueFrequency.get(sender).put(issue, new HashMap<Value, Integer>());

                ArrayList<Value> values = getValues(issue);
                for(Value value : values){
                    rejectedValueFrequency.get(sender).get(issue).put(value, 0);
                }
            }
        }

        /**
         * 2018/5/22
         * 最初の30sの個数だけカウント
         */
        // 各issue毎に個数をカウント
        for(Issue issue : issues) {
            Value value = bid.getValue(issue.getNumber());
            rejectedValueFrequency.get(sender).get(issue).put(value, rejectedValueFrequency.get(sender).get(issue).get(value) + 1);
        }

        if(isPrinting_Stats){
            System.out.println("[isPrint_Stats] (REJECT) " + sender.toString() + ":");
            for(Issue issue : issues){
                ArrayList<Value> values = getValues(issue);
                for(Value value : values){
                    System.out.print(rejectedValueFrequency.get(sender).get(issue).get(value) + " ");
                }
                System.out.println();
            }

            getMostRejectedValues(sender);
        }
    }


    public void updateNegoStats(Object sender, Bid offeredBid) throws Exception {
        // current session の提案履歴 への追加

        rivalsBidHist.get(sender).add(offeredBid);
        //System.out.println("rivalsBidHist.get(" + sender + "):" +  rivalsBidHist.get(sender));
        //System.out.println("???????????????");
        updateAgreedValues(sender, offeredBid);

        /**
         * 2018/5/23
         * 交渉相手の重要value情報を更新
         * ナッシュ解に近いbidを探す
         */
        valuePriorSearch(sender);//valuePriorを更新

        //valuePriorAllを初期化する
        //valuePriorAllInit(valuePriorAll);

        /**
         * 2018/5/25
         * valuePriorAllを更新する
         * valuePriorAllとは全ての重視issueとそのvaluelistのセット
         */
        valueIssueSetGet();



        // 交渉相手の統計情報 の更新
        double util = info.getUtilitySpace().getUtility(offeredBid);
        rivalsSum.put(sender, rivalsSum.get(sender) + util); // 和
        rivalsPowSum.put(sender, rivalsPowSum.get(sender) + Math.pow(util, 2)); // 二乗和

        int round_num = rivalsBidHist.get(sender).size();
        rivalsMean.put(sender, rivalsSum.get(sender) / round_num); // 平均
        rivalsVar.put(sender, (rivalsPowSum.get(sender) / round_num) - Math.pow(rivalsMean.get(sender), 2)); // 分散

        if(rivalsVar.get(sender) < 0){rivalsVar.put(sender, 0.0);}
        rivalsSD.put(sender, Math.sqrt(rivalsVar.get(sender))); // 標準偏差


        // 最大最小の更新
        if(util > rivalsMax.get(sender)){
            rivalsMax.put(sender, util);
        } else if (util < rivalsMin.get(sender)){
            rivalsMin.put(sender, util);
        }

        if(isPrinting_Stats){
            System.out.println("[isPrint_Stats] Mean: " + getRivalMean(sender) + " (Agent: " + sender.toString() + ")");
        }
    }

    /**
     * 自身の提案情報の更新
     * @param offerBid
     */
    public void updateMyBidHist(Bid offerBid) {
        myBidHist.add(offerBid);
    }


    // 交渉における基本情報 の取得
    /**
     * 論点一覧を返す
     * @return
     */
    public List<Issue> getIssues() {
        return issues;
    }

    /**
     * 交渉相手の一覧を返す
     * @return
     */
    public ArrayList<Object> getRivals() {
        return rivals;
    }

    /**
     * 交渉者数（自身を含む）を返す
     * @return
     */
    public int getNegotiatorNum(){
        // + 1: 自分
        return rivals.size() + 1;
    }

    /**
     * 論点における取り得る値の一覧を返す
     * @param issue
     * @return
     */
    public ArrayList<Value> getValues(Issue issue) {
        ArrayList<Value> values = new ArrayList<Value>();

        // 効用情報のtype毎に処理が異なる
        switch(issue.getType()) {
            case DISCRETE:
                List<ValueDiscrete> valuesDis = ((IssueDiscrete)issue).getValues();
                for(Value value:valuesDis){
                    values.add(value);
                }
                break;
            case INTEGER:
                int min_value = ((IssueInteger)issue).getUpperBound();
                int max_value = ((IssueInteger)issue).getUpperBound();
                for(int j=min_value; j<=max_value; j++){
                    Object valueObject = new Integer(j);
                    values.add((Value)valueObject);
                }
                break;
            default:
                try {
                    throw new Exception("issue type \""+ issue.getType() + "\" not supported by" + info.getAgentID().getName());
                } catch (Exception e) {
                    System.out.println("[Exception] 論点の取り得る値の取得に失敗しました");
                    e.printStackTrace();
                }
        }

        return values;
    }

    /**
     * 線形効用空間であるかどうかを返す
     * @return
     */
    public boolean isLinerUtilitySpace () {
        return isLinerUtilitySpace;
    }

    /**
     * 相対効用行列を返す
     * @return
     */
    public HashMap<Issue, HashMap<Value, Double>> getValueRelativeUtility(){
        return valueRelativeUtility;
    }

    /**
     * 2018/5/25
     * valuePriorAll:issueとそのvaluelistのセットを返す
     */
    public HashMap<Issue,ArrayList<Value>> getValuePriorAll(){
    		return valuePriorAll;
    }

    // current session の提案履歴 の取得

    /**
     * エージェントsenderの論点issueにおける最大Agree数となる選択肢valueを取得
     * @param sender
     * @param issue
     * @return
     */
    public Value getMostAgreedValue (Object sender, Issue issue){
        ArrayList<Value> values = getValues(issue);

        int maxN = 0;
        Value mostAgreedValue = values.get(0);
        for (Value value : values){
            int tempN = agreedValueFrequency.get(sender).get(issue).get(value);
            // もし最大数が更新されたら
            if(maxN < tempN){
                maxN = tempN;
                mostAgreedValue = value;
            }
        }

        return mostAgreedValue;
    }

    /**2018/5/18
     * エージェントsenderの論点issueにおける選択肢valueとそのAgree率を返す
     * @param sender
     * @param issue
     * @param value
     * @return
     */
    public HashMap<Value,Double> getProbAgreedValue1 (Object sender, Issue issue, Value value){
        ArrayList<Value> values = getValues(issue);
        HashMap<Value,Double> prob = new HashMap<Value, Double>();

        int sum = 0;
        for(Value v : values){
            sum += agreedValueFrequency.get(sender).get(issue).get(v);
        }

        prob.put(value,agreedValueFrequency.get(sender).get(issue).get(value) * 1.0 / sum);

        return prob;
    }

    /**
     * 2018/5/22
     * エージェントsenderの論点issueにおける選択肢valueとそのAgree率を返す
     * Agree率を返すとき、時間とともに重み付きで減少
     */
    public HashMap<Value,Double> getProbAgreedValue2 (Object sender, Issue issue, Value value){
        ArrayList<Value> values = getValues(issue);
        HashMap<Value,Double> prob = new HashMap<Value, Double>();

        double sum = 0;
        for(Value v : values){
            sum += agreedValueFrequency.get(sender).get(issue).get(v);
        }

        prob.put(value,agreedValueFrequency.get(sender).get(issue).get(value) / sum);

        return prob;
    }


    /**
     * エージェントsenderの論点issueにおける選択肢valueのAgree率を返す
     * @param sender
     * @param issue
     * @param value
     * @return
     */
    public double getProbAgreedValue (Object sender, Issue issue, Value value){
        ArrayList<Value> values = getValues(issue);

        int sum = 0;
        for(Value v : values){
            sum += agreedValueFrequency.get(sender).get(issue).get(v);
        }

        double prob = agreedValueFrequency.get(sender).get(issue).get(value) * 1.0 / sum;
        return prob;
    }




    /**
     * 2018/5/18
     * エージェントSenderにおける各論点における累積確率(CP)を取得
     * 同意を受けたvalueの特徴を拡大する
     * @param sender
     * @param issue
     * @return
     */
    public HashMap<Value, ArrayList<Double>> getCPAgreedValue1 (Object sender, Issue issue, double pow){
        HashMap<Value, ArrayList<Double>> CPMap = new HashMap<Value, ArrayList<Double>>();

        ArrayList<Value> values = getValues(issue);

        double sumOfProb = 0.0;
        for(Value value : values){
        	double prob = getProbAgreedValue1(sender,issue,value).get(value);
            sumOfProb += Math.pow(prob, pow);
        }

        double tempCP1 = 0.0;
        for(Value value : values) {
        		ArrayList<Double> tempArray = new ArrayList<Double> ();

        		HashMap<Value,Double> probUpdate = new HashMap<Value, Double>();
        		double prob = getProbAgreedValue1(sender,issue,value).get(value);
        		probUpdate.put(value, Math.pow(prob,pow) / sumOfProb);

            // 範囲のStartを格納
            tempArray.add(tempCP1);

            // 範囲のEndを格納
            tempCP1 += probUpdate.get(value);
            tempArray.add(tempCP1);

            CPMap.put(value, tempArray);
        }

        return CPMap;
    }


    /**
     * エージェントSenderにおける各論点における累積確率(CP)を取得
     * @param sender
     * @param issue
     * @return
     */
    public HashMap<Value, ArrayList<Double>> getCPAgreedValue (Object sender, Issue issue){
        HashMap<Value, ArrayList<Double>> CPMap = new HashMap<Value, ArrayList<Double>>();

        ArrayList<Value> values = getValues(issue);
        int sum = 0;
        for(Value value : values){
            sum += agreedValueFrequency.get(sender).get(issue).get(value);
        }

        double tempCP = 0.0;
        for(Value value : values){
            ArrayList<Double> tempArray = new ArrayList<Double> ();
            // 範囲のStartを格納
            tempArray.add(tempCP);

            // 範囲のEndを格納
            tempCP += agreedValueFrequency.get(sender).get(issue).get(value) * 1.0 / sum;
            tempArray.add(tempCP);

            CPMap.put(value, tempArray);
        }

        return CPMap;
    }



    /**
     * エージェントSenderにおける各論点における最大Agree数となる選択肢valueをArrayListで取得
     * @param sender
     * @return
     */
    public ArrayList<Value> getMostAgreedValues (Object sender){
        ArrayList<Value> values = new ArrayList<Value>();

        // issueの内部的な順番はa-b-c-d-...じゃないので注意
        for (Issue issue : issues){
            values.add(getMostAgreedValue(sender, issue));
        }


        if(isPrinting_Stats){
            System.out.print("[isPrint_Stats] ");
            for(int i = 0; i < issues.size(); i++){
                //System.out.print(issues.get(i).toString() + ":" + values.get(issues.get(i).getNumber()-1) + " ");
                //System.out.print(issues.get(i).toString() + ":" + values.get(i) + "(" + getProbAgreedValue(sender,issues.get(i),values.get(i)) + ") ");

                HashMap<Value, ArrayList<Double>> cp = getCPAgreedValue(sender, issues.get(i));

                System.out.print(issues.get(i).toString() + ":"
                        + values.get(i) + "(" + cp.get(values.get(i)).get(0) + " - " + cp.get(values.get(i)).get(1)  + ") ");
            }
            System.out.println();
        }

        return values;
    }


    /**
     * エージェントsenderの論点issueにおける最大Reject数となる選択肢valueを取得
     * @param sender
     * @param issue
     * @return
     */
    public Value getMostRejectedValue (Object sender, Issue issue){
        ArrayList<Value> values = getValues(issue);

        int maxN = 0;
        Value mostRejectedValue = values.get(0);
        for (Value value : values){
            int tempN = rejectedValueFrequency.get(sender).get(issue).get(value);
            // もし最大数が更新されたら
            if(maxN < tempN){
                maxN = tempN;
                mostRejectedValue = value;
            }
        }

        return mostRejectedValue;
    }


    /**
     * 2018/5/18
     * エージェントsenderの論点issueにおける選択肢valueとそのRejected率を返す
     * @param sender
     * @param issue
     * @param value
     * @return
     */
    public HashMap<Value,Double> getProbRejectedValue1 (Object sender, Issue issue, Value value){
        ArrayList<Value> values = getValues(issue);
        HashMap<Value,Double> probHash = new HashMap<Value, Double>();

        int sum = 0;
        for(Value v : values){
            sum += rejectedValueFrequency.get(sender).get(issue).get(v);
        }

        probHash.put(value,rejectedValueFrequency.get(sender).get(issue).get(value) * 1.0 / sum);

        return probHash;
    }


    /**
     * 2018/5/18
     * エージェントSenderにおける各論点における累積確率(CP)を取得
     * 同意を受けたvalueの特徴を拡大する
     * @param sender
     * @param issue
     * @return
     */
    public HashMap<Value, ArrayList<Double>> getCPRejectedValue1 (Object sender, Issue issue, double pow){
        HashMap<Value, ArrayList<Double>> CPMap = new HashMap<Value, ArrayList<Double>>();

        ArrayList<Value> values = getValues(issue);

        double sumOfProb = 0.0;
        for(Value value : values){
        	double prob = getProbRejectedValue1(sender,issue,value).get(value);
            sumOfProb += Math.pow(prob, pow);
            //System.out.println("+++++++++++" + sumOfProb);
        }

        double tempCP1 = 0.0;
        for(Value value : values) {
        		ArrayList<Double> tempArray = new ArrayList<Double> ();

        		HashMap<Value,Double> probUpdate = new HashMap<Value, Double>();
        		double prob = getProbRejectedValue1(sender,issue,value).get(value);
        		probUpdate.put(value, Math.pow(prob,pow) / sumOfProb);

            // 範囲のStartを格納
            tempArray.add(tempCP1);

            // 範囲のEndを格納
            //System.out.println(value + "：" + tempCP1);
            tempCP1 += probUpdate.get(value);

            tempArray.add(tempCP1);

            CPMap.put(value, tempArray);
        }
        //System.out.println("区切り");

        return CPMap;
    }


    /**
     * エージェントsenderの論点issueにおける選択肢valueのReject率を返す
     * @param sender
     * @param issue
     * @param value
     * @return
     */
    public double getProbRejectedValue (Object sender, Issue issue, Value value){
        ArrayList<Value> values = getValues(issue);

        int sum = 0;
        for(Value v : values){
            sum += rejectedValueFrequency.get(sender).get(issue).get(v);
        }

        double prob = rejectedValueFrequency.get(sender).get(issue).get(value) * 1.0 / sum;
        return prob;
    }

    /**
     * エージェントSenderにおける各論点における累積確率(CP)を取得
     * @param sender
     * @param issue
     * @return
     */
    public HashMap<Value, ArrayList<Double>> getCPRejectedValue (Object sender, Issue issue){
        HashMap<Value, ArrayList<Double>> CPMap = new HashMap<Value, ArrayList<Double>>();

        ArrayList<Value> values = getValues(issue);
        int sum = 0;
        for(Value value : values){
            sum += rejectedValueFrequency.get(sender).get(issue).get(value);
        		//System.out.println("+++++++++++" + sum);
        }
        //System.out.println("区切り");


        double tempCP = 0.0;
        for(Value value : values){
            ArrayList<Double> tempArray = new ArrayList<Double> ();
            // 範囲のStartを格納
            tempArray.add(tempCP);

            //System.out.println(value + "：" + tempCP);
            // 範囲のEndを格納
            tempCP += rejectedValueFrequency.get(sender).get(issue).get(value) * 1.0 / sum;

            tempArray.add(tempCP);

            CPMap.put(value, tempArray);
        }
        //System.out.println("区切り");

        return CPMap;
    }

    /**
     * エージェントSenderにおける各論点における最大Reject数となる選択肢valueをArrayListで取得
     * @param sender
     * @return
     */
    public ArrayList<Value> getMostRejectedValues (Object sender){
        ArrayList<Value> values = new ArrayList<Value>();

        // issueの内部的な順番はa-b-c-d-...じゃないので注意
        for (Issue issue : issues){
            values.add(getMostRejectedValue(sender, issue));
        }


        if(isPrinting_Stats){
            System.out.print("[isPrint_Stats] ");
            for(int i = 0; i < issues.size(); i++){
                //System.out.print(issues.get(i).toString() + ":" + values.get(issues.get(i).getNumber()-1) + " ");
                //System.out.print(issues.get(i).toString() + ":" + values.get(i) + "(" + getProbRejectedValue(sender,issues.get(i),values.get(i)) + ") ");

                HashMap<Value, ArrayList<Double>> cp = getCPRejectedValue(sender, issues.get(i));

                System.out.print(issues.get(i).toString() + ":"
                        + values.get(i) + "(" + cp.get(values.get(i)).get(0) + " - " + cp.get(values.get(i)).get(1)  + ") ");
            }
            System.out.println();
        }

        return values;
    }


    // 交渉相手の統計情報 の取得
    /**
     * 平均
     * @param sender
     * @return
     */
    public double getRivalMean(Object sender) {
        return rivalsMean.get(sender);
    }

    /**
     * 分散
     * @param sender
     * @return
     */
    public double getRivalVar(Object sender) {
        return rivalsVar.get(sender);
    }

    /**
     * 標準偏差
     * @param sender
     * @return
     */
    public double getRivalSD(Object sender) {
        return rivalsSD.get(sender);
    }


    /**
     * エージェントSenderにおける今sessionにおける提案の自身の最大効用値
     * @param sender
     * @return
     */
    public double getRivalMax(Object sender){
        return rivalsMax.get(sender);
    }

    /**
     * エージェントSenderにおける今sessionにおける提案の自身の最小効用値
     * @param sender
     * @return
     */
    public double getRivalMin(Object sender){
        return rivalsMin.get(sender);
    }

    /**
     * 2018/5/24
     * valuePriorAllを初期化する
     */
    public void valuePriorAllInit(HashMap<Issue,ArrayList<Value>> valuePriorAll) {
    		valuePriorAll = new HashMap<Issue,ArrayList<Value>>();
    }

    /**
     * 2018/5/25
     * 重視するissueとそのissueの重視するvalueのlistをセットで返す
     */
	public void valueIssueSetGet() {
    		ArrayList<HashMap<Issue,Value>> valueIssueSets = new ArrayList<HashMap<Issue,Value>>();
    		valuePriorAll = new HashMap<Issue,ArrayList<Value>>();

    		//全相手の重視issueとそのissueの重視のセットを１つのlistにまとめる
    		for(Object rival:rivals) {
    			if(!valuePrior.get(rival).isEmpty()) {//重視issueがある場合
    			if(!valuePrior.get(rival).get(0).isEmpty()) {
    			System.out.println("valuePrior.get(rival).get(0):" + valuePrior.get(rival).get(0));
    			valueIssueSets.add(valuePrior.get(rival).get(0));
    			}

    			System.out.println("issues.size:" + issues.size());
    			if(valuePrior.get(rival).size() == 2) {//issueのsizeが２以上の場合
    			if(!valuePrior.get(rival).get(1).isEmpty()) {
    			valueIssueSets.add(valuePrior.get(rival).get(1));
    			}
    			}
    			}
    		}
    		System.out.println("valueIssueSets:" + valueIssueSets);
    		//重視issueとvalueのセットを合併
    		for(HashMap<Issue,Value> valueIssueSet: valueIssueSets) {
    			//System.out.println("valueIssueSet.keySet():" + valueIssueSet.keySet());
    			Iterator<Issue> key_itr = valueIssueSet.keySet().iterator();
    			Issue issue_tmp = (Issue) key_itr.next();

    			//System.out.println("issue_tmp:" + issue_tmp);
    			//System.out.println("valuePriorAll:" + valuePriorAll);

    			//issueはすでに追加された場合

    			if(valuePriorAll.containsKey(issue_tmp)) {
    				//valueはissueのvaluelistに存在するかどうか判断
    				boolean valueExist = false;
    				for(Value value: valuePriorAll.get(issue_tmp)) {
    					if(valueIssueSet.get(issue_tmp) == value) {
    						valueExist = true;
    					}
    				}
    				//valueはissueのvaluelistに存在しない場合、追加
    				if(!valueExist) {
    					ArrayList<Value> valueList =  valuePriorAll.get(issue_tmp);
    					valueList.add((Value) valueIssueSet.get(issue_tmp));
    					valuePriorAll.put(issue_tmp, valueList);
    				}
    			}

    			//issueはまだ追加されていない場合
    			if(!valuePriorAll.containsKey(issue_tmp)) {
    				ArrayList<Value> valueList = new ArrayList<Value>();
    				valueList.add((Value) valueIssueSet.get(issue_tmp));
    				//issueとそのvaluelistを追加
    				valuePriorAll.put(issue_tmp, valueList);
    			}
    		}
    		System.out.println("valuePriorAll:" + valuePriorAll);
	}
}
