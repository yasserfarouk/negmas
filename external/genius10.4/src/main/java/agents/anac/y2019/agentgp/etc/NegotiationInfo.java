package agents.anac.y2019.agentgp.etc;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import genius.core.list.Tuple;
//import list.Tuple;

import java.util.Map.Entry;
import java.util.Random;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.utility.AbstractUtilitySpace;

public class NegotiationInfo {
    private AbstractUtilitySpace utilitySpace; // 効用空間
    private ArrayList<Issue> issues; // 論点
    private ArrayList<Object> opponents; // 自身以外の交渉参加者のList
	private ArrayList<Bid> MyBidHistory = null; // 提案履歴
	private ArrayList<Bid> BOBHistory = null; // BestOfferedBidの更新履歴
	private ArrayList<Bid> PBList = null; // BestPopularBidのリスト
	private HashMap<Object, Integer> opponentsAcceptNum;  // acceptした回数
	private HashMap<Object, ArrayList<Bid>> opponentsBidHistory = null; // 提案履歴
	private HashMap<Object, Double> opponentsAverage;  // 平均
	private HashMap<Object, Double> opponentsVariance; // 分散
	private HashMap<Object, Double> opponentsSum;    // 和
	private HashMap<Object, Double> opponentsPowSum; // 二乗和
	private HashMap<Object, Double> opponentsStandardDeviation; // 標準偏差
    private HashMap<Issue, HashMap<Value, Double>> valueRelativeUtility = null; // 自身の効用空間における各論点値の相対効用値行列（線形効用空間用）
    private HashMap<Issue, HashMap<Value, Integer>> allValueFrequency = null; // 全員分の頻度行列
    private HashMap<Object, HashMap<Issue, HashMap<Value, Integer>>> opponentsValueFrequency = null; // 交渉者別の頻度行列
    private HashMap<Object, HashMap<Issue, ArrayList<Tuple<Value,Integer>>>> opponentsValueOrdinalScale = null; // 交渉者別の各issueごとの順序尺度
    private double BOU = 0.0; // BestOfferedUtility 2者間交渉以外では使わない
    private double MPBU = 0.0; // MaxPopularBidUtility
    private double time_scale = 0.0; // 自分の手番が回ってくる時間間隔
    private int round = 0; // 自分の手番数
    private int negotiatorNum = 3; // 交渉者数（不具合が生じた場合に備えて3人で初期化）
	private boolean isLinerUtilitySpace = true; // 線形効用空間であるかどうか

	private boolean isPrinting = false;

    public NegotiationInfo(AbstractUtilitySpace utilitySpace, boolean isPrinting) {
    	// 初期化
    	this.isPrinting = isPrinting;
    	this.utilitySpace = utilitySpace;
    	issues = (ArrayList<Issue>) utilitySpace.getDomain().getIssues();
    	opponents = new ArrayList<Object>();
    	MyBidHistory = new ArrayList<>();
    	BOBHistory = new ArrayList<>();
    	PBList = new ArrayList<>();
    	opponentsAcceptNum = new HashMap<Object, Integer>();
    	opponentsBidHistory = new HashMap<Object, ArrayList<Bid>>();
    	opponentsAverage = new HashMap<Object, Double>();
    	opponentsVariance = new HashMap<Object, Double>();
    	opponentsSum = new HashMap<Object, Double>();
    	opponentsPowSum = new HashMap<Object, Double>();
    	opponentsStandardDeviation = new HashMap<Object, Double>();
		valueRelativeUtility = new HashMap<Issue, HashMap<Value, Double>> ();
    	allValueFrequency = new HashMap<Issue, HashMap<Value, Integer>> ();
    	opponentsValueFrequency = new HashMap<Object, HashMap<Issue, HashMap<Value, Integer>>> ();
		opponentsValueOrdinalScale = new HashMap<Object, HashMap<Issue, ArrayList<Tuple<Value,Integer>>>>();

    	try { initAllValueFrequency(); }
    	catch (Exception e1) {
			System.out.println("全員分の頻度行列の初期化に失敗しました");
			e1.printStackTrace();
		}
		try { initValueRelativeUtility(); }
		catch (Exception e) {
			System.out.println("相対効用行列の初期化に失敗しました");
			e.printStackTrace();
		}

		if(this.isPrinting){ System.out.println("NegotiationInfo:success"); }
    }

    public void initOpponent(Object sender) {
		initNegotiatingInfo(sender); // 交渉情報を初期化
		try {
			initOpponentsValueFrequency(sender);
			initOpponentValueOrdinalScale(sender);
			opponentsAcceptNum.put(sender, 0);
		} // senderの頻度行列を初期化
		catch (Exception e) {
			System.out.println("交渉参加者の頻度行列の初期化に失敗しました");
			e.printStackTrace();
		}
		opponents.add(sender); // 交渉参加者にsenderを追加
	}


    public void updateInfo(Object sender, Bid offeredBid) {
    	try { updateNegotiatingInfo(sender, offeredBid); } // 交渉情報の更新
    	catch (Exception e1) {
			System.out.println("交渉情報の更新に失敗しました");
			e1.printStackTrace();
		}
    	try {
    		updateFrequencyList(sender, offeredBid);
			updateOrdinalScale(sender, offeredBid);
    	} // senderの頻度行列の更新
    	catch (Exception e) {
			System.out.println("頻度行列の更新に失敗しました");
			e.printStackTrace();
		}
    }

    private void initNegotiatingInfo(Object sender) {
    	opponentsBidHistory.put(sender, new ArrayList<Bid>());
    	opponentsAverage.put(sender, 0.0);
    	opponentsVariance.put(sender, 0.0);
    	opponentsSum.put(sender, 0.0);
    	opponentsPowSum.put(sender, 0.0);
    	opponentsStandardDeviation.put(sender, 0.0);
    }

    private void initOpponentsValueFrequency(Object sender) throws Exception{
        opponentsValueFrequency.put(sender, new HashMap<Issue, HashMap<Value, Integer>>()); // senderの頻度行列の初期化
		for(Issue issue:issues){
			opponentsValueFrequency.get(sender).put(issue, new HashMap<Value, Integer>()); // 頻度行列における論点行の初期化
			ArrayList<Value> values = getValues(issue);
			for(Value value:values){ opponentsValueFrequency.get(sender).get(issue).put(value, 0); } // 論点行の要素出現数の初期化
    	}
    }

    public void initOpponentValueOrdinalScale(Object sender) throws Exception{
        opponentsValueOrdinalScale.put(sender, new HashMap<Issue, ArrayList<Tuple<Value, Integer>>>()); // senderの頻度行列の初期化
		for(Issue issue:issues){
			opponentsValueOrdinalScale.get(sender).put(issue, new ArrayList<Tuple<Value, Integer>>()); // 頻度行列における論点行の初期化
    	}
    }

	// 全員分の頻度行列の初期化
    private void initAllValueFrequency() throws Exception{
		ArrayList<Value> values = null;
		for(Issue issue:issues){
			allValueFrequency.put(issue, new HashMap<Value, Integer>()); // 論点行の初期化
			values = getValues(issue);
			for(Value value:values){ allValueFrequency.get(issue).put(value, 0); } // 論点行の要素の初期化
    	}
	}

	// 相対効用行列の初期化
    private void initValueRelativeUtility() throws Exception{
		ArrayList<Value> values = null;
		for(Issue issue:issues){
			valueRelativeUtility.put(issue, new HashMap<Value, Double>()); // 論点行の初期化
			values = getValues(issue);
			for(Value value:values){ valueRelativeUtility.get(issue).put(value, 0.0); } // 論点行の要素の初期化
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
				valueRelativeUtility.get(issue).put(value, utilitySpace.getUtility(currentBid) - utilitySpace.getUtility(maxBid));
			}
    	}
	}

    public void updateNegotiatingInfo(Object sender, Bid offeredBid) throws Exception {
    	opponentsBidHistory.get(sender).add(offeredBid); // 提案履歴

    	double util = utilitySpace.getUtility(offeredBid);
    	opponentsSum.put(sender, opponentsSum.get(sender) + util); // 和
    	opponentsPowSum.put(sender, opponentsPowSum.get(sender) + Math.pow(util, 2)); // 二乗和

    	int round_num = opponentsBidHistory.get(sender).size();
    	opponentsAverage.put(sender, opponentsSum.get(sender) / round_num); // 平均
    	opponentsVariance.put(sender, (opponentsPowSum.get(sender) / round_num) - Math.pow(opponentsAverage.get(sender), 2)); // 分散

    	if(opponentsVariance.get(sender) < 0){opponentsVariance.put(sender, 0.0);}
    	opponentsStandardDeviation.put(sender, Math.sqrt(opponentsVariance.get(sender))); // 標準偏差

    	if(util > BOU){
    		BOBHistory.add(offeredBid); // BOUの更新履歴に追加
    		BOU = util;  // BOUの更新
    	}
    }
    public void updateAcceptNum(Object sender){
    	opponentsAcceptNum.put(sender, opponentsAcceptNum.get(sender)+1);
    }

    public HashMap<Issue, ArrayList<Tuple<Value,Integer>>> getValueOrdinalScale(){
    	//AcceptNumが多いエージェントを選びやすくしてる。
    	int allAcceptNum = 0;
    	for (Object agent : opponentsAcceptNum.keySet()) {
    		allAcceptNum += opponentsAcceptNum.get(agent);
    	}
		Object selectedAgent = null;
    	if(allAcceptNum != 0){
    		double rand = Math.random();
			int acceptNum = 0;
			for (Object agent : opponentsAcceptNum.keySet()) {
				acceptNum += opponentsAcceptNum.get(agent);
				if(rand <= acceptNum/allAcceptNum){
					selectedAgent = agent;
					break;
				}
			}
    	}
    	else{
    		int rand = (new Random()).nextInt(opponents.size());
    		int acceptNum = 0;
			for (Object agent : opponentsAcceptNum.keySet()) {
				acceptNum++;
				if(rand <= acceptNum){
					selectedAgent = agent;
					break;
				}
			}
    	}
    	//nullを返すことはないはず。
    	return opponentsValueOrdinalScale.get(selectedAgent);
    }

    // 頻度行列を更新
    private void updateFrequencyList(Object sender, Bid offeredBid) throws Exception{
		for(Issue issue:issues){
			Value value = offeredBid.getValue(issue.getNumber());
			opponentsValueFrequency.get(sender).get(issue).put(value, opponentsValueFrequency.get(sender).get(issue).get(value) + 1);  // リストを更新
			allValueFrequency.get(issue).put(value, allValueFrequency.get(issue).get(value) + 1);  // リストを更新
    	}
    }
    // 順序尺度行列を更新
    private void updateOrdinalScale(Object sender, Bid offeredBid) throws Exception{
		for(Issue issue:issues){
			Value value = offeredBid.getValue(issue.getNumber());
			int i = 0;
			for(Tuple<Value, Integer> tuple: opponentsValueOrdinalScale.get(sender).get(issue)){
				if(tuple.get1().equals(value)){
					opponentsValueOrdinalScale.get(sender).get(issue).set(i, new Tuple<Value, Integer>(value, tuple.get2()+1));
					break;
				}
				i++;
			}
			if(opponentsValueOrdinalScale.get(sender).get(issue).size() == i)
				opponentsValueOrdinalScale.get(sender).get(issue).add(new Tuple<Value, Integer>(value, 1));
    	}
    }
    // 順序尺度行列をソート
    public void sortOrdinalScale() throws Exception{
    	for(Object opponent: opponents){
			for(Issue issue:issues){
				Tuple<Value, Integer> tuplePre = opponentsValueOrdinalScale.get(opponent).get(issue).get(opponentsValueOrdinalScale.get(opponent).get(issue).size()-1);
				for (int i = opponentsValueOrdinalScale.get(opponent).get(issue).size() - 2; i >= 0; i--) {
					if(opponentsValueOrdinalScale.get(opponent).get(issue).get(i).get2() < tuplePre.get2()){
						opponentsValueOrdinalScale.get(opponent).get(issue).set(i+1,opponentsValueOrdinalScale.get(opponent).get(issue).get(i));
						opponentsValueOrdinalScale.get(opponent).get(issue).set(i, tuplePre);
					}
					tuplePre = opponentsValueOrdinalScale.get(opponent).get(issue).get(i);
				}
			}
    	}
    }

    // 交渉者数を返す
    public void updateOpponentsNum(int num) { negotiatorNum = num; }

    // 線形効用空間でない場合
    public void utilitySpaceTypeisNonLiner() { isLinerUtilitySpace = false; }

    // 自身の提案情報の更新
    public void updateMyBidHistory(Bid offerBid) { MyBidHistory.add(offerBid); }

    // 提案時間間隔を返す
    public void updateTimeScale(double time) {
    	round = round + 1;
    	time_scale = time / round;
    }

    // PopularBidListを返す
    public void updatePBList(Bid popularBid) throws Exception {
    	if(!PBList.contains(popularBid)){ // 一意に記録
	    	PBList.add(popularBid);
	    	MPBU = Math.max(MPBU, utilitySpace.getUtility(popularBid));
	    	Collections.sort(PBList, new UtilityComparator()); // ソート

	    	if(isPrinting){
	    		System.out.println("ranking");
	    		for(int i=0; i<PBList.size();i++){ System.out.println(utilitySpace.getUtility(PBList.get(i))); }
	    	    System.out.println("Size:" + PBList.size() + ", Min:" + utilitySpace.getUtility(PBList.get(0)) + ", Max:" + utilitySpace.getUtility(PBList.get(PBList.size()-1)) + ", Opponents:" + opponents);
	    	}
    	}
    }

    public class UtilityComparator implements Comparator<Bid> {
        public int compare(Bid a, Bid b) {
			try {
				double u1 = utilitySpace.getUtility(a);
				double u2 = utilitySpace.getUtility(b);
	            if (u1 < u2) { return 1; }
	            if (u1 == u2) { return 0; }
	            if (u1 > u2) { return -1; }
			} catch (Exception e) {
    			System.out.println("効用値に基づくソートに失敗しました");
				e.printStackTrace();
			}
			return 0; // 例外処理
        }
    }

    // 平均
    public double getAverage(Object sender) { return opponentsAverage.get(sender); }

    // 分散
    public double getVariancer(Object sender) { return opponentsVariance.get(sender); }

    // 標準偏差
    public double getStandardDeviation(Object sender) { return opponentsStandardDeviation.get(sender); }

    // 相手の提案履歴の要素数を返す
    public int getPartnerBidNum(Object sender) { return opponentsBidHistory.get(sender).size(); }

    // 相手の提案履歴の頻度を返す
    public HashMap<Issue, HashMap<Value, Integer>> getHistory(Object sender) { return opponentsValueFrequency.get(sender); }

    // 自身のラウンド数を返す
    public int getRound() { return round; }

    // 交渉者数を返す
    public int getNegotiatorNum () { return negotiatorNum; }

    // 相対効用行列を返す
    public HashMap<Issue, HashMap<Value, Double>> getValueRelativeUtility(){return valueRelativeUtility; }

    // 線形効用空間であるかどうかを返す
    public boolean isLinerUtilitySpace () { return isLinerUtilitySpace; }

    // 論点一覧を返す
    public ArrayList<Issue> getIssues() { return issues; }

    // BestOfferedUtilityを返す
    public double getBOU() { return BOU; }

    // MaxPopularBidUtilityを返す
    public double getMPBU() { return MPBU; }

    // 交渉者全体のBOBHistoryを返す
    public ArrayList<Bid> getBOBHistory() { return BOBHistory; }

    // PBListを返す
    public ArrayList<Bid> getPBList() { return PBList; }

    // 提案時間間隔を返す
    public double getTimeScale() { return time_scale; }

    // 論点における取り得る値の一覧を返す
    public ArrayList<Value> getValues(Issue issue) {
    	ArrayList<Value> values = new ArrayList<Value>();
    	switch(issue.getType()) {
			case DISCRETE:
				List<ValueDiscrete> valuesDis = ((IssueDiscrete)issue).getValues();
				for(Value value:valuesDis){ values.add(value); }
				break;
			case INTEGER:
				int min_value = ((IssueInteger)issue).getLowerBound();
				//int min_value = ((IssueInteger)issue).getUpperBound();
				int max_value = ((IssueInteger)issue).getUpperBound();
				for(int j=min_value; j<=max_value; j++){
					ValueInteger valueObject = new ValueInteger(j);
					values.add((Value)valueObject);
				}
				break;
			default: try {
				throw new Exception("issue type "+ issue.getType()+" not supported by Atlas3");
			} catch (Exception e) {
    			System.out.println("論点の取り得る値の取得に失敗しました");
				e.printStackTrace();
			}
    	}
    	return values;
    }

    // 交渉相手の一覧を返す
    public ArrayList<Object> getOpponents() { return opponents; }

    // FrequencyListの頻度に基づき，要素を返す
    public Value getValuebyFrequencyList(Object sender, Issue issue) {
    	int current_f = 0;
    	int max_f = 0; // 最頻出要素の出現数
    	Value max_value = null; // 最頻出要素
    	ArrayList<Value> randomOrderValues = getValues(issue);
		Collections.shuffle(randomOrderValues); // ランダムに並び替える(毎回同じ順番で評価した場合，出現数が同値の場合に返却値が偏るため)

		for(Value value:randomOrderValues){
			current_f = opponentsValueFrequency.get(sender).get(issue).get(value);
			// 最頻出要素を記録
			if(max_value == null || current_f > max_f){
				max_f = current_f;
				max_value = value;
			}
		}
		return max_value;
    }

    // 全員分のFrequencyListの頻度に基づき，要素を返す
    public Value getValuebyAllFrequencyList(Issue issue) {
    	int current_f = 0;
    	int max_f = 0; // 最頻出要素の出現数
    	Value max_value = null; // 最頻出要素
    	ArrayList<Value> randomOrderValues = getValues(issue);
		Collections.shuffle(randomOrderValues); // ランダムに並び替える(毎回同じ順番で評価した場合，出現数が同値の場合に返却値が偏るため)

		for(Value value:randomOrderValues){
			current_f = allValueFrequency.get(issue).get(value);
			// 最頻出要素を記録
			if(max_value == null || current_f > max_f){
				max_f = current_f;
				max_value = value;
			}
		}
		return max_value;
    }

    public void saveFrequency(){
		try{
			File file = new File("opponentBid.txt");
			FileWriter filewriter = new FileWriter(file);

			for(Object opponent: opponents){
				for(Issue issue: issues){
					filewriter.write(opponent.toString()+"\n");
					HashMap<Value, Integer> values = opponentsValueFrequency.get(opponent).get(issue);
			    	  // List 生成 (ソート用)
			    	List<Map.Entry<Value,Integer>> entries = new ArrayList<Map.Entry<Value ,Integer>>(values.entrySet());
			    	Collections.sort(entries, new Comparator<Map.Entry<Value,Integer>>() {
			    	      @Override
			    	      public int compare(
			    	            Entry<Value,Integer> entry1, Entry<Value,Integer> entry2) {
			    	          return ((Integer)entry2.getValue()).compareTo((Integer)entry1.getValue());
			    	      }
			    	});

			    	// 内容を表示
			    	for (Entry<Value,Integer> s : entries) {
					  filewriter.write(s.getKey() + " , " + s.getValue()+"\n");
			    	}
				}
			}
			filewriter.close();
			}catch(IOException e){
				System.out.println(e);
		}
    }
    //public void saveSort(){
	//	try{
	//		File file = new File("sortBid.txt");
	//		FileWriter filewriter = new FileWriter(file);
	//		filewriter.write("save sort"+"\n");
	//
	//		//for(Object opponent: opponents){
	//			//filewriter.write(opponent.toString()+"\n");
	//			//for(Issue issue: issues){
	//				//filewriter.write(issue.toString()+"\n");
	//				//ArrayList<Tuple<Value, Integer>> values = opponentsValueOrdinalScale.get(opponent).get(issue);
	//		    	// 内容を表示
	//		    	//for (Tuple<Value,Integer> s : values) {
	//				  //System.out.print("("+s.get1()+","+s.get2()+"):");
	//		    	//}
	//			//}
	//			//System.out.println("");
	//		//}
	//		for (Bid s : MyBidHistory) {
	//		  filewriter.write(s + " , " + (1-utilitySpace.getUtility(s))+"\n");
	//	   	}
	//		filewriter.close();
	//		}catch(IOException e){
	//			System.out.println(e);
	//	}
    //}
}
