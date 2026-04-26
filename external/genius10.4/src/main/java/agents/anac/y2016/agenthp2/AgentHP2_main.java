package agents.anac.y2016.agenthp2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Inform;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.utility.AbstractUtilitySpace;

/**
 * AgentHP2 main class
 * 
 * @author Hiroyuki Shinohara
 * @date 2016/03/30
 */
public class AgentHP2_main extends AbstractNegotiationParty {

	private int partyNum = 0; // 交渉参加者合計人数
	private int issueNum = 0; // 論点数
	private int targetID = 0; // 判断対象の提案者ID
	private Bid targetBid = null; // 判断対象の提案
	private double reservationValue = 0.0; // 予約値
	private double discountFactor = 0.0; // 割引係数
	private AbstractUtilitySpace myUtilitySpace = null; // 自身の効用関数保存用
	private double underlimitUtility = 1.0; // 効用値の下限値(時間で変化)
	private HashMap<String, Integer> participantList; // 参加者名と配列要素数の対応格納Hash
	private double sumUtility[]; // 相手の提案の自分にとっての効用値合計
	private int othersOfferCount[]; // 参加者それぞれの累計提案数
	private module_AHP AHPEvaluator = null; // AHPクラス用インスタンス
	private boolean isExpectedUtilitySpace = false; // 効用空間予測の初回実行済み判断フラグ
	private int nextExpectTiming = 0; // 次の効用空間予測タイミング
	private double expectUtilitySpaceTimingArray[]; // 効用空間の予測タイミング格納用配列
	private double sumAHPEvalationForOpponent[]; // 相手の提案の相手に取ってのAHP評価値の合計
	private double sumAHPEvalationForOpponent2[]; // 相手の提案の相手に取ってのAHP評価値の二乗の合計
	private double estimateAHPEvaluationMax[]; // 交渉相手の予測最大AHP評価値
	private double maximumConcessionDegree = 0.0; // 最大でどこまで譲歩するか
	private module_BidGenerate bidGenerater = null; // Bid作成クラス用インスタンス
	private ArrayList<Bid> offeredBidList[] = null; // 交渉相手からの提案一時保存用

	/**
	 * 初期化(geniusから呼び出し)
	 *
	 * @param utilitySpace
	 *            自分の効用空間
	 * @param deadlines
	 *            交渉の制限時間
	 * @param timeline
	 *            経過時間(0～1)
	 * @param randomSeed
	 *            乱数用シード
	 * @param agentID
	 *            自分の表示名
	 */
	@Override
	public void init(NegotiationInfo info) {

		// 親クラス初期化メソッド
		super.init(info);

		// 最低効用値取得
		reservationValue = utilitySpace.getReservationValueUndiscounted();

		// 割引係数取得
		discountFactor = utilitySpace.getDiscountFactor();

		// 自身の効用関数取得
		myUtilitySpace = utilitySpace;

		// 初期化
		partyNum = 0;
		issueNum = 0;
		targetID = 0;
		targetBid = null;
		underlimitUtility = 0.0;
		participantList = null;
		sumUtility = null;
		othersOfferCount = null;
		AHPEvaluator = null;
		isExpectedUtilitySpace = false;
		expectUtilitySpaceTimingArray = null;
		nextExpectTiming = 0;
		sumAHPEvalationForOpponent = null;
		sumAHPEvalationForOpponent2 = null;
		estimateAHPEvaluationMax = null;
		offeredBidList = null;

		// 論点数取得
		issueNum = utilitySpace.getDomain().getIssues().size();

		// 参加者名と配列要素数対応Hash作成
		participantList = new HashMap<String, Integer>();

		// 提案作成最低効用値初期値
		underlimitUtility = 1.0;

		// 効用空間予測タイミング作成
		int expectTimingNum = 50;
		expectUtilitySpaceTimingArray = new double[expectTimingNum];
		for (int i = 0; i < expectTimingNum; i++) {
			expectUtilitySpaceTimingArray[i] = (i + 1.0) / (double) expectTimingNum;
		}

		// Bid作成クラス読み込み
		bidGenerater = new module_BidGenerate(utilitySpace);

		// 最大でどこまで譲歩するか決定
		maximumConcessionDegree = 0.35;
		if (1.0 - maximumConcessionDegree < reservationValue) {
			maximumConcessionDegree = 1.0 - reservationValue;
		}
	}

	/**
	 * 自分の番に呼ばれ，AcceptかOfferか選択
	 *
	 * @param validActions
	 *            選べる行動のクラスが格納されている(Accept or Offer)
	 * @return 選んだ行動のクラス(Accept or Offer)
	 */
	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {

		if (timeline.getTime() > expectUtilitySpaceTimingArray[nextExpectTiming]) {

			// 自分以外の参加者人数
			int otherCount = partyNum - 1;

			// 効用空間予測更新
			for (int i = 0; i < otherCount; i++) {
				AHPEvaluator.updateExpectUtilitySpace(i);
			}

			// 次の更新タイミングへ
			nextExpectTiming += 1;

			// 効用関数の初回予測時のみ実行
			if (!isExpectedUtilitySpace) {

				// 効用関数予測済みフラグON
				isExpectedUtilitySpace = true;

				// 効用関数予測前の提案のAHP評価値をまとめて合計に加算
				for (int i = 0; i < otherCount; i++) {
					int listSize = offeredBidList[i].size();
					for (int j = 0; j < listSize; j++) {
						updateAHPEvaluationSum(i, offeredBidList[i].get(j));
					}
				}
			}
		}

		// 効用関数予測済みなら歩み寄り度合い更新
		if (isExpectedUtilitySpace) {
			makeConcessionParameterByAHP();
		}

		// 承諾が可能な時
		if (validActions.contains(Accept.class)) {

			// 承諾判断
			double probability = getAcceptableProbability();
			if (probability >= Math.random() || probability >= 0.90) {
				return new Accept(getPartyId(), ((ActionWithBid) getLastReceivedAction()).getBid());
			}
		}

		// 提案作成
		Bid nextBid = null;
		if (isExpectedUtilitySpace && underlimitUtility < 1.0) {

			// Bid作成
			nextBid = generateOfferBid();

			// 予測作成前&譲歩前は最大効用Bid
		} else {
			nextBid = bidGenerater.getMaximumEvaluationBid();
		}

		// 判断対象に自分のOfferBidを設定
		targetBid = nextBid;

		// 提案
		return new Offer(getPartyId(), nextBid);
	}

	/**
	 * 他の参加者のすべての行動(Accept or Offer)がメッセージとして受信
	 *
	 * @param sender
	 *            行動者の情報
	 * @param action
	 *            行動内容(Accept,Offer,Inform)
	 */
	@Override
	public void receiveMessage(AgentID sender, Action action) {

		// オーバーライドのため
		super.receiveMessage(sender, action);

		// 初回のみ呼び出し
		if (action.getClass() == Inform.class) {

			// 交渉参加合計人数取得
			partyNum = (int) ((Inform) action).getValue();

			// 配列&Hash初期化
			initHashArray(partyNum);

			// AHPクラス呼び出し
			AHPEvaluator = new module_AHP(utilitySpace, partyNum);

		} else {

			// 参加者に対応する配列要素番号取得
			int partyID = getSenderID(sender.toString());
			if (partyID < 0) {
				partyID = participantList.size();
				participantList.put(sender.toString(), partyID);
			}

			// senderが誰かの提案を拒否してofferした時
			if (action.getClass() == Offer.class) {

				// 相手の提案を取得し判断対象に設定 & 提案者ID保存
				Offer received_offer = (Offer) action;
				targetBid = received_offer.getBid();
				targetID = partyID;

				// senderが誰かの提案をAcceptした時
			} else if (action.getClass() == Accept.class) {

				// 提案者ID保存
				// senderがacceptした時はtargetBidを提案したとする
				targetID = partyID;
			}

			// 提案内容の数え上げ
			AHPEvaluator.countBid(partyID, targetBid);

			// 各交渉相手の提案回数をカウント
			othersOfferCount[partyID]++;

			// 効用値合計更新
			double util = getMyUtility(targetBid);
			sumUtility[partyID] += util;

			// 交渉相手の効用関数予測済みかで処理分岐
			if (isExpectedUtilitySpace) {

				// 相手の提案の推定譲歩度合い(相手から見たAHP評価値)合計更新
				updateAHPEvaluationSum(partyID, targetBid);

			} else {

				// 交渉相手の効用関数が未予測なので提案をリストに保存
				offeredBidList[partyID].add(targetBid);
			}
		}
	}

	/**
	 * 同意確率を計算
	 * 
	 * @return double 同意確率
	 */
	private double getAcceptableProbability() {

		double probability = 0.0;
		double time = timeline.getTime();
		double offeredUtility = getMyUtility(targetBid);
		double offeredAHPEvaluation = 0.0;

		// 以下の条件では必ず承諾
		// 1. 効用が一定以上
		// 2. 交渉の残り時間がギリギリ
		// 3. 提案の効用が現在の提案作成基準値以上
		if (offeredUtility >= 0.90 || time >= 0.99 || offeredUtility >= underlimitUtility) {
			return 1.0;
		}

		// 効用空間予測が完了している時
		if (isExpectedUtilitySpace) {

			// 提案の提案者から見たAHP評価値を取得
			offeredAHPEvaluation = AHPEvaluator.getAHPEvaluationForOpponents(targetID, targetBid);

			// 譲歩率を元に承諾確率計算
			double tmp1 = (Math.pow(time, 5.0) / (5.0 * discountFactor));
			tmp1 += (offeredAHPEvaluation - estimateAHPEvaluationMax[targetID]) + (offeredUtility - underlimitUtility);

			// 承諾確率を決定
			probability = tmp1;
		}

		return probability;
	}

	/**
	 * 提案作成
	 * 
	 * @return Bid 提案
	 */
	private Bid generateOfferBid() {

		Bid nextBid = null;
		int createBidNum = 10;
		Bid tmpBidArray[] = new Bid[createBidNum];

		// 作成Bidの効用範囲作成
		double upperLimitUtility = underlimitUtility + 0.05;
		double lowerLimitUtility = underlimitUtility - 0.05;
		if (upperLimitUtility > 1.0) {
			upperLimitUtility = 1.0;
		}
		if (lowerLimitUtility < reservationValue) {
			lowerLimitUtility = reservationValue;
		}

		// 効用値が範囲内のBidを複数作成
		for (int i = 0; i < createBidNum; i++) {
			Bid tmp = null;
			double util = 0.0;
			int loop = 100;
			do {
				tmp = generateRandomBid();
				util = getUtility(tmp);
				if (--loop <= 0) {
					loop = 100;
					lowerLimitUtility -= 0.01;
					upperLimitUtility += 0.01;
				}

			} while (tmp != null && (util < lowerLimitUtility || util > upperLimitUtility));
			tmpBidArray[i] = tmp;
		}

		// 作成BidをAHPにて評価し一番評価値の高いものを採用
		double evalValue = 0.0;
		for (int i = 0; i < createBidNum; i++) {
			double tmp = AHPEvaluator.getAHPEvaluation(tmpBidArray[i]);
			if (tmp > evalValue) {
				evalValue = tmp;
				nextBid = tmpBidArray[i];
			}
		}

		// エラー対策
		if (nextBid == null) {
			nextBid = tmpBidArray[0];
		}

		return nextBid;
	}

	/**
	 * Hashと配列初期化用
	 * 
	 * @param int
	 *            交渉参加人数
	 */
	private void initHashArray(int partyNum) {

		// 自分以外の参加人数
		int otherCount = partyNum - 1;

		// 相手提案の効用値合計格納配列作成
		sumUtility = new double[otherCount];
		for (int i = 0; i < otherCount; i++) {
			sumUtility[i] = 0.0;
		}

		// 参加者の累計提案数格納配列作成
		othersOfferCount = new int[otherCount];
		for (int i = 0; i < otherCount; i++) {
			othersOfferCount[i] = 0;
		}

		// 相手にとってのAHP評価値合計格納用配列作成
		sumAHPEvalationForOpponent = new double[otherCount];
		sumAHPEvalationForOpponent2 = new double[otherCount];
		for (int i = 0; i < otherCount; i++) {
			sumAHPEvalationForOpponent[i] = 0.0;
			sumAHPEvalationForOpponent2[i] = 0.0;
		}

		// 譲歩関数用
		estimateAHPEvaluationMax = new double[otherCount];
		for (int i = 0; i < otherCount; i++) {
			estimateAHPEvaluationMax[i] = 0.0;
		}

		// 提案保存用List作成
		offeredBidList = new ArrayList[otherCount];
		for (int i = 0; i < otherCount; i++) {
			offeredBidList[i] = new ArrayList<Bid>();
		}
	}

	/**
	 * 参加者名から対応する配列要素数を取得
	 * 
	 * @param String
	 *            参加者名
	 * @return int 配列要素数
	 */
	private int getSenderID(String name) {

		int id = -1;
		if (participantList.containsKey(name)) {
			id = participantList.get(name);
		}

		return id;
	}

	/**
	 * AHPによる評価値を利用した譲歩関数
	 */
	private void makeConcessionParameterByAHP() {

		int otherCount = partyNum - 1;
		double time = timeline.getTime();
		double minEstimateMax = 1.0;

		// 承諾判定用パラメータ作成
		for (int i = 0; i < otherCount; i++) {

			double average = sumAHPEvalationForOpponent[i] / othersOfferCount[i];
			double average2 = sumAHPEvalationForOpponent2[i] / othersOfferCount[i];
			double variance = average2 - average * average;

			// 交渉相手から引き出せるAHP評価値の予測最大
			estimateAHPEvaluationMax[i] = average + ((1.0 - average) * Math.sqrt(12.0) * variance);

			// 予測最大値の小さい方(将来的に譲歩が少ない方)を保存(0除算警戒)
			if (minEstimateMax > estimateAHPEvaluationMax[i] && estimateAHPEvaluationMax[i] > 0.0) {
				minEstimateMax = estimateAHPEvaluationMax[i];
			}
		}

		// 提案作成用基準効用値作成
		double tmpUtility = 1.0 - maximumConcessionDegree
				* Math.pow(time, (Math.pow(discountFactor, 3.0) / Math.pow(minEstimateMax, 1.0)));
		if (underlimitUtility > tmpUtility) {
			underlimitUtility = tmpUtility;
		}
	}

	/**
	 * 指定されたBidの自身の効用値を取得
	 * 
	 * @param Bid
	 *            効用値を取得したいBid
	 * @return double 指定されたBidの効用値
	 */
	private double getMyUtility(Bid getTargetBid) {

		double utility = 0.0;
		try {
			utility = myUtilitySpace.getUtility(getTargetBid);
		} catch (Exception e) {
			System.out.println("Get My Utility Target Bid Is Wrong");
			System.out.println("Wrong Bid Content: " + targetBid.toString());
		}

		return utility;
	}

	/**
	 * 提案の提案者からみたAHP評価値(=提案の推定譲歩度合い)の合計を更新
	 * 
	 * @param int
	 *            参加者ID
	 * @param Bid
	 *            更新対象のBid
	 */
	private void updateAHPEvaluationSum(int partyID, Bid updateTarget) {

		double ahpEval = AHPEvaluator.getAHPEvaluationForOpponents(partyID, updateTarget);
		sumAHPEvalationForOpponent[partyID] += ahpEval;
		sumAHPEvalationForOpponent2[partyID] += ahpEval * ahpEval;
	}

	@Override
	public String getDescription() {
		return "ANAC2016";
	}
}
