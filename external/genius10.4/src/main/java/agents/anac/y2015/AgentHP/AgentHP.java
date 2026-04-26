package agents.anac.y2015.AgentHP;

import java.util.HashMap;
import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Inform;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.ValueDiscrete;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;

/**
 * This is your negotiation party.
 */
public class AgentHP extends AbstractNegotiationParty {

	// 提案保存数最大
	private int logMax = 500;

	/* ここまではコンストラクタ内で初期化必要なし */

	// 交渉参加者合計人数
	private int partyNum = 0;

	// 論点数
	private int issueNum = 0;

	// 判断対象の提案
	private Bid targetBid = null;

	// 予約値
	private double reservationValue = 0.0;

	// 効用値の下限値(時間で変化)
	private double underlimitUtility = 1.0;

	// 現在の提案保存数
	private int logNum[];

	// 提案ログの論点ごとの項目出現数格納Hash用テンプレ
	private HashMap<String, Integer>[] TemplateLogBidCountHash;

	// 提案ログから作成した予測効用空間用テンプレ
	private HashMap<String, Double>[] TemplateLogBidValueHash;

	// 提案ログ保存ハッシュ
	private Bid[][] logBid;

	// 提案ログの論点ごとの項目出現数格納Hash
	private HashMap<String, Integer>[][] logBidCountHash;

	// 提案ログから作成した予測効用空間格納Hash
	private HashMap<String, Double>[][] logBidValueHash;

	// 参加者名と配列要素数の対応格納Hash
	private HashMap<String, Integer> participantList;

	// 一対比較段数
	private int compareSize = 0;

	// 一対比較結果用配列
	private double compareResultArray[];

	// 予測した相手の論点ごとの重み格納用配列
	private double issueWeightArray[][];

	// 予測の更新回数
	private int estimateCount[];

	// 効用空間合算時の自分の重み
	private double weightUtilSpace = 0.0;

	// 合算効用空間Hash
	private HashMap<String, Double>[] addupValueHash;

	// 合算論点間重み配列
	private double addupWeightArray[];

	// 直近の合算タイミング
	private int prevAddupTiming = 0;

	// 相手の提案の自分にとっての効用値合計&効用値2乗合計
	private double sumUtility[];
	private double sumUtility2[];

	// 承諾確率計算用
	private double estimateMax = 0.0;
	private double utilityBarometer = 0.0;

	/**
	 * コンストラクタ(geniusから呼び出し)
	 *
	 * @param utilitySpace
	 *            自分の効用空間
	 * @param deadlines
	 *            交渉の制限時間
	 * @param timeline
	 *            経過時間(0～1)
	 * @param randomSeed
	 *            乱数用シード
	 */
	@Override
	public void init(NegotiationInfo info) {

		// 親クラスコンストラクタ
		super.init(info);

		// 予約値取得
		reservationValue = utilitySpace.getReservationValueUndiscounted();

		// 初期化
		partyNum = 0;
		issueNum = 0;
		targetBid = null;
		underlimitUtility = 0.0;
		logNum = null;
		TemplateLogBidCountHash = null;
		TemplateLogBidValueHash = null;
		logBid = null;
		logBidCountHash = null;
		logBidValueHash = null;
		participantList = null;
		compareSize = 0;
		compareResultArray = null;
		issueWeightArray = null;
		estimateCount = null;
		weightUtilSpace = 0.0;
		addupValueHash = null;
		addupWeightArray = null;
		prevAddupTiming = 0;
		sumUtility = null;
		sumUtility2 = null;
		estimateMax = 0.0;
		utilityBarometer = 0.0;

		// 論点数取得
		issueNum = utilitySpace.getDomain().getIssues().size();

		// 提案保存Hashテンプレ&合算効用空間保存Hash作成
		TemplateLogBidCountHash = new HashMap[issueNum];
		TemplateLogBidValueHash = new HashMap[issueNum];
		addupValueHash = new HashMap[issueNum];
		for (Issue tmp : utilitySpace.getDomain().getIssues()) {

			int issue_num = tmp.getNumber(); // 論点番号
			IssueDiscrete tmpDiscrete = (IssueDiscrete) tmp;

			TemplateLogBidValueHash[issue_num
					- 1] = new HashMap<String, Double>();
			TemplateLogBidCountHash[issue_num
					- 1] = new HashMap<String, Integer>();
			addupValueHash[issue_num - 1] = new HashMap<String, Double>();
			for (int j = 0; j < tmpDiscrete.getNumberOfValues(); j++) {
				TemplateLogBidValueHash[issue_num - 1]
						.put(tmpDiscrete.getValue(j).toString(), 0.0);
				TemplateLogBidCountHash[issue_num - 1]
						.put(tmpDiscrete.getValue(j).toString(), 0);
				addupValueHash[issue_num - 1]
						.put(tmpDiscrete.getValue(j).toString(), 0.0);
			}
		}

		// 参加者名と配列要素数対応Hash作成
		participantList = new HashMap<String, Integer>();

		// 提案作成最低効用値初期値
		underlimitUtility = 1.0;

		// 効用空間合算の自分の重み初期値
		weightUtilSpace = 1.0;
	}

	/**
	 * 自分の番に呼ばれ，AcceptかOfferか選択
	 *
	 * @param validActions
	 *            選べる行動のクラスが格納されている(Accept or
	 *            Offer)
	 * @return 選んだ行動のクラス(Accept or Offer)
	 */
	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {

		if (isEstimated(partyNum)) {

			// 歩み寄り用係数作成
			makeConcession();

			// 予測効用空間を合算
			addupUtilSpace(weightUtilSpace);
		}

		// 承諾が可能な時
		if (validActions.contains(Accept.class)) {

			// 承諾判断
			if (getAcceptableProbability() > Math.random()) {
				return new Accept(getPartyId(),
						((ActionWithBid) getLastReceivedAction()).getBid());
			}
		}

		// 提案作成
		Bid nextBid = null;
		if (isEstimated(partyNum) && underlimitUtility < 1.0) {

			// Bid作成
			nextBid = generateOfferBid();

			// 予測作成前&譲歩前は最大効用Bid
		} else {
			try {
				nextBid = utilitySpace.getMaxUtilityBid();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		// 判断対象に自分のOfferBidを設定
		targetBid = nextBid;

		// ログ
		// System.out.println("Time = "+timeline.getTime());
		// System.out.println("UnderLimitUtil = "+underlimitUtility);
		// System.out.println("AddupWeight = "+weightUtilSpace);
		// System.out.println("Util(UnDiscount) = "+getUtility(nextBid));
		// System.out.println("Util(Discount) =
		// "+getUtilityWithDiscount(nextBid));

		// 提案
		return new Offer(getPartyId(), nextBid);
	}

	/**
	 * 他の参加者のすべての行動(Accept or
	 * Offer)がメッセージとして受信
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
			partyNum = (Integer) ((Inform) action).getValue();

			// 配列&Hash初期化
			initHashArray(partyNum);

		} else {

			// 参加者に対応する配列要素番号取得
			int partyID = getSenderID(sender.toString());
			if (partyID < 0) {
				partyID = participantList.size();
				participantList.put(sender.toString(), partyID);
				estimateCount[partyID] = 0;
			}

			// senderが誰かの提案を拒否してofferした時
			if (action.getClass() == Offer.class) {

				// 相手の提案を取得し判断対象に設定
				Offer received_offer = (Offer) action;
				targetBid = received_offer.getBid();
			}

			// 提案ログ更新
			// senderがacceptした時はtargetBidを提案したとする
			if (logNum[partyID] < logMax) {

				logBid[partyID][logNum[partyID]] = targetBid;
				logNum[partyID]++;
			}

			// 効用値合計更新
			double util = getUtility(targetBid);
			sumUtility[partyID] += util;
			sumUtility2[partyID] += util * util;

			// 効用空間予測
			if (logNum[partyID] == logMax) {

				// 出現数一時格納Hash作成
				HashMap<String, Integer>[] TmpCountHash = new HashMap[issueNum];
				for (int i = 0; i < issueNum; i++) {
					TmpCountHash[i] = new HashMap<String, Integer>(
							TemplateLogBidCountHash[i]);
				}

				// ログから相手のValueの回数計測
				for (int i = 0; i < logMax; i++) {
					for (int j = 1; j <= issueNum; j++) {
						try {
							if (TmpCountHash[j - 1]
									.containsKey(logBid[partyID][i].getValue(j)
											.toString())) {
								int tmp = TmpCountHash[j - 1]
										.get(logBid[partyID][i].getValue(j)
												.toString());
								TmpCountHash[j - 1].put(logBid[partyID][i]
										.getValue(j).toString(), tmp + 1);
							}
						} catch (Exception e) {
							System.out.println("getValue Error !\n");
						}
						;
					}
				}

				// 偏りのあるデータを一部改変
				for (int i = 0; i < issueNum; i++) {
					int tmp = 0;
					for (String key : logBidCountHash[partyID][i].keySet()) {
						int count = TmpCountHash[i].get(key);
						if (tmp < count) {
							tmp = count;
						}
					}

					if (tmp >= logMax) {
						for (String key : logBidCountHash[partyID][i]
								.keySet()) {
							int count = TmpCountHash[i].get(key);
							if (count < logMax) {
								TmpCountHash[i].put(key,
										logMax / compareSize + 1);
							}
						}
					}
				}

				// 合計出現数に加算&issueごとの最大値も取得
				int itemCountMax[] = new int[issueNum];
				for (int i = 0; i < issueNum; i++) {
					int tmpMax = 0;
					for (String key : logBidCountHash[partyID][i].keySet()) {

						int now_count = TmpCountHash[i].get(key);
						int sum_count = logBidCountHash[partyID][i].get(key);
						logBidCountHash[partyID][i].put(key,
								(now_count + sum_count));

						if (tmpMax < (now_count + sum_count)) {
							tmpMax = (now_count + sum_count);
						}
					}
					itemCountMax[i] = tmpMax;
				}

				// 論点内項目の一対比較
				for (int i = 0; i < issueNum; i++) {

					// 一対比較閾値格納用配列作成(段階数+1が閾値の数)
					int compareArray[] = new int[compareSize + 1];
					for (int j = 0; j <= compareSize; j++) {
						compareArray[j] = (int) (itemCountMax[i]
								* ((double) j / (double) compareSize));
					}

					// 一対比較結果をHashに格納
					for (String key : logBidCountHash[partyID][i].keySet()) {
						int tmp = logBidCountHash[partyID][i].get(key);
						for (int k = 0; k < compareSize; k++) {
							if (tmp >= compareArray[k]
									&& tmp <= compareArray[k + 1]) {
								logBidValueHash[partyID][i].put(key,
										compareResultArray[k]);
								break;
							}
						}
					}
				}

				// 論点ごとの重み作成のための一対比較
				int tmpMax = 0;
				for (int i = 0; i < issueNum; i++) {
					if (tmpMax < itemCountMax[i]) {
						tmpMax = itemCountMax[i];
					}
				}
				int compareArray[] = new int[compareSize + 1];
				for (int i = 0; i <= compareSize; i++) {
					compareArray[i] = (int) (tmpMax
							* ((double) i / (double) compareSize));
				}
				double tmpWeightArray[] = new double[issueNum];
				for (int i = 0; i < issueNum; i++) {
					int tmp = itemCountMax[i];
					for (int j = 0; j < compareSize; j++) {
						if (tmp >= compareArray[j]
								&& tmp <= compareArray[j + 1]) {
							tmpWeightArray[i] = compareResultArray[j];
							break;
						}
					}
				}

				// 幾何平均で論点ごとの重み作成
				double averageMatrix[][] = new double[issueNum][issueNum];
				for (int i = 0; i < issueNum; i++) {
					for (int j = 0; j < issueNum; j++) {
						averageMatrix[i][j] = tmpWeightArray[j]
								/ tmpWeightArray[i];
					}
				}
				double sumMultiply = 0.0;
				for (int i = 0; i < issueNum; i++) {
					double multiply = 1.0;
					for (int j = 0; j < issueNum; j++) {
						multiply *= averageMatrix[i][j];
					}
					multiply = Math.pow(multiply, 1.0 / issueNum);
					sumMultiply += multiply;
					issueWeightArray[partyID][i] = multiply;
				}
				for (int i = 0; i < issueNum; i++) {
					issueWeightArray[partyID][i] /= sumMultiply;
				}

				// 次のログ採取へ移行
				logNum[partyID] = 0;
				estimateCount[partyID]++;
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
		double offeredUtility = getUtility(targetBid);

		// 自分にとって好条件 or
		// 時間ギリギリなら他条件に関わらず承諾確定
		if (offeredUtility >= 0.90 || time >= 0.95) {
			return 1.0;
		}

		// 効用空間予測が完了している時
		if (isEstimated(partyNum)) {

			// 譲歩率を元に承諾確率計算
			double tmp1 = (Math.pow(time, 5.0) / 5.0);
			tmp1 += (offeredUtility - estimateMax)
					+ (offeredUtility - utilityBarometer);

			// AHP評価値を元に承諾確率を計算
			double tmp2 = (Math.pow(time, 5.0) / 5.0);
			double ahpResult = getAHPEvaluation(targetBid);
			tmp2 += (ahpResult - estimateMax) + (ahpResult - utilityBarometer);

			// 相手の予測効用値を元に承諾確率を計算
			double tmp3 = 0.0;
			for (String name : participantList.keySet()) {
				double otherUtil = getTargetUtility(name, targetBid);
				if (otherUtil < offeredUtility) {
					tmp3 += 1.0 / (partyNum - 1.0);
				} else {
					tmp3 -= 1.0 / (partyNum - 1.0);
				}
			}

			// 承諾確率を決定
			probability = tmp1 * 0.4 + tmp2 * 0.4 + tmp3 * 0.2;
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
		int bidNum = 10;
		Bid tmpBid[] = new Bid[bidNum];

		// 作成Bidの効用値上限
		double upperLimitUtility = underlimitUtility + 0.2;
		if (upperLimitUtility > 1.0) {
			upperLimitUtility = 1.0;
		}

		// 効用値が範囲内のBidを複数作成
		for (int i = 0; i < bidNum; i++) {
			Bid tmp = null;
			double util = 0.0;
			int loop = 100;
			double tmpUnderlLimit = underlimitUtility;
			do {
				tmp = generateRandomBid();
				util = getUtility(tmp);
				if (--loop <= 0) {
					loop = 100;
					tmpUnderlLimit -= 0.001;
				}
			} while (tmp != null
					&& (util < tmpUnderlLimit || util > upperLimitUtility));
			tmpBid[i] = tmp;
		}

		// 作成BidをAHPにて評価し一番評価値の高いものを採用
		double evalValue = 0.0;
		for (int i = 0; i < bidNum; i++) {
			double tmp = getAHPEvaluation(tmpBid[i]);
			if (tmp > evalValue) {
				tmp = evalValue;
				nextBid = tmpBid[i];
			}
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

		// 提案ログ現状数格納二次元配列作成
		logNum = new int[otherCount];

		// 提案ログ格納用二次元配列作成
		logBid = new Bid[otherCount][logMax];

		// 提案内出現回数カウントHash配列作成
		logBidCountHash = new HashMap[otherCount][issueNum];
		for (int i = 0; i < otherCount; i++) {
			for (int j = 0; j < issueNum; j++) {
				logBidCountHash[i][j] = new HashMap<String, Integer>(
						TemplateLogBidCountHash[j]);
			}
		}

		// 提案内出現回数カウントHash配列作成
		logBidValueHash = new HashMap[otherCount][issueNum];
		for (int i = 0; i < otherCount; i++) {
			for (int j = 0; j < issueNum; j++) {
				logBidValueHash[i][j] = new HashMap<String, Double>(
						TemplateLogBidValueHash[j]);
			}
		}

		// 予測効用空間の論点間重み格納配列作成
		issueWeightArray = new double[otherCount][issueNum];

		// 一対比較の段階数設定&結果配列作成
		compareSize = issueNum;
		compareResultArray = new double[compareSize];
		for (int j = 1; j <= compareSize; j++) {
			compareResultArray[j - 1] = (2.0 * j - 1.0)
					/ (2.0 * compareSize - 1.0);
		}

		// 予測の更新回数格納配列作成
		estimateCount = new int[otherCount];
		for (int i = 0; i < otherCount; i++) {
			estimateCount[i] = 0;
		}

		// 合算した論点間重み格納配列作成
		addupWeightArray = new double[issueNum];

		// 相手提案の効用値合計格納配列作成
		sumUtility = new double[otherCount];
		sumUtility2 = new double[otherCount];
		for (int i = 0; i < otherCount; i++) {
			sumUtility[i] = 0.0;
			sumUtility2[i] = 0.0;
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
	 * 参加者全員分の予測が作成済みであるかどうか
	 * 
	 * @param int
	 *            交渉参加人数
	 * @return boolean 作成済み=true
	 */
	private boolean isEstimated(int partyNum) {

		int otherCount = partyNum - 1;
		boolean flag = true;
		for (int i = 0; i < otherCount; i++) {
			if (estimateCount[i] <= 0) {
				flag = false;
				break;
			}
		}

		return flag;
	}

	/**
	 * 効用空間合算
	 * 
	 * @param double
	 *            自分の重み
	 */
	private void addupUtilSpace(double myWeight) {

		// 合算更新の必要がない
		// &予測が作成済みでない場合はreturn
		int timing = 0;
		for (int i = 0; i < partyNum - 1; i++) {
			timing += estimateCount[i];
		}
		if (timing == prevAddupTiming || !isEstimated(partyNum)) {
			return;
		} else {
			prevAddupTiming = timing;
		}

		// 自分以外の参加者の重み
		double otherWeight = (1.0 - myWeight) / (partyNum - 1);

		// 自分の効用空間値取得
		for (Issue tmp : utilitySpace.getDomain().getIssues()) {

			// 論点番号取得
			int issue_num = tmp.getNumber();

			// 論点の重み取得し重み付き格納
			addupWeightArray[issue_num
					- 1] = ((AdditiveUtilitySpace) utilitySpace)
							.getWeight(issue_num) * myWeight;

			// 論点内の項目事の値を取得
			IssueDiscrete tmpDiscrete = (IssueDiscrete) tmp;
			EvaluatorDiscrete evaluator = (EvaluatorDiscrete) ((AdditiveUtilitySpace) utilitySpace)
					.getEvaluator(issue_num);
			for (int j = 0; j < tmpDiscrete.getNumberOfValues(); j++) {

				ValueDiscrete value = tmpDiscrete.getValue(j);
				try {
					addupValueHash[issue_num - 1].put(value.toString(),
							evaluator.getEvaluation(value) * myWeight);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}

		// 他参加者の情報を重み付き格納
		for (int i = 0; i < partyNum - 1; i++) {
			for (int j = 0; j < issueNum; j++) {

				// 論点間の重みを重み付き格納
				addupWeightArray[j] += issueWeightArray[i][j] * otherWeight;

				// 論点内の項目事の値を重み付き格納
				for (String key : logBidValueHash[i][j].keySet()) {
					double tmp1 = addupValueHash[j].get(key);
					double tmp2 = logBidValueHash[i][j].get(key);
					addupValueHash[j].put(key, tmp1 + tmp2 * otherWeight);
				}
			}
		}
	}

	/**
	 * 自分以外の参加者の効用値予測を取得
	 * 
	 * @param String
	 *            参加者名
	 * @param Bid
	 *            予測対象提案
	 * @return double 予測効用値
	 */
	private double getTargetUtility(String name, Bid targetBid) {

		int partyID = getSenderID(name);
		double utility = 0.0;
		if (targetBid != null && partyID >= 0) {

			for (int i = 1; i <= issueNum; i++) {

				try {
					// 相手の提案を元に予測効用空間から効用を取得し重み掛けて加算
					String valName = targetBid.getValue(i).toString();
					utility += (logBidValueHash[partyID][i - 1].get(valName)
							* issueWeightArray[partyID][i - 1]);

				} catch (Exception e) {
					System.out.println("getValue Error !\n");
				}
				;
			}
		}

		return utility;
	}

	/**
	 * AHPによる提案の評価値を取得
	 * 
	 * @param Bid
	 *            予測対象提案
	 * @return double 予測評価値(効用値)
	 */
	private double getAHPEvaluation(Bid targetBid) {

		double utility = 0.0;
		if (targetBid != null && isEstimated(partyNum)) {

			for (int i = 1; i <= issueNum; i++) {

				try {
					// 合算した効用空間から効用を取得
					String valName = targetBid.getValue(i).toString();
					utility += (addupValueHash[i - 1].get(valName)
							* addupWeightArray[i - 1]);

				} catch (Exception e) {
					System.out.println("getValue Error !\n");
				}
				;
			}
		}

		return utility;
	}

	/**
	 * 譲歩度合い設定
	 */
	private void makeConcession() {

		int otherCount = partyNum - 1;
		double time = timeline.getTime();
		double tremor = 1.0; // 行動のゆらぎ幅
		double gt = 0.2; // 最高でどの程度まで譲歩するか(1.0-gt)まで

		// 参加者全員の効用値合計
		double allSum = 0.0;
		double allSum2 = 0.0;
		for (int i = 0; i < otherCount; i++) {
			allSum += sumUtility[i];
			allSum2 += sumUtility2[i];
		}

		// 現在の相手全員からの合計提案数
		int round = 0;
		for (int i = 0; i < otherCount; i++) {
			round += estimateCount[i] * logMax + logNum[i];
		}

		// 平均
		double mean = allSum / round;

		// 分散
		double variance = Math.sqrt(((allSum2) / round) - (mean * mean));
		if (Double.isNaN(variance)) {
			variance = 0.0;
		}

		// 行動の幅(論文でのd(t))
		double width = Math.sqrt(12) * variance;
		if (Double.isNaN(width)) {
			width = 0.0;
		}

		// 最大効用の推定値
		estimateMax = mean + ((1.0 - mean) * width);

		// 歩み寄り速度調整用
		double alpha = 1.0 + tremor + (7.0 * mean) - (2.0 * tremor * mean);
		double beta = alpha + (Math.random() * tremor) - (tremor / 2);

		// 歩み寄り後の最低効用値(仮)
		// 1:承諾判断用,2:提案作成用
		double tmpConsession1 = 1.0
				- (Math.pow(time, alpha) * (1.0 - estimateMax));
		double tmpConsession2 = 1.0
				- (Math.pow(time, beta) * (1.0 - estimateMax));

		// 歩み寄り度合い
		double ratio1 = (width + gt) / (1.0 - tmpConsession1);
		if (Double.isNaN(ratio1) || ratio1 > 2.0) {
			ratio1 = 2.0;
		}
		double ratio2 = (width + gt) / (1.0 - tmpConsession2);
		if (Double.isNaN(ratio2) || ratio2 > 2.0) {
			ratio2 = 2.0;
		}

		// 提案作成用最低効用値作成
		// 作成提案最低効用値と効用空間合算時の自分の重みに利用
		double tmp = ratio2 * tmpConsession2 + (1.0 - ratio2);
		if (tmp < underlimitUtility) {
			underlimitUtility = tmp;
			if (underlimitUtility < reservationValue) {
				underlimitUtility = reservationValue;
			}
			weightUtilSpace = 1.0 - ((1.0 - tmp) * 2);
			if (weightUtilSpace < 0.4) {
				weightUtilSpace = 0.4;
			}
		}

		// 承諾判断用効用値指標作成
		utilityBarometer = ratio1 * tmpConsession1 + (1.0 - ratio1);
	}

	@Override
	public String getDescription() {
		return "ANAC2015";
	}
}
