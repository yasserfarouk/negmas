package agents.anac.y2016.agenthp2;

import java.util.HashMap;

import genius.core.Bid;
import genius.core.issue.ISSUETYPE;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.utility.AbstractUtilitySpace;

/**
 * AHP Evaluation class
 * 
 * @author Hiroyuki Shinohara
 * @date 2016/03/30
 */
public class module_AHP {

	private int issueNum; // 効用空間の論点数
	private AbstractUtilitySpace myUtilitySpace; // 自分の効用空間
	private int issueElementCount[]; // 論点内の要素数
	private HashMap<String, Integer>[][] valueCountHash; // 提案内出現回数カウント用Hash
	private HashMap<String, Double>[][] expectUtilSpaceHash; // 予測効用空間保存用Hash
	private double expectIssueWeightArray[][]; // 予測論点間重み保存用配列
	private double geometricAverageMatrix[][]; // 幾何平均計算用一対比較行列格納用配列
	private int pairComparisonSize; // 一対比較の段階数
	private double pairComparisonResultArray[]; // 一対比較の結果参照用配列
	private int pairComparisonThresholdArray[]; // 一対比較の出現数閾値格納用配列
	private int valueCountMaxArray[]; // 論点ごと選択肢最大出現回数格納用配列
	private double evaluationUtilityWeightArray[]; // AHP評価値に利用する参加者ごとの効用値重み格納用配列
	private ISSUETYPE issueTypeArray[]; // 論点ごとの論点タイプの保存用配列

	/**
	 * コンストラクタ（配列等初期化）
	 * 
	 * @param UtilitySpace
	 *            効用空間
	 * @param int 参加者数
	 */
	public module_AHP(AbstractUtilitySpace utilitySpace, int participantNum) {

		// 論点数取得
		issueNum = utilitySpace.getDomain().getIssues().size();

		// 効用空間保存
		myUtilitySpace = utilitySpace;

		// 自分以外の参加人数
		int otherCount = participantNum - 1;

		// 予測効用保存&提案カウント用Hash作成
		// 各論点タイプを保存
		valueCountHash = new HashMap[otherCount][issueNum];
		expectUtilSpaceHash = new HashMap[otherCount][issueNum];
		issueTypeArray = new ISSUETYPE[issueNum];
		for (int i = 0; i < otherCount; i++) {
			for (Issue tmp : utilitySpace.getDomain().getIssues()) {

				// 論点番号取得
				int issue_num = tmp.getNumber();

				// 論点タイプ保存
				issueTypeArray[issue_num - 1] = tmp.getType();

				// Hash作成
				expectUtilSpaceHash[i][issue_num - 1] = new HashMap<String, Double>();
				valueCountHash[i][issue_num - 1] = new HashMap<String, Integer>();
				switch (tmp.getType()) {

				// 論点が離散値の場合
				case DISCRETE:
					IssueDiscrete tmpDiscrete = (IssueDiscrete) tmp;
					for (int j = 0; j < tmpDiscrete.getNumberOfValues(); j++) {
						expectUtilSpaceHash[i][issue_num - 1].put(tmpDiscrete
								.getValue(j).toString(), 0.0);
						valueCountHash[i][issue_num - 1].put(tmpDiscrete
								.getValue(j).toString(), 0);
					}
					break;

				// 論点が連続値の場合
				case INTEGER:
					IssueInteger tmpInteger = (IssueInteger) tmp;
					int lowest = tmpInteger.getLowerBound();
					int highest = tmpInteger.getUpperBound();
					for (int j = lowest; j <= highest; j++) {
						expectUtilSpaceHash[i][issue_num - 1].put(
								String.valueOf(j), 0.0);
						valueCountHash[i][issue_num - 1].put(String.valueOf(j),
								0);
					}
					break;

				case OBJECTIVE:
				case REAL:
				case UNKNOWN:
				default:
					break;
				}
			}
		}

		// 論点内要素数保存配列作成
		issueElementCount = new int[issueNum];
		for (Issue tmp : utilitySpace.getDomain().getIssues()) {
			switch (tmp.getType()) {

			// 論点が離散値の場合
			case DISCRETE:
				issueElementCount[tmp.getNumber() - 1] = ((IssueDiscrete) tmp)
						.getNumberOfValues();
				break;

			// 論点が連続値の場合
			case INTEGER:
				issueElementCount[tmp.getNumber() - 1] = ((IssueInteger) tmp)
						.getUpperBound()
						- ((IssueInteger) tmp).getLowerBound()
						+ 1;
				break;

			case OBJECTIVE:
			case REAL:
			case UNKNOWN:
			default:
				break;
			}
		}

		// 予測効用空間の論点間重み格納配列作成
		expectIssueWeightArray = new double[otherCount][issueNum];

		// 一対比較の段階数設定&閾値格納用配列作成&結果配列作成
		pairComparisonSize = 9; // 1,3,5,7,9 + 偶数値
		pairComparisonThresholdArray = new int[pairComparisonSize + 1]; // 閾値なので+1
																		// |A1|A2|A3|：数値3,閾値4
		pairComparisonResultArray = new double[pairComparisonSize];
		for (int j = 1; j <= pairComparisonSize; j++) {
			pairComparisonResultArray[j - 1] = (double) j
					/ (double) pairComparisonSize;
		}

		// 幾何平均用一対比較行列格納配列作成
		int tmpMax = issueNum;
		for (int i = 0; i < issueElementCount.length; i++) {
			if (tmpMax < issueElementCount[i]) {
				tmpMax = issueElementCount[i];
			}
		}
		geometricAverageMatrix = new double[tmpMax][tmpMax];

		// 論点ごと要素カウント最大格納用配列作成
		valueCountMaxArray = new int[issueNum];

		// AHP評価値用効用重み配列作成
		evaluationUtilityWeightArray = new double[otherCount];
		for (int i = 0; i < otherCount; i++) {
			evaluationUtilityWeightArray[i] = 1.0 / (double) otherCount;
		}
	}

	/**
	 * 相手からのBidの内容をカウント
	 * 
	 * @param int 参加者番号(0〜)
	 * @param Bid
	 *            カウント対象Bid
	 */
	public void countBid(int partyID, Bid targetBid) {

		// Bidの内容を論点ごとにHashにカウント
		for (int i = 1; i <= issueNum; i++) {
			try {
				if (valueCountHash[partyID][i - 1].containsKey(targetBid
						.getValue(i).toString())) {
					int tmp = valueCountHash[partyID][i - 1].get(targetBid
							.getValue(i).toString());
					valueCountHash[partyID][i - 1].put(targetBid.getValue(i)
							.toString(), tmp + 1);
				}
			} catch (Exception e) {
				System.out.println("Offerd Bid From Party " + partyID
						+ " Is Wrong");
				System.out
						.println("Wrong Bid Content: " + targetBid.toString());
			}
		}
	}

	/**
	 * 相手の効用空間予測更新(一対比較&幾何平均)
	 * 
	 * @param int 参加者番号(0〜)
	 */
	public void updateExpectUtilitySpace(int partyID) {

		// 論点内項目の評価値予測
		for (int i = 0; i < issueNum; i++) {

			// 論点内要素のカウントを配列に格納
			int arrayKey = 0;
			int elementCount[] = new int[issueElementCount[i]];
			for (String key : valueCountHash[partyID][i].keySet()) {

				int count = valueCountHash[partyID][i].get(key);
				elementCount[arrayKey++] = count;
			}

			// 数え上げ結果を元にした論点内の各要素への重要度の割り当て
			double elementPairComparisonArray[] = new double[issueElementCount[i]];
			switch (issueTypeArray[i]) {

			// 論点が離散値の場合
			case DISCRETE:

				// 評価が最大と予想される要素(=最大の出現数の要素)のカウントも保存
				int tmpMax = 0;
				for (int k = 0; k < elementCount.length; k++) {
					if (tmpMax < elementCount[k]) {
						tmpMax = elementCount[k];
					}
				}
				valueCountMaxArray[i] = tmpMax;

				// 論点内要素評価値予測のための一対比較(重要度割り当て)
				setPairComparisonArray(elementCount, elementPairComparisonArray);

				break;

			// 論点が連続値の場合
			case INTEGER:

				// 連続値の前半と後半それぞれの出現数の和を取得
				int elementSize = elementCount.length;
				int sum_former = 0,
				sum_latter = 0;
				for (int k = 0; k < elementSize; k++) {
					if ((elementSize / 2) > k) {
						sum_former += elementCount[k];
					} else {
						sum_latter += elementCount[k];
					}
				}

				// 前半の方が出現数の総和が多い: 連続値に効用は反比例(1.0→0.0)
				if (sum_former > sum_latter) {

					// 評価が最大の要素(=最小の連続値)のカウントも保存
					valueCountMaxArray[i] = elementCount[0];

					// 最小の連続値には重要度9を割り当て,最大の連続値には重要度1を割り当て(9→1)
					double offset = pairComparisonResultArray[pairComparisonSize - 1]
							- pairComparisonResultArray[0];
					for (int j = elementSize - 1; j >= 0; j--) {
						elementPairComparisonArray[elementSize - j - 1] = pairComparisonResultArray[0]
								+ (offset * (double) j / (double) (elementSize - 1));
					}

					// 後半の方が出現数の総和が多い: 連続値に効用は比例(0.0→1.0)
				} else {

					// 評価が最大の要素(=最大の連続値)のカウントも保存
					valueCountMaxArray[i] = elementCount[elementSize - 1];

					// 最小の連続値には重要度1を割り当て,最大の連続値には重要度9を割り当て(1→9)
					double offset = pairComparisonResultArray[pairComparisonSize - 1]
							- pairComparisonResultArray[0];
					for (int j = elementSize - 1; j >= 0; j--) {
						elementPairComparisonArray[j] = pairComparisonResultArray[0]
								+ (offset * (double) j / (double) (elementSize - 1));
					}
				}

				break;

			case OBJECTIVE:
			case REAL:
			case UNKNOWN:
			default:
				break;
			}

			// 一対比較結果を幾何平均で論点内要素評価値予測(最大1になるよう正規化)
			double elementGeometricAverageArray[] = new double[issueElementCount[i]];
			setGeometricAverageArray(issueElementCount[i],
					elementPairComparisonArray, elementGeometricAverageArray);
			double maxGeometricAverage = 0.0;
			for (int j = 0; j < issueElementCount[i]; j++) {
				if (maxGeometricAverage < elementGeometricAverageArray[j]) {
					maxGeometricAverage = elementGeometricAverageArray[j];
				}
			}
			for (int j = 0; j < issueElementCount[i]; j++) {
				elementGeometricAverageArray[j] /= maxGeometricAverage;
			}

			// 予測評価値をHashに保存
			arrayKey = 0;
			for (String key : valueCountHash[partyID][i].keySet()) {
				expectUtilSpaceHash[partyID][i].put(key,
						elementGeometricAverageArray[arrayKey++]);
			}
		}

		// 論点間重み予測のための一対比較
		double weightPairComparisonArray[] = new double[issueNum];
		setPairComparisonArray(valueCountMaxArray, weightPairComparisonArray);

		// 一対比較結果を幾何平均で論点間重み予測(総和1になるよう正規化)
		setGeometricAverageArray(issueNum, weightPairComparisonArray,
				expectIssueWeightArray[partyID]);
		double sumGeometricAverage = 0.0;
		for (int i = 0; i < issueNum; i++) {
			sumGeometricAverage += expectIssueWeightArray[partyID][i];
		}
		for (int i = 0; i < issueNum; i++) {
			expectIssueWeightArray[partyID][i] /= sumGeometricAverage;
		}

		// log
		/*
		 * System.out.println("Party ID = "+partyID); for(int
		 * i=0;i<issueNum;i++){ System.out.println("Issue " + (i+1) +
		 * " Count = " + valueCountHash[partyID][i].toString());
		 * System.out.println("Issue " + (i+1) + " Expect = " +
		 * expectUtilSpaceHash[partyID][i].toString());
		 * System.out.println("Issue " + (i+1) + " Weitght = " +
		 * expectIssueWeightArray[partyID][i]); }
		 */
	}

	/**
	 * 一対比較結果取得
	 * 
	 * @param int[] 一対比較対象配列
	 * @param double[] 一対比較結果配列
	 */
	private void setPairComparisonArray(int targetArray[],
			double pairComparison[]) {

		int arraySize = targetArray.length;

		// 一対比較対象の最大値&最小値取得
		int tmpMaxCount = -1;
		int tmpMinCount = -1;
		for (int i = 0; i < arraySize; i++) {
			if (tmpMaxCount == -1 || tmpMaxCount < targetArray[i]) {
				tmpMaxCount = targetArray[i];
			}
			if (tmpMinCount == -1 || tmpMinCount > targetArray[i]) {
				tmpMinCount = targetArray[i];
			}
		}
		if (tmpMaxCount - tmpMinCount <= pairComparisonSize)
			tmpMinCount = 0;

		// 一対比較閾値格納用配列作成(段階数+1が閾値の数)
		// [最小,最大]の範囲を分割
		// 重要度割り当て用範囲を一定でなく重み付け
		double tmpRangeLimit = tmpMaxCount - tmpMinCount;
		double rangeWeightArray[] = { 0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.55,
				0.70, 0.85, 1.00 }; // 偶数利用+重要度5-9に重み
		for (int i = 0; i <= pairComparisonSize; i++) {
			pairComparisonThresholdArray[i] = (int) ((double) tmpRangeLimit * rangeWeightArray[i]);
			pairComparisonThresholdArray[i] += tmpMinCount;
		}

		// 一対比較
		for (int i = 0; i < arraySize; i++) {

			int tmp = targetArray[i];
			for (int j = 0; j < pairComparisonSize; j++) {
				if (tmp >= pairComparisonThresholdArray[j]
						&& tmp <= pairComparisonThresholdArray[j + 1]) {
					pairComparison[i] = pairComparisonResultArray[j];
					break;
				}
			}
		}
	}

	/**
	 * 重要度割り当て(一対比較)結果より幾何平均計算
	 * 
	 * @param int 一対比較行列サイズ
	 * @param double[] 一対比較結果配列
	 * @param double[] 幾何平均格納用配列
	 */
	private void setGeometricAverageArray(int matrixSize,
			double pairComparison[], double geometricAverage[]) {

		// 一対比較結果より一対比較行列作成
		for (int i = 0; i < matrixSize; i++) {
			for (int j = 0; j < matrixSize; j++) {
				geometricAverageMatrix[i][j] = pairComparison[i]
						/ pairComparison[j];
			}
		}

		// 一対比較行列の行方向に幾何平均計算
		for (int i = 0; i < matrixSize; i++) {
			double multiply = 1.0;
			for (int j = 0; j < matrixSize; j++) {
				multiply *= geometricAverageMatrix[i][j];
			}
			geometricAverage[i] = Math.pow(multiply, 1.0 / matrixSize);
		}
	}

	/**
	 * AHPの評価値取得
	 * 
	 * @param Bid
	 *            評価対象提案
	 * @return double AHP評価値
	 */
	public double getAHPEvaluation(Bid targetBid) {

		// 予測効用の重み付き総和をAHP評価値
		double evaluationValue = 0.0;
		for (int i = 0; i < evaluationUtilityWeightArray.length; i++) {
			evaluationValue += getExpectUtility(i, targetBid)
					* evaluationUtilityWeightArray[i];
		}

		return evaluationValue;
	}

	/**
	 * 提案の予測効用値を計算
	 * 
	 * @param int 参加者番号(0〜)
	 * @param Bid
	 *            提案
	 * @return double 予測効用値
	 */
	public double getExpectUtility(int partyID, Bid targetBid) {

		double utility = 0.0;
		if (targetBid != null) {

			for (int i = 1; i <= issueNum; i++) {

				try {
					// 相手の提案を元に予測効用空間から効用を取得し重み掛けて加算
					String valName = targetBid.getValue(i).toString();
					utility += (expectUtilSpaceHash[partyID][i - 1]
							.get(valName) * expectIssueWeightArray[partyID][i - 1]);

				} catch (Exception e) {
					System.out
							.println("Get Expect Utility Target Bid Is Wrong");
					System.out.println("Wrong Bid Content: "
							+ targetBid.toString());
				}
			}
		}

		return utility;
	}

	/**
	 * 特定の参加者から見たAHP評価値取得
	 * 
	 * @param Bid
	 *            評価対象Bid
	 * @param int 参加者番号(0〜)
	 * @return double 特定の参加者から見たAHP評価値
	 */
	public double getAHPEvaluationForOpponents(int partyID, Bid targetBid) {

		double evaluationValue = 0.0;
		double weight = 1.0 / (double) evaluationUtilityWeightArray.length; // 重みは一定

		// 指定交渉相手以外の予測効用を重み付き加算
		for (int i = 0; i < evaluationUtilityWeightArray.length; i++) {
			if (i != partyID) {
				evaluationValue += getExpectUtility(i, targetBid) * weight;
			}
		}

		// 自分の効用値を重み付き加算
		double myUtility = 0.0;
		try {
			myUtility = myUtilitySpace.getUtility(targetBid);
		} catch (Exception e) {
			System.out.println("Get My Utility Target Bid Is Wrong");
			System.out.println("Wrong Bid Content: " + targetBid.toString());
		}
		evaluationValue += myUtility * weight;

		return evaluationValue;
	}
}
