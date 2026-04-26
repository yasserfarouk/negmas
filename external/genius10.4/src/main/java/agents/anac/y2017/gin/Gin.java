package agents.anac.y2017.gin;

import java.util.List;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map.Entry;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Value;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;

/**
 * Agent Name: Gin. Team members: Jin Tanda. Affiliation: Nagoya Institute of
 * Technology, Takayuki Ito. Laboratory Contact person: Jin Tanda. Contact
 * E-mail: tanda.jin@itolab.nitech.ac.jp
 */
public class Gin extends AbstractNegotiationParty {

	private Bid lastReceivedBid = null;
	private NegotiationInfo info = null;
	private StandardInfoList history;
	private int fitToOpponent = 0;

	@Override
	public void init(NegotiationInfo info) {

		super.init(info);

		System.out.println("Discount Factor is "
				+ getUtilitySpace().getDiscountFactor());
		System.out.println("Reservation Value is "
				+ getUtilitySpace().getReservationValueUndiscounted());
		// if you need to initialize some variables, please initialize them
		// below
		this.info = info;
		initBidTable(); // 論点数に応じたhashmapによるデータ構造を定義
		checkOpponentConcessions();
	}

	/**
	 * Each round this method gets called and ask you to accept or offer. The
	 * first party in the first round is a bit different, it can only propose an
	 * offer.
	 *
	 * @param validActions
	 *            Either a list containing both accept and offer or only offer.
	 * @return The chosen action.
	 */
	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		double myUtil = 0;
		double threshold = 1;
		boolean reservationPriceFlag = false;
		Bid bid = null;
		if (lastReceivedBid != null)
			myUtil = utilitySpace.getUtility(lastReceivedBid);
		threshold = concessionsFunction();
		double reservationPrice = utilitySpace
				.getReservationValueUndiscounted();
		if (threshold < reservationPrice)
			reservationPriceFlag = true;
		else
			reservationPriceFlag = false;
		// if we are the first party, offer.
		if ((lastReceivedBid == null || !validActions.contains(Accept.class)
				|| myUtil < threshold) && reservationPriceFlag == false) {
			if (lastReceivedBid == null || info.getTimeline().getTime() < 0.3) {
				while (myUtil < concessionsFunction()) {
					bid = generateRandomBid();
					myUtil = utilitySpace.getUtility(bid);
				}
			} else
				bid = generateMyBidFromTable();
			return new Offer(getPartyId(), bid);
		} else {
			// printTable();
			return new Accept(getPartyId(), lastReceivedBid);
		}
	}

	private void checkOpponentConcessions() {
		history = (StandardInfoList) getData().get();
		if (!history.isEmpty()) {
			int cnt = 0;
			for (StandardInfo si : history) {
				if (si.getAgreement().get1() == null) {
					cnt++;
				}
			}
			if (((float) cnt / (float) history.size()) > 0.4) {
				// 3者間の交渉が過去情報から参照してうまくいっていない時
				// 自身の譲歩関数を定義し直す
				fitToOpponent = 2;
			} else if (((float) cnt / (float) history.size()) > 0.2) {
				fitToOpponent = 1;
			} else
				fitToOpponent = 0;
		}
	}

	private Bid generateMyBidFromTable() {
		Bid myBid = generateRandomBid();
		// 各論点において、bidを頻出順にソート
		List<List<Entry<Value, Integer>>> lists = new ArrayList<List<Entry<Value, Integer>>>();
		for (int i = 0; i < bidTable.length; i++) {
			List<Entry<Value, Integer>> list_entries = new ArrayList<Entry<Value, Integer>>(
					bidTable[i].entrySet());
			// 比較関数Comparatorを使用してMap.Entryの値を比較する（降順）
			Collections.sort(list_entries,
					new Comparator<Entry<Value, Integer>>() {
						// compareを使用して値を比較する
						@Override
						public int compare(Entry<Value, Integer> obj1,
								Entry<Value, Integer> obj2) {
							// 降順
							return obj2.getValue().compareTo(obj1.getValue());
						}
					});
			lists.add(list_entries);
		}
		// 3者間における最良のものを寄せ集めたbid候補でbidを初期化
		for (int i = 0; i < lists.size(); i++) {
			Value candidate = lists.get(i).get(0).getKey();
			myBid = myBid.putValue(i + 1, candidate);
		}
		// 各論点をtableを元に一つずつ変更し、myUtilの値に応じてループ
		int maxLoop = 1000;
		int cnt = 0;
		double permissionRange;
		while (utilitySpace.getUtility(myBid) < concessionsFunction()) {
			if (cnt > maxLoop) {
				double myUtil = utilitySpace.getUtility(myBid);
				while (myUtil < concessionsFunction()) {
					myBid = generateRandomBid();
					myUtil = utilitySpace.getUtility(myBid);
				}
				// System.out.println("強制的にbidを決定します");
				break;
			}
			for (int i = 0; i < lists.size(); i++) {
				List<Entry<Value, Integer>> issue = lists.get(i);
				permissionRange = ((double) cnt / (double) (maxLoop + 1))
						* (issue.size());
				// System.out.println("cnt:issue.size() = " + cnt + " " +
				// issue.size());
				// System.out.println("permissionRange = " + permissionRange);
				Value value = issue.get((int) (Math.random() * permissionRange))
						.getKey();
				myBid = myBid.putValue(i + 1, value);
				// System.out.println("myBid = " + myBid);
				// System.out.println("Utility = " +
				// utilitySpace.getUtility(myBid));
				if (utilitySpace.getUtility(myBid) > concessionsFunction())
					break;
			}
			cnt++;
		}
		// System.out.println("Gin's Bid = " + myBid);
		double time = info.getTimeline().getTime();
		if (time > 0.98) {
			for (StandardInfo si : history) {
				if (si.getAgreement() != null) {
					myBid = si.getAgreement().get1();
					// System.out.println("getAgreementBid = " + myBid);
				}
			}
		}
		return myBid;
	}

	private double concessionsFunction() {
		double time = info.getTimeline().getTime(); // 全体時間(round)に対する正規化された時間:0-1
		double fx = 0;
		// System.out.println("fitToOpponent = " + fitToOpponent);
		if (fitToOpponent == 0) {
			if (time < 0.85)
				fx = 0.9;
			else if (time < 0.95)
				fx = 0.88;
			else
				fx = 0.80;
		} else if (fitToOpponent == 1) {
			if (time < 0.85)
				fx = 0.95 - (time / 3.0);
			else if (time < 0.95)
				fx = 0.8;
			else
				fx = 0.7;
		} else {
			if (time < 0.7) {
				fx = 0.95 - (time / 3.0);
			} else {
				fx = 0.5;
			}
		}
		return fx;
	}

	// 三者間における頻出bid格納テーブルを表示
	private void printTable() {
		for (int i = 0; i < bidTable.length; i++) {
			System.out.println("bidTable[" + i + "]" + ":" + bidTable[i]);
		}
	}

	// 三者間における頻出bid格納テーブルを初期化
	private HashMap[] bidTable;

	private void initBidTable() {
		int issueNum = generateRandomBid().getIssues().size();
		bidTable = new HashMap[issueNum];
		for (int i = 0; i < issueNum; i++) {
			bidTable[i] = new HashMap<Value, Integer>();
		}
	}

	// 三者間における頻出bid格納テーブルを更新
	private void updateOurTable(Bid bid) {
		Value key = null;
		int value = 1;
		for (int i = 0; i < bid.getIssues().size(); i++) {
			key = bid.getValue(i + 1);
			if (bidTable[i].get(key) != null) {
				value = (int) bidTable[i].get(key) + 1;
			}
			bidTable[i].put(key, value);
		}
		return;
	}

	/**
	 * All offers proposed by the other parties will be received as a message.
	 * You can use this information to your advantage, for example to predict
	 * their utility.
	 *
	 * @param sender
	 *            The party that did the action. Can be null.
	 * @param action
	 *            The action that party did.
	 */
	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		if (action instanceof Offer) {
			lastReceivedBid = ((Offer) action).getBid();
			updateOurTable(lastReceivedBid);
			// System.out.println("getBid,name = " + sender + " , "+
			// lastReceivedBid);
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2017";
	}

}
