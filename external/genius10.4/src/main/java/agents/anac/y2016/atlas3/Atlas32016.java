package agents.anac.y2016.atlas3;

import java.util.List;

import java.util.ArrayList;

import agents.anac.y2016.atlas3.etc.bidSearch;
import agents.anac.y2016.atlas3.etc.negotiationInfo;
import agents.anac.y2016.atlas3.etc.negotiationStrategy;
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Inform;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AbstractUtilitySpace;

/**
 * This is your negotiation party.
 */

public class Atlas32016 extends AbstractNegotiationParty {
	private TimeLineInfo timeLineInfo; // タイムライン
	private AbstractUtilitySpace utilitySpace;
	private negotiationInfo negotiationInfo; // 交渉情報
	private bidSearch bidSearch; // 合意案候補の探索
	private negotiationStrategy negotiationStrategy; // 交渉戦略

	private Bid offeredBid = null; // 最近提案された合意案候補
	private int supporter_num = 0; // 支持者数
	private int CList_index = 0; // CListのインデックス：最終提案フェーズにおける遡行を行うために利用(ConcessionList)

	private boolean isPrinting = false; // デバッグ用

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		if (isPrinting)
			System.out.println("*** SampleAgent2016 v1.0 ***");

		this.timeLineInfo = info.getTimeline();
		this.utilitySpace = getUtilitySpace();
		negotiationInfo = new negotiationInfo(utilitySpace, isPrinting);
		negotiationStrategy = new negotiationStrategy(utilitySpace,
				negotiationInfo, isPrinting);

		try {
			bidSearch = new bidSearch(utilitySpace, negotiationInfo,
					isPrinting);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
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
		double time = timeLineInfo.getTime(); // 現在の時刻
		negotiationInfo.updateTimeScale(time); // 自身の手番が回ってくる時間間隔を記録

		// 最終提案フェーズにおけるアクション
		ArrayList<Bid> CList = negotiationInfo.getPBList();
		if (time > 1.0 - negotiationInfo.getTimeScale() * (CList.size() + 1)) {
			try {
				return chooseFinalAction(offeredBid, CList);
			} catch (Exception e) {
				System.out.println(
						"最終提案フェーズにおけるActionの選択に失敗しました");
				e.printStackTrace();
			}
		}

		// Acceptの判定
		if (validActions.contains(Accept.class)
				&& negotiationStrategy.selectAccept(offeredBid, time)) {
			return new Accept(getPartyId(), offeredBid);
		}

		// EndNegotiationの判定
		if (negotiationStrategy.selectEndNegotiation(time)) {
			return new EndNegotiation(getPartyId());
		}

		// 他のプレイヤーに新たなBidをOffer
		return OfferAction();
	}

	public Action OfferAction() {
		Bid offerBid = bidSearch.getBid(
				utilitySpace.getDomain().getRandomBid(null),
				negotiationStrategy.getThreshold(timeLineInfo.getTime()));
		return OfferBidAction(offerBid);
	}

	public Action OfferBidAction(Bid offerBid) {
		negotiationInfo.updateMyBidHistory(offerBid);
		return new Offer(getPartyId(), offerBid);
	}

	public Action chooseFinalAction(Bid offeredBid, ArrayList<Bid> CList)
			throws Exception {
		double offeredBid_util = 0;
		double rv = utilitySpace.getReservationValue();

		if (offeredBid != null) {
			offeredBid_util = utilitySpace.getUtility(offeredBid);
		}
		if (CList_index >= CList.size()) {
			if (offeredBid_util >= rv)
				return new Accept(getPartyId(), offeredBid); // 遡行を行っても合意が失敗する場合，Acceptする
			else
				OfferAction();
		}

		// CListの遡行
		Bid CBid = CList.get(CList_index);
		double CBid_util = utilitySpace.getUtility(CBid);
		if (CBid_util > offeredBid_util && CBid_util > rv) {
			CList_index++;
			OfferBidAction(CBid);
		} else if (offeredBid_util > rv)
			return new Accept(getPartyId(), offeredBid);

		return OfferAction();
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
		// プレイヤーのアクションを受信
		super.receiveMessage(sender, action);

		if (isPrinting) {
			System.out.println("Sender:" + sender + ", Action:" + action);
		}

		if (action != null) {
			if (action instanceof Inform
					&& ((Inform) action).getName() == "NumberOfAgents"
					&& ((Inform) action).getValue() instanceof Integer) {
				Integer opponentsNum = (Integer) ((Inform) action).getValue();
				negotiationInfo.updateOpponentsNum(opponentsNum);
				if (isPrinting) {
					System.out.println("NumberofNegotiator:"
							+ negotiationInfo.getNegotiatorNum());
				}
			} else if (action instanceof Accept) {
				if (!negotiationInfo.getOpponents().contains(sender)) {
					negotiationInfo.initOpponent(sender);
				} // 初出の交渉者は初期化
				supporter_num++;
			} else if (action instanceof Offer) {
				if (!negotiationInfo.getOpponents().contains(sender)) {
					negotiationInfo.initOpponent(sender);
				} // 初出の交渉者は初期化
				supporter_num = 1; // supporterをリセット
				offeredBid = ((Offer) action).getBid(); // 提案された合意案候補
				try {
					negotiationInfo.updateInfo(sender, offeredBid);
				} // 交渉情報を更新
				catch (Exception e) {
					System.out.println(
							"交渉情報の更新に失敗しました");
					e.printStackTrace();
				}
			} else if (action instanceof EndNegotiation) {
			}

			// 自身以外が賛成している合意案候補を記録（自身以外のエージェントを1つの交渉者とみなす．そもそも自身以外のエージェントが二人以上非協力であれば，自身の選択に関わらず合意は不可能である）
			if (supporter_num == negotiationInfo.getNegotiatorNum() - 1) {
				if (offeredBid != null) {
					try {
						negotiationInfo.updatePBList(offeredBid);
					} catch (Exception e) {
						System.out.println(
								"PBListの更新に失敗しました"); // PopularBidHistoryを更新
						e.printStackTrace();
					}
				}
			}
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2016";
	}

}
