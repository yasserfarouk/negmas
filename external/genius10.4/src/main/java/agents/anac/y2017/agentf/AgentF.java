package agents.anac.y2017.agentf;

import java.util.List;

import java.util.ArrayList;
import java.util.Random;

import agents.anac.y2017.agentf.etc.bidSearch;
import agents.anac.y2017.agentf.etc.negotiationInfo;
import agents.anac.y2017.agentf.etc.negotiationStrategy;
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Inform;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.PersistentDataType;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AbstractUtilitySpace;;

/**
 * This is your negotiation party.
 */
public class AgentF extends AbstractNegotiationParty {

	private TimeLineInfo timelineInfo; // タイムライン
	private AbstractUtilitySpace utilitySpace;

	private Bid lastReceivedBid = null;
	private StandardInfoList history; // 過去の交渉情報

	private negotiationInfo negotiationInfo; // 交渉情報
	private bidSearch bidSearch; // 合意案候補の探索
	private negotiationStrategy negotiationStrategy; // 交渉戦略

	private Bid offeredBid = null; // 最近提案された合意案候補
	private int supporter_num = 0; // 支持者数
	private int CList_index = 0; // CListのインデックス：最終提案フェーズにおける遡行を行うために利用(ConcessionList)
	private boolean isPrinting = false; // デバッグ用

	@Override
	// public void init(AbstractUtilitySpace utilSpace, Deadline dl,
	// negotiator.timeline.TimeLineInfo tl, long randomSeed,
	// AgentID agentId, PersistentDataContainer storage) {
	// super.init(utilSpace, dl, tl, randomSeed, agentId, storage);
	public void init(NegotiationInfo info) {
		super.init(info);

		this.timelineInfo = timeline;
		this.utilitySpace = getUtilitySpace();

		negotiationInfo = new negotiationInfo(utilitySpace, isPrinting);
		negotiationStrategy = new negotiationStrategy(utilitySpace, negotiationInfo, isPrinting);
		try {
			bidSearch = new bidSearch(utilitySpace, negotiationInfo, isPrinting);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		if (getData().getPersistentDataType() == PersistentDataType.STANDARD) {
			history = (StandardInfoList) getData().get();
			if (history.size() > 0) {
				for (StandardInfo standardInfo : history) {
					if (standardInfo.getAgreement().get2() > 0.1) {
						try {
							negotiationInfo.updatePBList(standardInfo.getAgreement().get1());
						} catch (Exception e) {
							e.printStackTrace();
						}
					}
				}
			}
			// throw new IllegalStateException("need standard persistent data");
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
		// negotiationInfo.saveFrequency();
		// negotiationInfo.saveSort();

		double time = timelineInfo.getTime(); // 現在の時刻
		negotiationInfo.updateTimeScale(time); // 自身の手番が回ってくる時間間隔を記録

		// 最終提案フェーズにおけるアクション
		ArrayList<Bid> CList = negotiationInfo.getPBList();
		if (time > 1.0 - negotiationInfo.getTimeScale() * (CList.size() + 1)) {
			// System.out.println("FinalAction!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
			try {
				return chooseFinalAction(offeredBid, CList);
			} catch (Exception e) {
				System.out.println("最終提案フェーズにおけるActionの選択に失敗しました");
				e.printStackTrace();
			}
		}

		// Acceptの判定
		if (validActions.contains(Accept.class) && negotiationStrategy.selectAccept(offeredBid, time)) {
			return new Accept(getPartyId(), lastReceivedBid);
		}
		// EndNegotiationの判定
		if (negotiationStrategy.selectEndNegotiation(time)) {
			return new EndNegotiation(getPartyId());
		}
		// 他のプレイヤーに新たなBidをOffer
		return OfferAction();

	}

	public Action OfferAction() {
		Bid offerBid = bidSearch.getBid(utilitySpace.getDomain().getRandomBid(new Random()),
				negotiationStrategy.getThreshold(timelineInfo.getTime()), timelineInfo.getTime());
		return OfferBidAction(offerBid);
	}

	public Action OfferBidAction(Bid offerBid) {
		// System.out.println("call CA:"+utilitySpace.getUtility(offerBid));
		negotiationInfo.updateMyBidHistory(offerBid);
		return new Offer(getPartyId(), offerBid);
	}

	public Action chooseFinalAction(Bid offeredBid, ArrayList<Bid> CList) throws Exception {
		double offeredBid_util = 0;
		double rv = utilitySpace.getReservationValue();

		if (offeredBid != null) {
			offeredBid_util = utilitySpace.getUtility(offeredBid);
		}
		if (CList_index >= CList.size()) {
			if (offeredBid_util >= rv)
				return new Accept(getPartyId(), lastReceivedBid); // 遡行を行っても合意が失敗する場合，Acceptする
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
			return new Accept(getPartyId(), lastReceivedBid);

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
	// private int callRM = 1;
	@Override
	public void receiveMessage(AgentID sender, Action action) {

		// if(action instanceof Inform)
		// System.out.println("Inf RM:"+callRM+++","+sender);
		// if(action instanceof Accept)
		// System.out.println("Acc RM:"+callRM+++","+sender);
		// if(action instanceof Offer)
		// //System.out.println("Off
		// RM:"+callRM+++","+sender+":"+utilitySpace.getUtility(((Offer)
		// action).getBid()));
		// System.out.println("Off RM:"+callRM+++","+sender+":"+((Offer)
		// action).getBid());
		// if(action instanceof EndNegotiation){
		// System.out.println("End RM:"+callRM+++","+sender);
		// }
		super.receiveMessage(sender, action);
		if (action instanceof Offer) {
			lastReceivedBid = ((Offer) action).getBid();
		}
		// if(isPrinting){ System.out.println("Sender:"+sender+",
		// Action:"+action); }

		if (action != null) {
			if (action instanceof Inform && ((Inform) action).getName() == "NumberOfAgents"
					&& ((Inform) action).getValue() instanceof Integer) {
				Integer opponentsNum = (Integer) ((Inform) action).getValue();
				negotiationInfo.updateOpponentsNum(opponentsNum);
				// if(isPrinting){ System.out.println("NumberofNegotiator:" +
				// negotiationInfo.getNegotiatorNum());}
			} else if (action instanceof Accept) {
				if (!negotiationInfo.getOpponents().contains(sender)) {
					negotiationInfo.initOpponent(sender);
				} // 初出の交渉者は初期化
				supporter_num++;
				negotiationInfo.updateAcceptNum(sender);
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
					System.out.println("交渉情報の更新に失敗しました");
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
						System.out.println("PBListの更新に失敗しました"); // PopularBidHistoryを更新
						e.printStackTrace();
					}
				}
			}
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2017";
	}

}
