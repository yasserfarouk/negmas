package agents.anac.y2016.myagent;

import java.util.List;

import agents.anac.y2016.myagent.etc.bidSearch;
import agents.anac.y2016.myagent.etc.negotiationInfo;
import agents.anac.y2016.myagent.etc.negotiationStrategy;
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

public class MyAgent extends AbstractNegotiationParty {
	private TimeLineInfo timeLineInfo; // タイムライン
	private AbstractUtilitySpace utilitySpace;
	private negotiationInfo negotiationInfo; // 交渉情報
	private agents.anac.y2016.myagent.etc.bidSearch bidSearch; // 合意案候補の探索
	private agents.anac.y2016.myagent.etc.negotiationStrategy negotiationStrategy; // 交渉戦略
	private Bid offeredBid = null; // 最近提案された合意案候補

	private boolean isPrinting = false; // デバッグ用

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);

		if (isPrinting)
			System.out.println("*** SampleAgent2016 v1.0 ***");

		this.timeLineInfo = timeline;
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

		// Acceptの判定
		if (validActions.contains(Accept.class)
				&& negotiationStrategy.selectAccept(offeredBid, time)) {
			return new Accept(getPartyId(), offeredBid);
		}

		negotiationInfo.updateLast(false);

		// EndNegotiationの判定
		if (negotiationStrategy.selectEndNegotiation(time)) {
			return new EndNegotiation(getPartyId());
		}

		// 他のプレイヤーに新たなBidをOffer
		Bid offerBid = bidSearch.getBid(
				utilitySpace.getDomain().getRandomBid(null),
				negotiationStrategy.getThreshold(timeLineInfo.getTime()));

		// offeredBidの更新
		offeredBid = offerBid;

		negotiationInfo.updateMyBidHistory(offerBid);
		return new Offer(getPartyId(), offerBid);
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

		// System.out.println("Sender:"+sender+", Action:"+action);

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
			}
			if (action instanceof Accept) {
				if (!negotiationInfo.getOpponents().contains(sender)) {
					negotiationInfo.initOpponent(sender);
				} // 初出の交渉者は初期化
				negotiationInfo.updateLast(true);
				// Accept数の更新
				negotiationInfo.updateAcceptList(sender, offeredBid);
			}
			if (action instanceof Offer) {

				if (!negotiationInfo.getOpponents().contains(sender)) {
					negotiationInfo.initOpponent(sender);
				} // 初出の交渉者は初期化
				offeredBid = ((Offer) action).getBid(); // 提案された合意案候補
				try {
					negotiationInfo.updateInfo(sender, offeredBid);
				} // 交渉情報を更新
				catch (Exception e) {
					System.out.println(
							"交渉情報の更新に失敗しました");
					e.printStackTrace();
				}

				negotiationInfo.updateOpponentIssueWeight(sender, offeredBid); // 相手の論点重みを更新
				// System.out.println("---------" + sender + "---------");
				// System.out.println(negotiationInfo.getOpponentMaxValues(sender));
				// System.out.println(negotiationInfo.getRecentWeight(sender,
				// 20));
				// System.out.println(negotiationInfo.getRecentMaxWeight(sender,
				// 20));
				// System.out.println(negotiationInfo.getSlant(sender));
				// System.out.println("--------------------------------");
			}
			if (action instanceof EndNegotiation) {
			}
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2016";
	}

}
