package agents.anac.y2016.syagent;

import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.DefaultAction;
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

public class SYAgent extends AbstractNegotiationParty {
	private TimeLineInfo timeLineInfo; // タイムライン
	private AbstractUtilitySpace utilitySpace;
	private negotiationInfo negotiationInfo; // 交渉情報
	private bidSearch bidSearch; // 合意案候補の探索
	private negotiationStrategy negotiationStrategy; // 交渉戦略
	private Bid offeredBid = null; // 最近提案された合意案候補
	private boolean isPrinting = false; // 本番用

	// private boolean isPrinting = true; // デバッグ用

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		if (isPrinting)
			System.out.println("*** SYAgent2016 v1.0 ***");

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
		negotiationInfo.countRound();
		// double discountedUtility =
		// utilitySpace.discount(utilitySpace.getUtility(offeredBid), time);
		// discount の計算方法 = (Bid の Utility) * Math.pow(df,time);

		// Acceptの判定
		if (validActions.contains(Accept.class)
				&& negotiationStrategy.selectAccept(offeredBid, time)) {
			return new Accept(getPartyId(), offeredBid);
		}

		// EndNegotiationの判定
		if (negotiationStrategy.selectEndNegotiation(time)) {
			return new EndNegotiation(getPartyId());
		}

		// Accept でも EndNegotiation でない時
		// 他のプレイヤーに新たなBidをOffer
		Bid offerBid = bidSearch.getBid(
				utilitySpace.getDomain().getRandomBid(null),
				negotiationStrategy.getThreshold(timeLineInfo.getTime()));
		// 自分の提案履歴を更新
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
	// 自分以外のエージェントのActionを受信するメソッド
	@Override
	public void receiveMessage(AgentID sender, Action action) {
		// sender(送り主)のActionを受信
		super.receiveMessage(sender, action);

		// Action がある時
		if (action != null) {
			// 交渉者数 の更新
			if (action instanceof Inform
					&& ((Inform) action).getName() == "NumberOfAgents"
					&& ((Inform) action).getValue() instanceof Integer) {
				Integer opponentsNum = (Integer) ((Inform) action).getValue();
				negotiationInfo.updateOpponentsNum(opponentsNum);
				if (isPrinting) {
					System.out.println("Number of Negotiator: "
							+ negotiationInfo.getNegotiatorNum());
				}
			}
			// Accept の時
			if (action instanceof Accept) {
				if (!negotiationInfo.getOpponents().contains(sender)) {
					negotiationInfo.initOpponent(sender);
				} // 初出の交渉者は初期化
				negotiationInfo.updateAcceptedBid(sender,
						DefaultAction.getBidFromAction(action)); // accept
																	// された
																	// bid
																	// を記録
			}
			// Offer の時
			if (action instanceof Offer) {
				if (!negotiationInfo.getOpponents().contains(sender)) {
					negotiationInfo.initOpponent(sender);
				} // 初出の交渉者は初期化
				offeredBid = ((Offer) action).getBid(); // senderによって提案された合意案候補
				try {
					negotiationInfo.updateInfo(sender, offeredBid);

				} // 交渉情報を更新
				catch (Exception e) {
					System.out.println(
							"交渉情報の更新に失敗しました");
					e.printStackTrace();
				}
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
