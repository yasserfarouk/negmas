package agents.anac.y2015.AgentW;

import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Inform;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * This is your negotiation party.
 */
public class AgentW extends AbstractNegotiationParty {
	private negotiatingInfo negotiatingInfo; // 交渉情報
	private bidSearch bidSearch; // Bid探索
	private strategy strategy; // 交渉戦略
	public Bid offeredBid = null; // 提案された合意案候補

	// デバッグ用
	public static boolean isPrinting = false; // メッセージを表示する

	/**
	 * Please keep this constructor. This is called by genius.
	 *
	 * @param utilitySpace
	 *            Your utility space.
	 * @param deadlines
	 *            The deadlines set for this negotiation.
	 * @param timeline
	 *            Value counting from 0 (start) to 1 (end).
	 * @param randomSeed
	 *            If you use any randomization, use this seed for it.
	 * @throws Exception
	 */
	@Override
	public void init(NegotiationInfo info) {
		super.init(info);

		if (isPrinting) {
			System.out.println("*** Agent W ***");
		}

		negotiatingInfo = new negotiatingInfo(
				(AdditiveUtilitySpace) utilitySpace);
		try {
			bidSearch = new bidSearch((AdditiveUtilitySpace) utilitySpace,
					negotiatingInfo);
		} catch (Exception e) {
			throw new RuntimeException("init failed:" + e, e);
		}
		strategy = new strategy((AdditiveUtilitySpace) utilitySpace,
				negotiatingInfo);
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
	@SuppressWarnings("rawtypes")
	@Override
	// Actionの選択
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		double time = timeline.getTime(); // 現在の交渉時刻を取得

		// Accept
		if (validActions.contains(Accept.class)
				&& strategy.selectAccept(offeredBid, time)) {
			return new Accept(getPartyId(), offeredBid);
		}

		// EndNegotiation
		if (strategy.selectEndNegotiation(time)) {
			return new EndNegotiation(getPartyId());
		}

		// Offer
		return OfferAction();
	}

	public Action OfferAction() {
		Bid seedBid = generateRandomBid();
		if (offeredBid != null) {
			seedBid = offeredBid;
			// System.out.println(" "+seedBid);
		}

		Bid offerBid = bidSearch.getBid(seedBid,
				strategy.getThreshold(timeline.getTime()));
		negotiatingInfo.updateMyBidHistory(offerBid);
		return new Offer(getPartyId(), offerBid);
	}

	/**
	 * All offers proposed by the other parties will be received as a message.
	 * You can use this information to your advantage, for example to predict
	 * their utility.
	 *
	 * @param sender
	 *            The party that did the action.
	 * @param action
	 *            The action that party did.
	 */
	@Override
	// 自身以外の交渉参加者のActionを受信
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		// Here you can listen to other parties' messages
		if (isPrinting) {
			System.out.println("Sender:" + sender + ", Action:" + action);
		}

		if (action != null) {
			if (action instanceof Inform
					&& ((Inform) action).getName() == "NumberOfAgents"
					&& ((Inform) action).getValue() instanceof Integer) {
				Integer opponentsNum = (Integer) ((Inform) action).getValue();
				negotiatingInfo.updateOpponentsNum(opponentsNum);
				if (isPrinting) {
					System.out.println("NumberofNegotiator:"
							+ negotiatingInfo.getNegotiatorNum());
				}
			} else if (action instanceof Accept) {
				if (!negotiatingInfo.getOpponents().contains(sender)) {
					negotiatingInfo.initOpponent(sender);
				} // 初出の交渉者は初期化
			} else if (action instanceof Offer) {
				if (!negotiatingInfo.getOpponents().contains(sender)) {
					negotiatingInfo.initOpponent(sender);
				} // 初出の交渉者は初期化
				offeredBid = ((Offer) action).getBid(); // 提案された合意案候補
				try {
					negotiatingInfo.updateInfo(sender, offeredBid);
				} // 交渉情報を更新
				catch (Exception e) {
					System.out.println(
							"交渉情報の更新に失敗しました");
					e.printStackTrace();
				}
			} else if (action instanceof EndNegotiation) {
			}
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2015";
	}

}
