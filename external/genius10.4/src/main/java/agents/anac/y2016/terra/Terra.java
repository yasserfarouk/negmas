package agents.anac.y2016.terra;

import java.util.List;

import agents.anac.y2016.terra.etc.bidSearch;
import agents.anac.y2016.terra.etc.negotiationInfo;
import agents.anac.y2016.terra.etc.negotiationStrategy;
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
public class Terra extends AbstractNegotiationParty {

	private TimeLineInfo timeLineInfo; // タイムライン
	private AbstractUtilitySpace utilitySpace; // 効用空間
	private int opponentsNum;

	private negotiationInfo negotiationInfo; //
	private negotiationStrategy negotiationStrategy;
	private bidSearch bidSearch;

	private Bid offeredBid = null; // 最近提案された合意案候補

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);

		// System.out.println("agent Terra");
		// if (isPrinting) System.out.println("*** SampleAgent2016 v1.0 ***");

		this.timeLineInfo = timeline;
		this.utilitySpace = getUtilitySpace();

		negotiationInfo = new negotiationInfo(utilitySpace);
		negotiationStrategy = new negotiationStrategy(utilitySpace,
				negotiationInfo);
		try {
			bidSearch = new bidSearch(utilitySpace, negotiationInfo);
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
		// t:現在の時刻
		double time = timeLineInfo.getTime();

		// EndNegotiationの判定
		if (negotiationStrategy.selectEndNegotiation(time)) {
			return new EndNegotiation(getPartyId());
		} // 任意のタイミングで交渉放棄を宣言可能

		// Acceptの判定
		if (validActions.contains(Accept.class)
				&& negotiationStrategy.selectAccept(offeredBid, time)) {
			return new Accept(getPartyId(), offeredBid); // offeredBidをAccept
		}

		// 他のプレイヤーに新たなBidをOffer
		Bid offerBid = bidSearch.getBid(
				utilitySpace.getDomain().getRandomBid(null),
				negotiationStrategy.getThreshold(timeLineInfo.getTime()));

		negotiationInfo.addMyBidHistory(offerBid);
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

		// HashMap<Object,HashMap<Issue, ArrayList<Value>>> value =
		// negotiationInfo.getOfferedValue();
		// System.out.println(value);

		// 受信したアクションの種類によって行動
		if (action != null) {
			if (action instanceof Inform
					&& ((Inform) action).getName() == "NumberOfAgents"
					&& ((Inform) action).getValue() instanceof Integer) {
				opponentsNum = (Integer) ((Inform) action).getValue();
			}

			if (sender != null)
				negotiationInfo.updateOpponents(sender, opponentsNum);

			if (action instanceof Accept) { /* Acceptの処理 */
				negotiationInfo.addAgreedList(sender, offeredBid);
			}
			if (action instanceof Offer) { /* Offerの処理 */
				negotiationInfo.addDisagreedList(sender, offeredBid);
				offeredBid = ((Offer) action).getBid(); // 提案された合意案候補を記録
				negotiationInfo.addOfferedList(sender, offeredBid);
			}
			if (action instanceof EndNegotiation) { /*
													 * EndNegotiationの処理
													 */
			}
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2016";
	}

}
