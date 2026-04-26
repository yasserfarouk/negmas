package agents.anac.y2015.agenth;

import java.util.HashMap;
import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * This is your negotiation party.
 */
public class AgentH extends AbstractNegotiationParty {

	/** 現在の bid */
	protected Bid mCurrentBid;
	/** 現在の bid での効用値 */
	protected double mCurrentUtility;
	/** estimatorMap */
	protected HashMap<Object, BidStrategy> mEstimatorMap;
	/** bid 履歴 */
	protected BidHistory mBidHistory;
	/** ヘルパー */
	protected BidHelper mBidHelper;

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
		// Make sure that this constructor calls it's parent.
		super.init(info);

		mEstimatorMap = new HashMap<Object, BidStrategy>();
		mBidHistory = new BidHistory((AdditiveUtilitySpace) getUtilitySpace());
		try {
			mBidHelper = new BidHelper(this);
		} catch (Exception e) {
			throw new RuntimeException("init failed:" + e, e);
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
		// 経過時間 [0,1] を取得
		final double time = getTimeLine().getTime();

		// トップバッターなら適当に bid // FIXME
		if (!validActions.contains(Accept.class)) {
			final Bid bid = generateRandomBid();
			mBidHistory.offer(this, bid, getUtility(bid));
			mCurrentBid = new Bid(bid);
			return new Offer(getPartyId(), bid);
		}

		final double v = mCurrentUtility * time;
		// System.out.println("OreoreAgent#chooseAction(): v="+v);

		// 時間と共に
		if (v < 0.45) {
			final Bid bid = generateNextBid(time);
			mBidHistory.offer(this, bid, getUtility(bid));
			mCurrentBid = new Bid(bid);
			return new Offer(getPartyId(), bid);
		} else {
			return new Accept(getPartyId(),
					((ActionWithBid) getLastReceivedAction()).getBid());
		}
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
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		// Here you can listen to other parties' messages

		// 現在の bid を更新
		if (action instanceof Offer) {
			mCurrentBid = ((Offer) action).getBid();
			mCurrentUtility = getUtility(mCurrentBid);

			// 記録
			mBidHistory.offer(sender, mCurrentBid, mCurrentUtility);
		} else if (action instanceof Accept) {
			// 記録
			mBidHistory.accept(sender, mCurrentBid);
		}
	}

	/**
	 * 次に自分が出す bid を生成する
	 * 
	 * @return
	 */
	protected Bid generateNextBid(double time) {
		Bid bid;

		bid = mBidHelper.generateFromRelativeUtilitySearch(1.0 * time);
		if (bid == null) {
			bid = mBidHelper.generateFromHistory(1.0 * time);
		}
		if (bid == null) {
			bid = generateRandomBid();
		}

		return bid;
	}

	@Override
	public String getDescription() {
		return "ANAC2015";
	}

}
