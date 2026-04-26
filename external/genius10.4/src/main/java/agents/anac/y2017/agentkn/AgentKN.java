package agents.anac.y2017.agentkn;

import java.util.List;

import java.util.HashMap;
import java.util.Map;

import agents.anac.y2017.agentkn.etc.bidSearch;
import agents.anac.y2017.agentkn.etc.negotiationInfo;
import agents.anac.y2017.agentkn.etc.negotiationStrategy;
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Inform;
import genius.core.actions.Offer;
import genius.core.list.Tuple;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.PersistentDataType;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.AdditiveUtilitySpace;

public class AgentKN extends AbstractNegotiationParty {
	private TimeLineInfo timelineInfo; // タイムライン
	private AbstractUtilitySpace utilitySpace;

	private Bid mLastReceivedBid = null;
	private int nrChosenActions = 0; // number of times chosenAction was called.
	private StandardInfoList history;
	private negotiationStrategy mStrategy;
	private negotiationInfo mNegotiationInfo;
	private bidSearch mBidSerch;

	private boolean isPrinting = false;

	private Bid mOfferedBid = null;

	@Override
	public void init(NegotiationInfo aInfo) {
		super.init(aInfo);

		if (isPrinting) {
			System.out.println("*** AgentKN ***");
		}
		System.out.println("Discount Factor is " + getUtilitySpace().getDiscountFactor());
		System.out.println("Reservation Value is " + getUtilitySpace().getReservationValueUndiscounted());

		mNegotiationInfo = new negotiationInfo((AdditiveUtilitySpace) getUtilitySpace(), isPrinting);
		try {
			mBidSerch = new bidSearch((AdditiveUtilitySpace) getUtilitySpace(), mNegotiationInfo, isPrinting);
		} catch (Exception e) {
			throw new RuntimeException("init failed: " + e);
		}

		// need standard data 過去情報取得
		if (getData().getPersistentDataType() != PersistentDataType.STANDARD) {
			throw new IllegalStateException("need standard persistent data");
		}
		history = (StandardInfoList) getData().get();

		if (!history.isEmpty()) {
			// example of using the history. Compute for each party the maximum
			// utility of the bids in last session.
			Map<String, Double> maxutils = new HashMap<String, Double>();
			StandardInfo lastinfo = history.get(history.size() - 1);
			for (Tuple<String, Double> offered : lastinfo.getUtilities()) {
				String party = offered.get1();
				Double util = offered.get2();
				maxutils.put(party, maxutils.containsKey(party) ? Math.max(maxutils.get(party), util) : util);
			}
		}

		mStrategy = new negotiationStrategy((AdditiveUtilitySpace) getUtilitySpace(), mNegotiationInfo,
				isPrinting);
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		double time = getTimeLine().getTime();
		return (Action) (validActions.contains(Accept.class) && mStrategy.selectAccept(this.mOfferedBid, time)
				? new Accept(this.getPartyId(), this.mOfferedBid)
				: (mStrategy.selectEndNegotiation(time) ? new EndNegotiation(this.getPartyId()) : this.OfferAction()));
	}

	private Action OfferAction() {
		Bid offerBid = mBidSerch.getBid(this.generateRandomBid(), mStrategy.getThreshold(getTimeLine().getTime()));
		mNegotiationInfo.updateMyBidHistory(offerBid);
		return new Offer(this.getPartyId(), offerBid);
	}

	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);

		if (action != null) {
			if (action instanceof Inform && ((Inform) action).getName() == "NumberOfAgents"
					&& ((Inform) action).getValue() instanceof Integer) {
				Integer e = (Integer) ((Inform) action).getValue();
				mNegotiationInfo.updateOpponentsNum(e.intValue());
				if (isPrinting) {
					System.out.println("NumberofNegotiator:" + mNegotiationInfo.getNegotiatorNum());
				}
			} else if (action instanceof Accept) {
				if (!mNegotiationInfo.getOpponents().contains(sender)) {
					mNegotiationInfo.initOpponent(sender);
				}
			} else if (action instanceof Offer) {
				if (!mNegotiationInfo.getOpponents().contains(sender)) {
					mNegotiationInfo.initOpponent(sender);
				}

				mOfferedBid = ((Offer) action).getBid();

				try {
					mNegotiationInfo.updateInfo(sender, this.mOfferedBid);
					mNegotiationInfo.updateOfferedValueNum(sender, this.mOfferedBid, getTimeLine().getTime());
				} catch (Exception e) {
					System.out.println("交渉情報の更新に失敗しました");
					e.printStackTrace();
				}
			} else {
				boolean var10000 = action instanceof EndNegotiation;
			}
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2017 accept Nth offer";
	}

}
