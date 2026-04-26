package agents.anac.y2010.AgentSmith;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.SupportedNegotiationSetting;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * ANAC2010 competitor Agent Smith.
 */
public class AgentSmith extends Agent {
	private PreferenceProfileManager fPreferenceProfile;
	private ABidStrategy fBidStrategy;
	private BidHistory fBidHistory;

	// Some possible requirements for the bayes agent

	// private ArrayList<Bid> ourPreviousBids;

	private boolean firstRound = true;
	private double sMargin = 0.9;

	/**
	 * The version of this agent
	 * 
	 * @return
	 */
	@Override
	public String getVersion() {
		return "3"; //
	}

	@Override
	public String getName() {
		return "Agent Smith";
	}

	/**
	 * The agent will be initialized here.
	 */
	@Override
	public void init() {

		fBidHistory = new BidHistory();
		fPreferenceProfile = new PreferenceProfileManager(fBidHistory,
				(AdditiveUtilitySpace) this.utilitySpace);
		fBidStrategy = new SmithBidStrategy(fBidHistory,
				(AdditiveUtilitySpace) this.utilitySpace, fPreferenceProfile,
				getAgentID());
	}

	/**
	 * This is called when a action was done, by the other agent.
	 */
	@Override
	public void ReceiveMessage(Action pAction) {
		/*
		 * Leaving it for now, to prevent it from breaking but it can be
		 * replaced with:
		 */

		if (pAction == null)
			return;

		if (pAction instanceof Offer) {
			Bid lBid = ((Offer) pAction).getBid();

			fBidHistory.addOpponentBid(lBid);
			fPreferenceProfile.addBid(lBid);
		}

	}

	/**
	 * When we take turn, this function is invoked.
	 */
	@Override
	public Action chooseAction() {
		Bid currentBid = null;
		Action currentAction = null;
		try {
			if (fBidHistory.getOpponentLastBid() != null && utilitySpace
					.getUtility(fBidHistory.getOpponentLastBid()) > sMargin) {
				// bid higher then margin
				currentAction = new Accept(getAgentID(),
						fBidHistory.getOpponentLastBid());
			} else {
				// start with the highest bid
				if (firstRound && (fBidHistory.getMyLastBid() == null)) {
					firstRound = !firstRound;
					currentBid = getInitialBid();
					currentAction = new Offer(getAgentID(), currentBid);
					Bid lBid = ((Offer) currentAction).getBid();
					fBidHistory.addMyBid(lBid);

				} else {

					// the utility of the opponents' bid is higher then ours ->
					// accept!
					double utilOpponent = this.utilitySpace
							.getUtility(fBidHistory.getOpponentLastBid());
					double utilOur = this.utilitySpace
							.getUtility(fBidHistory.getMyLastBid());
					if (utilOpponent >= utilOur) {
						currentAction = new Accept(getAgentID(),
								fBidHistory.getOpponentLastBid());
					} else {
						// should be rewritten into something else...
						currentAction = fBidStrategy
								.getNextAction(timeline.getTime());
						if (currentAction instanceof Offer) {
							Bid lBid = ((Offer) currentAction).getBid();
							fBidHistory.addMyBid(lBid);
						}
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		return currentAction;
	}

	/*
	 * Get the initial bid, with the highest utility
	 */
	private Bid getInitialBid() throws Exception {
		return this.utilitySpace.getMaxUtilityBid();
	}

	@Override
	public SupportedNegotiationSetting getSupportedNegotiationSetting() {
		return SupportedNegotiationSetting.getLinearUtilitySpaceInstance();
	}

	@Override
	public String getDescription() {
		return "ANAC2010";
	}

}
