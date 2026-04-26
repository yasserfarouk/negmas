package agents.anac.y2011.Gahboninho;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.utility.AdditiveUtilitySpace;

// The agent is similar to a bully.
// on the first few bids it steadily goes down to 0.9 of utility to make sure that if the opponent is trying to 
// profile me, he could do it more easily.
// after that, it goes  totally selfish and almost giving up no points.
// more over, the nicer the opponent is (Noise serves as a "niceness" estimation) the more selfish this agent gets.
// only at the last few seconds the agent panics and gives up his utility.
// I believe this is an excellent strategy, but sadly we did not have enough manpower and time
// to calibrate it and make it shine :(
// an opponent-model would have helped us improving the result on panic stages, and the noise(niceness) calculation is too rough.

public class Gahboninho extends Agent {
	final int PlayerCount = 8; // if player count is 10, then we
	// may give 10 points to opponent in order to give 1 point to ourselves

	boolean WereBidsFiltered = false;
	OpponnentModel OM;
	IssueManager IM;

	@Override
	public void init() {
		super.init();
		OM = new OpponnentModel((AdditiveUtilitySpace) utilitySpace, timeline);
		IM = new IssueManager((AdditiveUtilitySpace) utilitySpace, timeline,
				OM);
		IM.Noise *= IM.GetDiscountFactor();
	}

	@Override
	public String getName() {
		return "Gahboninho V3";
	}

	Bid previousBid = null;
	Bid OpponentBid = null;

	@Override
	public void ReceiveMessage(Action opponentAction) {
		this.previousBid = this.OpponentBid;

		if (opponentAction instanceof Offer) {
			OpponentBid = ((Offer) opponentAction).getBid();
			if (this.previousBid != null) {
				try {
					this.IM.ProcessOpponentBid(this.OpponentBid);
					OM.UpdateImportance(OpponentBid);
				} catch (Exception e) {
					// Too bad
				}
			} else {
				try {
					this.IM.learnBids(this.OpponentBid);

				} // learn from the first opp. bid
				catch (Exception e) {

				}
			}
		}
	}

	int RoundCount = 0;

	int FirstActions = 40;
	int TotalFirstActions = 40;

	@Override
	public Action chooseAction() {
		try {
			// on the first few rounds don't get tempted so fast
			if (FirstActions > 0 && OpponentBid != null
					&& utilitySpace.getUtility(OpponentBid) > 0.95)
				return new Accept(this.getAgentID(), OpponentBid);

			double threshold = IM.GetMinimumUtilityToAccept();
			if (OpponentBid != null
					&& utilitySpace.getUtility(OpponentBid) >= threshold) {
				return new Accept(this.getAgentID(), OpponentBid);
			}

			++RoundCount;
			if (WereBidsFiltered == false && (timeline
					.getTime() > IM.GetDiscountFactor() * 0.9
					|| timeline.getTime()
							+ 3 * IM.BidsCreationTime > 1)) /*
															 * we must filter to
															 * make last bids
															 * efficient
															 */
			{
				WereBidsFiltered = true;

				int DesiredBidcount = (int) (RoundCount
						* (1 - timeline.getTime()));

				if (IM.Bids.size() > 200) // if we won't filter many bids
											// anyway, don't take the chance of
											// filtering
				{
					IM.Bids = OM.FilterBids(IM.Bids, DesiredBidcount);
				}
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		// on the first time we act offer max bid
		if (this.previousBid == null) {
			try {
				IM.AddMyBidToStatistics(this.IM.getMaxBid());
			} catch (Exception e2) {
			}
			return new Offer(this.getAgentID(), this.IM.getMaxBid());
		}

		Bid myBid;
		if (FirstActions >= 0 && timeline.getTime() < 0.15) {
			// on first few bids let the opponent learn some more about our
			// preferences

			double utilDecrease = (1 - 0.925) / TotalFirstActions;

			myBid = IM.GenerateBidWithAtleastUtilityOf(
					0.925 + utilDecrease * FirstActions);
			--FirstActions;
		} else {
			double threshold = IM.GetNextRecommendedOfferUtility();
			myBid = IM.GenerateBidWithAtleastUtilityOf(threshold);

			if (IM.InFrenzy == true)
				myBid = IM.BestEverOpponentBid;
		}

		try {
			IM.AddMyBidToStatistics(myBid);
		} catch (Exception e2) {
		}

		return new Offer(this.getAgentID(), myBid);
	}

	@Override
	public String getDescription() {
		return "ANAC2011 compatible with non-linear utility spaces";
	}

}