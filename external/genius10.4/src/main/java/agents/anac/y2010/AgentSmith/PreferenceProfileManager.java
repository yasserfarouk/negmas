package agents.anac.y2010.AgentSmith;

import java.util.List;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.Evaluator;

public class PreferenceProfileManager {

	protected AdditiveUtilitySpace fUtilitySpace;
	protected BidHistory fBidHistory;
	private IOpponentModel fOpponentModel;

	public PreferenceProfileManager(BidHistory pHist,
			AdditiveUtilitySpace pUtilitySpace) {
		this.fBidHistory = pHist;
		this.fUtilitySpace = pUtilitySpace;
		this.fOpponentModel = new OpponentModel(this, fBidHistory);

	}

	/**
	 * Retrieves the utility of an opponent, this can be based on the
	 * bidhistory.
	 * 
	 * @param b
	 * @return
	 */
	public double getOpponentUtility(Bid b) {
		return fOpponentModel.getUtility(b);
	}

	/*
	 * add a bid to the opponentmodel
	 */
	public void addBid(Bid b) {
		this.fOpponentModel.addBid(b);
	}

	/**
	 * Returns the utility for a bid for me. Will be just a call to
	 * utilityspace.
	 * 
	 * @param b
	 * @return
	 */
	public double getMyUtility(Bid b) {
		try {
			return this.fUtilitySpace.getUtility(b);
		} catch (Exception e) {
			return 0;
		}
	}

	/*
	 * returns the Evaluator of an issue
	 */
	public Evaluator getMyEvaluator(int issueID) {
		return this.fUtilitySpace.getEvaluator(issueID);
	}

	/*
	 * returns the domain
	 */
	public Domain getDomain() {
		return this.fUtilitySpace.getDomain();
	}

	/*
	 * returns the list of issues in the domain
	 */
	public List<Issue> getIssues() {
		return this.fUtilitySpace.getDomain().getIssues();
	}

	/*
	 * returns the opponentmodel
	 */
	public IOpponentModel getOpponentModel() {
		return fOpponentModel;
	}

	/*
	 * sets the opponent model
	 */
	public void setOpponentModel(IOpponentModel fOpponentModel) {
		this.fOpponentModel = fOpponentModel;
	}
}
