package negotiator.boaframework.opponentmodel.agentsmithv2;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.misc.ScoreKeeper;

/**
 * Class which keeps track of the issues for an optimized version of the Smith Frequency Model.
 * 
 * @author Mark Hendrikx
 */
public class IssueModel {
	
	/** Object to keep track of how many times each value of the issue has been offered */
	private ScoreKeeper<Value> keeper;
	/** The index of the issue in the domain's XML file */
	private int issueNr;
	
	/**
	 * @param issue of which values is kept track in this class
	 */
	public IssueModel(Issue issue) {
		keeper = new ScoreKeeper<Value>();
		this.issueNr = issue.getNumber();
	}
	
	/**
	 * Method which scores the value which was offered.
	 */
	public void addValue(Bid pBid) {
		keeper.score(getBidValueByIssue(pBid, issueNr));
	}
	
	/**
	 * The utility of a bid, which can be real, integer or discrete
	 */
	public double getUtility(Bid pBid) {
		double lUtility = keeper.getRelativeScore(getBidValueByIssue(pBid, issueNr));
		return lUtility;
	}
	
	/**
	 * Get's the importance of this issues utility
	 */
	public double getWeight() {
		return ((double) keeper.getMaxValue() / (double) keeper.getTotal());
	}
	
	/**
	 * returns the value of an issue in a bid
	 */
	public static Value getBidValueByIssue(Bid pBid, int issueNumber) {
		Value lValue = null;
		try {
			lValue = pBid.getValue(issueNumber); 
		} catch(Exception e) { }
		
		return lValue;
		
	}
}
