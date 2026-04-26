package agents.anac.y2010.AgentSmith;

import java.util.HashMap;
import java.util.List;

import genius.core.Bid;
import genius.core.issue.Issue;

/**
 * The OpponentModel. This model manages the opponents preferences, stores the
 * bids and based on the number of times the opponent proposes a specific option
 * in a bid, that option becomes more important and our agent uses this
 * information to construct its own bids.
 */
public class OpponentModel implements IOpponentModel {
	private PreferenceProfileManager fPreferenceProfileManager;
	private HashMap<Issue, IssueModel> fIssueModels;

	/**
	 * Constructor
	 */
	public OpponentModel(PreferenceProfileManager pPreferenceProfileManager,
			BidHistory pBidHistory) {
		fPreferenceProfileManager = pPreferenceProfileManager;
		fIssueModels = new HashMap<Issue, IssueModel>();

		initIssueModels();
	}

	/**
	 * For each of the issues it initializes a model which stores the opponents'
	 * preferences
	 */
	private void initIssueModels() {
		List<Issue> lIssues = fPreferenceProfileManager.getIssues();
		for (Issue lIssue : lIssues) {
			IssueModel lModel;
			lModel = new IssueModel(lIssue);
			fIssueModels.put(lIssue, lModel);
		}
	}

	/**
	 * Adds the values of each issue of a bid to the preferenceprofilemanager
	 */
	public void addBid(Bid pBid) {
		List<Issue> lIssues = fPreferenceProfileManager.getIssues();
		for (Issue lIssue : lIssues) {
			fIssueModels.get(lIssue).addValue(
					IssueModel.getBidValueByIssue(pBid, lIssue));
		}
	}

	/**
	 * Returns a hashmap with the weights for each of the issues
	 */
	public HashMap<Issue, Double> getWeights() {
		HashMap<Issue, Double> lWeights = new HashMap<Issue, Double>();
		List<Issue> lIssues = fPreferenceProfileManager.getIssues();
		for (Issue lIssue : lIssues) {
			lWeights.put(lIssue, fIssueModels.get(lIssue).getWeight());
		}

		double lTotal = 0;
		for (Issue lIssue : lIssues)
			lTotal += lWeights.get(lIssue);

		for (Issue lIssue : lIssues)
			lWeights.put(lIssue, lWeights.get(lIssue) / lTotal);

		return lWeights;
	}

	/**
	 * Returns the utility of a bid, but instead of the normal utility it is
	 * based on the weights of each issues
	 */
	public double getUtility(Bid pBid) {
		double lUtility = 0;
		HashMap<Issue, Double> lWeights = getWeights();
		List<Issue> lIssues = fPreferenceProfileManager.getIssues();

		for (Issue lIssue : lIssues) {
			double lWeight = lWeights.get(lIssue);
			double lLocalUtility = fIssueModels.get(lIssue).getUtility(pBid);

			lUtility += lWeight * lLocalUtility;
		}

		return lUtility;
	}

}
