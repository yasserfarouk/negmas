package negotiator.boaframework.opponentmodel.agentsmith;

import java.util.HashMap;
import java.util.List;

import agents.bayesianopponentmodel.OpponentModel;
import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.utility.AbstractUtilitySpace;

/**
 * The OpponentModel. This model manages the opponents preferences, stores the
 * bids and based on the number of times the opponent proposes a specific option
 * in a bid, that option becomes more important and our agent uses this
 * information to construct its own bids.
 */
public class SmithModel extends OpponentModel {
	private HashMap<Issue, IssueModel> fIssueModels;
	private AbstractUtilitySpace space;

	/**
	 * Constructor
	 */
	public SmithModel(AbstractUtilitySpace space) {
		fIssueModels = new HashMap<Issue, IssueModel>();
		fDomain = space.getDomain();
		this.space = space;
		initIssueModels();
	}

	/**
	 * For each of the issues it initializes a model which stores the opponents'
	 * preferences
	 */
	private void initIssueModels() {
		List<Issue> lIssues = space.getDomain().getIssues();
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
		List<Issue> lIssues = space.getDomain().getIssues();
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
		List<Issue> lIssues = space.getDomain().getIssues();
		for (Issue lIssue : lIssues) {
			lWeights.put(lIssue, fIssueModels.get(lIssue).getWeight());
		}

		double lTotal = 0;
		for (Issue lIssue : lIssues)
			lTotal += lWeights.get(lIssue);

		for (Issue lIssue : lIssues) {
			lWeights.put(lIssue, lWeights.get(lIssue) / lTotal);
		}

		return lWeights;
	}

	public double getWeight(int issueNr) {
		HashMap<Issue, Double> lWeights = getWeights();
		Issue foundIssue = null;
		for (Issue issue : space.getDomain().getIssues()) {
			if (issue.getNumber() == issueNr) {
				foundIssue = issue;
				break;
			}
		}
		return lWeights.get(foundIssue);
	}

	/**
	 * Returns the utility of a bid, but instead of the normal utility it is
	 * based on the weights of each issues
	 */
	public double getNormalizedUtility(Bid pBid) {
		double lUtility = 0;
		HashMap<Issue, Double> lWeights = getWeights();
		List<Issue> lIssues = space.getDomain().getIssues();

		for (Issue lIssue : lIssues) {
			double lWeight = lWeights.get(lIssue);
			double lLocalUtility = fIssueModels.get(lIssue).getUtility(pBid);
			lUtility += lWeight * lLocalUtility;
		}
		return lUtility;
	}
}