package genius.core.utility;

import java.util.Enumeration;

import genius.core.Bid;
import genius.core.issue.Objective;

/* Created by Dimitrios Tsimpoukis
 * 
 * Subclass of AdditiveUtilitySpace that adds a perturbation in the utility of the bids.
 * Used for the introduction of Uncertainty. Only changed method is getUtility(Bid bid)
 *
 *
 */

public class AdditiveUtilitySpaceWithPerturbation extends AdditiveUtilitySpace {

	private static final long serialVersionUID = -8882680929008252676L;

	private double perturbation;

	public AdditiveUtilitySpaceWithPerturbation(AdditiveUtilitySpace uspace) {
		super(uspace);
	}

	public AdditiveUtilitySpaceWithPerturbation(AdditiveUtilitySpace uspace,
			double perturbation) {
		super(uspace);
		this.perturbation = perturbation;
	}

	@Override
	public double getUtility(Bid bid) {
		double utility = 0;

		Objective root = getDomain().getObjectivesRoot();
		Enumeration<Objective> issueEnum = root.getPreorderIssueEnumeration();
		while (issueEnum.hasMoreElements()) {
			Objective is = issueEnum.nextElement();
			Evaluator eval = getEvaluator(is);
			if (eval == null) {
				throw new IllegalArgumentException(
						"UtilitySpace does not contain evaluator for issue "
								+ is + ". ");
			}

			switch (eval.getType()) {
			case DISCRETE:
			case INTEGER:
			case REAL:
				utility += eval.getWeight()
						* getEvaluation(is.getNumber(), bid);
				break;
			case OBJECTIVE:
				// we ignore OBJECTIVE. Not clear what it is and why.
				break;
			}
		}
		double result = utility + perturbation; // Introduction of Perturbation
		if (result > 1)
			return 1;
		if (result < 0)
			return 0;
		return result;
	}

	public double getPerturbation() {
		return perturbation;
	}

	public void setPerturbation(double perturbation) {
		this.perturbation = perturbation;
	}

}
