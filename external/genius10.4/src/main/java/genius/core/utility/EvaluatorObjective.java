package genius.core.utility;

import genius.core.Bid;
import genius.core.issue.Objective;
import genius.core.xml.SimpleElement;

/**
 * Evaulator for an objective. In the current implementation it makes no sense
 * to createFrom this type of object.
 */
@SuppressWarnings("serial")
public class EvaluatorObjective implements Evaluator {

	private double fweight = 0;
	private boolean fweightLock = false;
	private boolean hasWeightP = false;

	/**
	 * Creates a new evaluator for an objective with a zero weight.
	 */
	public EvaluatorObjective() {
	}

	/**
	 * Copies the data from the given EvaluatorObjective.
	 * 
	 * @param e
	 *            other EvaluatorObjective
	 */
	public EvaluatorObjective(EvaluatorObjective e) {
		fweight = e.getWeight();
		fweightLock = e.weightLocked();
		hasWeightP = e.getHasWeight();
	}

	@Override
	public EvaluatorObjective clone() {
		return new EvaluatorObjective(this);
	}

	@Override
	public double getWeight() {
		return fweight;
	}

	@Override
	public void setWeight(double wt) {
		fweight = wt;
	}

	@Override
	public void lockWeight() {
		fweightLock = true;
	}

	@Override
	public void unlockWeight() {
		fweightLock = false;
	}

	@Override
	public boolean weightLocked() {
		return fweightLock;
	}

	@Override
	public Double getEvaluation(AdditiveUtilitySpace uspace, Bid bid,
			int index) {
		return 0.0;
	}

	@Override
	public EVALUATORTYPE getType() {
		return EVALUATORTYPE.OBJECTIVE;
	}

	@Override
	public void loadFromXML(SimpleElement pRoot) {
		// do nothing, we have no issues to load.
	}

	/**
	 * @param doesHaveWeight
	 *            signals that this objective has a weight.
	 */
	public void setHasWeight(boolean doesHaveWeight) {
		hasWeightP = doesHaveWeight;
	}

	/**
	 * @return true if objective has a weight.
	 */
	public boolean getHasWeight() {
		return hasWeightP;
	}

	/**
	 * Sets weights and evaluator properties for the object in SimpleElement
	 * representation that is passed to it.
	 * 
	 * @param evalObj
	 *            The object of which to set the evaluation properties.
	 * @return The modified simpleElement with all evaluator properties set.
	 */
	public SimpleElement setXML(SimpleElement evalObj) {

		return evalObj;
	}

	@Override
	public String isComplete(Objective whichobj) {
		return "Internal error: isComplete should be checked only with Issues, not with Objectives";
	}
}