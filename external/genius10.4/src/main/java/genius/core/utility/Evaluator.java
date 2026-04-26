package genius.core.utility;

import java.io.Serializable;

import genius.core.Bid;
import genius.core.issue.Objective;
import genius.core.xml.SimpleElement;

/**
 * Evaluator is an object that translates discrete values into an evaluation
 * value. The UtilitySpace attaches it to an issue.
 * 
 * @author Dmytro
 */
public interface Evaluator extends Serializable {

	// Interface methods
	/**
	 * @return the weight associated with this, a value in [0,1]
	 */
	public double getWeight();

	/**
	 * Sets the weigth with which an Objective or Issue is evaluated.
	 * 
	 * @param wt
	 *            The new weight, a value in [0,1].
	 */
	public void setWeight(double wt);

	/**
	 * lockWeight is a flag affecting the behaviour of the normalize function in
	 * the utility space. This is used to change behaviour when users drag
	 * sliders
	 */
	public void lockWeight();

	/**
	 * Method to unlock a weight. A weight must be unlocked to modify it.
	 */
	public void unlockWeight();

	/**
	 * @return true if weight is locked.
	 */
	public boolean weightLocked();

	/**
	 * This method returns the utility of the value of an issue. Note that the
	 * value is not multiplied by the issue weight, and is therefore
	 * non-normalized.
	 * 
	 * @param uspace
	 *            preference profile
	 * @param bid
	 *            in which the value is contained.
	 * @param index
	 *            unique ID of the issue in the bid for which we want an
	 *            evaluation.
	 * @return utility of the value for an issue, not normalized by the issue
	 *         weight.
	 */
	public Double getEvaluation(AdditiveUtilitySpace uspace, Bid bid,
			int index);

	/**
	 * @return type of evaluation function, for example EVALUATORTYPE.LINEAR.
	 */
	public EVALUATORTYPE getType();

	/**
	 * Load the evaluator from an XML file
	 * 
	 * @param pRoot
	 */
	public void loadFromXML(SimpleElement pRoot);

	/**
	 * Check whether the evaluator has enough information to make an evaluation.
	 * 
	 * @param whichObjective
	 *            is the objective/issue to which this evaluator is attached.
	 * @return String describing lacking component, or null if the evaluator is
	 *         complete.
	 */
	public String isComplete(Objective whichObjective);

	/**
	 * @return clone of the current object.
	 */
	public Evaluator clone();
}
