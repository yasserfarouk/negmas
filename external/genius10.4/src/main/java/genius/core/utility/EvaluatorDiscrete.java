package genius.core.utility;

import java.text.DecimalFormat;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import genius.core.Bid;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Objective;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.xml.SimpleElement;

/**
 * This class is used to convert the value of a discrete issue to a utility.
 * This object stores a mapping from each discrete value to a positive double,
 * the evaluation of the value.
 * 
 * When a {@link ValueDiscrete} is evaluated, there are two possibilities:
 * <ul>
 * <li>One or more utilities in the map are &gt; 1.0. Then, the evaluation of
 * the value is divided by the highest evaluation in the map.
 * <li>All utilities are &le;1.0. Then, the evaluation of the value is the same
 * as the utility stored in the map. This is useful to store absolute utilities,
 * which is needed for example in the PocketNegotiator.
 * </ul>
 * 
 * Note that most functions here are working with {@link Integer} utilities.
 * This is because we need to stay backwards compatible with older versions of
 * Genius.
 * 
 */
@SuppressWarnings("serial")
public class EvaluatorDiscrete implements Evaluator {

	private double fweight; // the weight of the evaluated Objective or Issue.
	private boolean fweightLock;
	private HashMap<ValueDiscrete, Double> fEval;

	private static final DecimalFormat f = new DecimalFormat("0.00");

	/**
	 * Creates a new discrete evaluator with weight 0 and no values.
	 */
	public EvaluatorDiscrete() {
		fEval = new HashMap<>();
		fweight = 0;
	}

	public EvaluatorDiscrete(HashMap<ValueDiscrete, Double> fEval) {
		this.fEval = fEval;
		normalizeAll();
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

	/**
	 * Loads {@link #fEval} from a SimpleElement containing something like this:
	 * {@code
	 * 			<item index="1" description=
	"Buy bags of chips and party nuts for all guests."
				value="Chips and Nuts" cost="100.0" evaluation="3">}.
	 * 
	 * Only the value and evaluation are used, the rest is ignored. NOTICE: the
	 * fWeight of this EvaluatorDiscrete is not set.
	 */
	@Override
	public void loadFromXML(SimpleElement pRoot) {
		Object[] xml_items = pRoot.getChildByTagName("item");
		int nrOfValues = xml_items.length;
		ValueDiscrete value;

		for (int j = 0; j < nrOfValues; j++) {
			value = new ValueDiscrete(
					((SimpleElement) xml_items[j]).getAttribute("value"));
			String evaluationStr = ((SimpleElement) xml_items[j])
					.getAttribute("evaluation");
			if (evaluationStr != null && !evaluationStr.equals("null")) {
				try {
					this.fEval.put(value, Double.valueOf(evaluationStr));
				} catch (Exception e) {
					System.out.println(
							"Problem reading XML file: " + e.getMessage());
				}
			}
			((SimpleElement) xml_items[j]).getAttribute("description");
		}
	}

	@Override
	public String isComplete(Objective whichobj) {
		try {
			if (!(whichobj instanceof IssueDiscrete))
				throw new Exception(
						"this discrete evaluator is associated with something of type "
								+ whichobj.getClass());
			// check that each issue value has an evaluator.
			IssueDiscrete issue = (IssueDiscrete) whichobj;
			List<ValueDiscrete> values = issue.getValues();
			for (ValueDiscrete value : values)
				if (fEval.get(value) == null)
					throw new Exception("the value " + value
							+ " has no evaluation in the objective ");
		} catch (Exception e) {
			return "Problem with objective " + whichobj.getName() + ":"
					+ e.getMessage();
		}
		return null;
	}

	@Override
	public EvaluatorDiscrete clone() {
		EvaluatorDiscrete ed = new EvaluatorDiscrete();
		ed.setWeight(fweight);
		try {
			for (ValueDiscrete val : fEval.keySet())
				ed.setEvaluationDouble(val, fEval.get(val));
		} catch (Exception e) {
			System.out.println("INTERNAL ERR. clone fails");
		}

		return ed;
	}

	@Override
	public Double getEvaluation(AdditiveUtilitySpace uspace, Bid bid,
			int issueID) {
		return normalize(fEval.get(bid.getValue(issueID)));
	}

	/**************** specific for EvaluatorDiscrete *****************/
	/**
	 * @param value
	 *            of which the evaluation is requested. ALways returns rounded
	 *            values, to be compatible with the old version of PN where
	 *            values could be only integers.
	 * 
	 * @return the non-normalized evaluation of the given value. The value is
	 *         rounded to the nearest integer. Returns 0 for values that are not
	 *         set or unknown.
	 */
	public Integer getValue(ValueDiscrete value) {
		if (fEval.containsKey(value)) {
			return (int) Math.round(fEval.get(value));
		}
		return 0;
	}

	/**
	 * @param value
	 * @return the exact double value/util of a issuevalue
	 */
	public Double getDoubleValue(ValueDiscrete value) {
		return fEval.get(value);
	}

	/**
	 * @param value
	 *            of the issue.
	 * @return normalized utility (between [0,1]) of the given value.
	 * @throws Exception
	 *             if value is null.
	 */
	public Double getEvaluation(ValueDiscrete value) throws Exception {
		return normalize(getDoubleValue(value));
	}

	/**
	 * @param bid
	 * @param ID
	 *            of the issue of which we are interested in the value
	 * @return non-normalized evaluation (positive integer) of the given value.
	 * @throws Exception
	 *             if bid or value is null.
	 */
	public Integer getEvaluationNotNormalized(Bid bid, int ID)
			throws Exception {
		return getValue(((ValueDiscrete) bid.getValue(ID)));
	}

	/**
	 * 
	 * @param value
	 *            of the issue.
	 * @return non-normalized evaluation (positive integer) of the given value.
	 *         Actually identical to {@link #getValue(ValueDiscrete)}.
	 * @throws Exception
	 *             if value is null.
	 */
	public Integer getEvaluationNotNormalized(ValueDiscrete value)
			throws Exception {
		return getValue(value);
	}

	public Double normalize(double EvalValueL) {
		double max = getEvalMax();
		if (max < 0.00001)
			return 0d;
		else
			/*
			 * this will throw if problem.
			 */
			return EvalValueL / max;
	}

	@Override
	public EVALUATORTYPE getType() {
		return EVALUATORTYPE.DISCRETE;
	}

	/**
	 * Sets the evaluation for Value <code>val</code>. If this value doesn't
	 * exist yet in this Evaluator, adds it as well.
	 * 
	 * @param val
	 *            The value to add or have its evaluation modified.
	 * @param evaluation
	 *            The new evaluation. only POSITIVE integer values acceptable as
	 *            evaluation
	 */
	public void setEvaluation(Value val, int evaluation) {
		if (evaluation < 0)
			throw new IllegalArgumentException(
					"Evaluation values have to be >= 0");
		fEval.put((ValueDiscrete) val, (double) evaluation);
	}

	/**
	 * identical to {@link #setEvaluation(Value, int)} but accepts double.
	 * 
	 * @param val
	 *            The value to add or have its evaluation modified.
	 * @param evaluation
	 *            The new evaluation. only POSITIVE integer values acceptable as
	 *            evaluation value
	 * 
	 */
	public void setEvaluationDouble(ValueDiscrete val, double evaluation) {
		if (evaluation < 0)
			throw new IllegalArgumentException(
					"Evaluation values have to be >= 0");
		fEval.put(val, evaluation);
	}

	/**
	 * wipe evaluation values.
	 */
	public void clear() {
		fEval.clear();
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

	/**
	 * Add a new possible value to the issue. Same as
	 * {@link #setEvaluation(Value, int)}. To set Double values, use
	 * {@link #setEvaluationDouble(ValueDiscrete, double)}.
	 * 
	 * @param value
	 *            to be added to the issue.
	 * @param evaluation
	 *            of the value.
	 */
	public void addEvaluation(ValueDiscrete value, Integer evaluation) {
		this.fEval.put(value, (double) evaluation);

	}

	/**
	 * @return value with the highest evaluation.
	 */
	public Value getMaxValue() {
		Iterator<Map.Entry<ValueDiscrete, Double>> it = fEval.entrySet()
				.iterator();
		Double minimum = -1e+300;// close to smallest double
		ValueDiscrete lValue = null;
		while (it.hasNext()) {
			Map.Entry<ValueDiscrete, Double> field = it.next();
			if (field.getValue() > minimum) {
				lValue = field.getKey();
				minimum = field.getValue();
			}
		}
		return lValue;
	}

	/**
	 * Sets the maximum value to 1 and scales values and weights accordingly
	 */
	public void normalizeAll() {
		double maxValue = fEval.get(getMaxValue());
		if (maxValue == 0)
			return; // avoid /0
		fEval.forEach(
				(key, value) -> fEval.put(key, fEval.get(key) / maxValue));
		setWeight(maxValue);
	}

	/**
	 * Sets the maximum value to 1, the minimum value to 0, and scales values
	 * accordingly
	 */
	public void scaleAllValuesFrom0To1() {
		double maxValue = fEval.get(getMaxValue());
		double minValue = fEval.get(getMinValue());
		if (maxValue == minValue)
			return; // avoid /0
		fEval.forEach((key, value) -> fEval.put(key,
				(fEval.get(key) - minValue) / (maxValue - minValue)));
	}

	/**
	 * @return value with the lowest evaluation.
	 */
	public Value getMinValue() {
		Iterator<Map.Entry<ValueDiscrete, Double>> it = fEval.entrySet()
				.iterator();
		Double lTmp = Double.MAX_VALUE;
		ValueDiscrete lValue = null;
		while (it.hasNext()) {
			Map.Entry<ValueDiscrete, Double> field = it.next();
			if (field.getValue() < lTmp) {
				lValue = field.getKey();
				lTmp = field.getValue();
			}

		}
		return lValue;

	}

	/**
	 * @return valid values for this issue.
	 */
	public Set<ValueDiscrete> getValues() {
		return fEval.keySet();
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result;
		result = prime * result;
		long temp;
		temp = Double.doubleToLongBits(fweight);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		result = prime * result + (fweightLock ? 1231 : 1237);
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		EvaluatorDiscrete other = (EvaluatorDiscrete) obj;
		if (fEval == null) {
			if (other.fEval != null)
				return false;
		} else if (!fEval.equals(other.fEval))
			return false;
		if (Double.doubleToLongBits(fweight) != Double
				.doubleToLongBits(other.fweight))
			return false;
		if (fweightLock != other.fweightLock)
			return false;
		return true;
	}

	@Override
	public String toString() {
		Object values[] = fEval.keySet().toArray();
		String result = "{";
		for (int i = 0; i < values.length; i++) {
			try {
				result += (values[i] + "="
						+ f.format(getDoubleValue((ValueDiscrete) values[i])));
			} catch (Exception e) {
				e.printStackTrace();
			}
			if (i != values.length - 1) {
				result += ", ";
			}
		}
		result += "}";
		return result;
	}

	public double getEvalMax() {
		return Collections.max(fEval.values());
	}

}