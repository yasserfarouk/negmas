package genius.core.utility;

import java.util.HashMap;
import java.util.Map.Entry;

import genius.core.Bid;
import genius.core.issue.Objective;
import genius.core.issue.Value;
import genius.core.issue.ValueReal;
import genius.core.xml.SimpleElement;

/**
 * 
 * An evaluator for real-type issues.
 * 
 * @author Dmytro Tykhonov
 *
 */
@SuppressWarnings("serial")
public class EvaluatorReal implements Evaluator {
	private static final double EPSILON = 0.0001;

	private double fweight;
	private boolean fweightLock;
	double lowerBound;
	double upperBound;
	EVALFUNCTYPE type;
	HashMap<Integer, Double> fParam;

	@Override
	public boolean equals(Object obj) {
		if (!(obj instanceof EvaluatorReal))
			return false;
		EvaluatorReal eval2 = (EvaluatorReal) obj;
		if (eval2.type != type)
			return false;
		if ((Math.abs(fweight - eval2.fweight) > EPSILON)
				|| (Math.abs(lowerBound - eval2.lowerBound) > EPSILON)
				|| (Math.abs(upperBound - eval2.upperBound) > EPSILON))
			return false;

		for (Entry<Integer, Double> entry : fParam.entrySet()) {
			if (entry.getValue() - eval2.fParam.get(entry.getKey()) > EPSILON)
				return false;
		}
		return true;
	}

	@Override
	public int hashCode() {
		return super.hashCode();
	}

	public EvaluatorReal() {
		fParam = new HashMap<>();
		fweight = 0;
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
		double utility;
		double value = -1;
		try {
			value = ((ValueReal) bid.getValue(index)).getValue();
		} catch (Exception e) {
			e.printStackTrace();
		}

		utility = getEvaluation(value);
		return utility;
	}

	@Override
	public EVALUATORTYPE getType() {
		return EVALUATORTYPE.REAL;
	}

	@Override
	public String isComplete(Objective whichobj) {
		return null;
	}

	public Double getEvaluation(double pValue) {
		double utility;
		switch (this.type) {
		case LINEAR:
			utility = EVALFUNCTYPE.evalLinear(pValue, this.fParam.get(1),
					this.fParam.get(0));
			if (utility < 0)
				utility = 0;
			else if (utility > 1)
				utility = 1;
			return utility;
		case CONSTANT:
			return this.fParam.get(0);
		case TRIANGULAR:
			utility = EVALFUNCTYPE.evalTriangular(pValue, this.fParam.get(0),
					this.fParam.get(1), this.fParam.get(2));
			return utility;
		case TRIANGULAR_VARIABLE_TOP:
			utility = EVALFUNCTYPE.evalTriangularVariableTop(pValue,
					this.fParam.get(0), this.fParam.get(1), this.fParam.get(2),
					this.fParam.get(3));
			return utility;
		default:
			return -1.0;
		}

	}

	public double getValueByEvaluation(double pUtility) {
		double lValue = 0;
		switch (this.type) {
		case LINEAR:
			lValue = EVALFUNCTYPE.evalLinearRev(pUtility, this.fParam.get(1),
					this.fParam.get(0));
			if (lValue < getLowerBound())
				lValue = getLowerBound();
			else if (lValue > getUpperBound())
				lValue = getUpperBound();
			return lValue;
		case CONSTANT:
			return this.fParam.get(0);
		default:
			return -1.0;
		}

	}

	public EVALFUNCTYPE getFuncType() {
		return this.type;
	}

	public double getLowerBound() {
		return lowerBound;
	}

	public double getUpperBound() {
		return upperBound;
	}

	/**
	 * Sets the lower bound for this evaluator.
	 * 
	 * @param lf
	 *            The new lower bound.
	 */
	public void setLowerBound(double lf) {
		lowerBound = lf;
	}

	/**
	 * Sets the upper bound for this evaluator
	 * 
	 * @param ub
	 *            The new upper bound
	 */
	public void setUpperBound(double ub) {
		upperBound = ub;
	}

	/**
	 * Sets the evaluator function type.
	 * 
	 * @param ft
	 *            The new type, either <code>"linear"</code> or
	 *            <code>"constant"</code>
	 */
	public void setftype(String ft) {
		type = EVALFUNCTYPE.convertToType(ft);
	}

	/**
	 * Sets the linear parameter for the evaluation function and changes the
	 * ftype of this evaluator to "linear".
	 * 
	 * @param par1
	 *            The new linear evaluation parameter.
	 */
	public void setLinearParam(double par1) {
		setftype("linear");
		fParam.put(new Integer(1), new Double(par1));
	}

	/**
	 * 
	 * @return The linear parameter of this Evaluator, or 0 if it doesn't exist.
	 */
	public double getLinearParam() {
		try {
			return fParam.get(new Integer(1));
		} catch (Exception e) {
			System.out.println("Linear parameter does not exist");
			e.printStackTrace();
		}
		return 0;
	}

	/**
	 * Sets the constant parameter for the evaluation function and changes the
	 * ftype of this evaluator to "constant"
	 * 
	 * @param par0
	 *            The new constant evalutation parameter.
	 */
	public void setConstantParam(double par0) {
		setftype("constant");
		fParam.put(new Integer(0), new Double(par0));
	}

	/**
	 * 
	 * @return The constant parameter of this Evaluator, or 0 if it doesn't
	 *         exist.
	 */
	public double getConstantParam() {
		try {
			return fParam.get(new Integer(0));
		} catch (Exception e) {
			System.out.println("Linear parameter does not exist");
			e.printStackTrace();
		}
		return 0;
	}

	/**
	 * 
	 * @return The top parameter of this Evaluator, or 0 if it doesn't exist.
	 */
	public double getTopParam() {
		try {
			return fParam.get(new Integer(2));
		} catch (Exception e) {
			System.out.println("Top parameter does not exist");
			e.printStackTrace();
		}
		return 0;
	}

	@Override
	public void loadFromXML(SimpleElement pRoot) {
		Object[] xml_item = pRoot.getChildByTagName("range");
		this.lowerBound = Double.valueOf(
				((SimpleElement) xml_item[0]).getAttribute("lowerbound"));
		this.upperBound = Double.valueOf(
				((SimpleElement) xml_item[0]).getAttribute("upperbound"));
		Object[] xmlItems = pRoot.getChildByTagName("evaluator");
		if (xmlItems.length != 0) {
			String ftype = ((SimpleElement) xmlItems[0]).getAttribute("ftype");
			if (ftype != null)
				this.type = EVALFUNCTYPE.convertToType(ftype);
			switch (this.type) {
			case TRIANGULAR:
				this.fParam.put(0, Double.valueOf(((SimpleElement) xmlItems[0])
						.getAttribute("parameter0")));
				this.fParam.put(1, Double.valueOf(((SimpleElement) xmlItems[0])
						.getAttribute("parameter1")));
				this.fParam.put(2, Double.valueOf(((SimpleElement) xmlItems[0])
						.getAttribute("parameter2")));
				break;

			case LINEAR:
				this.fParam.put(0, Double.valueOf(((SimpleElement) xmlItems[0])
						.getAttribute("parameter0")));
				this.fParam.put(1, Double.valueOf(((SimpleElement) xmlItems[0])
						.getAttribute("parameter1")));
				break;
			case CONSTANT:
				this.fParam.put(0, Double.valueOf(((SimpleElement) xmlItems[0])
						.getAttribute("parameter0")));
				break;
			}
		}
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

	public Value getMaxValue() {

		ValueReal lValue = null;
		switch (this.type) {
		case LINEAR:
			if (getEvaluation(upperBound) > getEvaluation(lowerBound))
				lValue = new ValueReal(upperBound);
			else
				lValue = new ValueReal(lowerBound);

			break;
		default:
			// TODO: DT: use discretization to find max value
		}
		return lValue;
	}

	public Value getMinValue() {
		ValueReal lValue = null;
		switch (this.type) {
		case LINEAR:
			if (getEvaluation(upperBound) < getEvaluation(lowerBound))
				lValue = new ValueReal(upperBound);
			else
				lValue = new ValueReal(lowerBound);

			break;
		default:
			// TODO: DT: use discretization to find max value
		}
		return lValue;
	}

	public void addParam(int index, double value) {
		fParam.put(new Integer(index), new Double(value));
	}

	public void setType(EVALFUNCTYPE pType) {
		type = pType;
	}

	@Override
	public EvaluatorReal clone() {
		EvaluatorReal ed = new EvaluatorReal();
		ed.setType(type);
		ed.setWeight(fweight);
		ed.setUpperBound(upperBound);
		ed.setLowerBound(lowerBound);
		try {
			for (Entry<Integer, Double> entry : fParam.entrySet())
				ed.addParam(new Integer(entry.getKey()),
						new Double(entry.getValue()));
		} catch (Exception e) {
			System.out.println("INTERNAL ERR. clone fails");
		}

		return ed;
	}
}
