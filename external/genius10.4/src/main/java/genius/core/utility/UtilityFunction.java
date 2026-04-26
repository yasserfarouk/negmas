package genius.core.utility;

import java.util.ArrayList;

import genius.core.Bid;

/**
 * Implements part of {@link NonlinearUtilitySpace}. Needs more documentation.
 *
 */
public class UtilityFunction {

	private AGGREGATIONTYPE aggregationType;
	private ArrayList<UtilityFunction> utilityFunctions;
	private ArrayList<Constraint> constraints;
	private double weight;

	public UtilityFunction() {
		this.utilityFunctions = new ArrayList<UtilityFunction>();
		this.constraints = new ArrayList<Constraint>();
		this.weight = 1;
		this.aggregationType = aggregationType.SUM;
	}

	public UtilityFunction(AGGREGATIONTYPE aggregationtype) {
		this.utilityFunctions = new ArrayList<UtilityFunction>();
		this.constraints = new ArrayList<Constraint>();
		this.weight = 1;
		this.aggregationType = aggregationtype;
	}

	public UtilityFunction(AGGREGATIONTYPE aggregationtype, double weight) {
		this.utilityFunctions = new ArrayList<UtilityFunction>();
		this.constraints = new ArrayList<Constraint>();
		this.weight = weight;
		this.aggregationType = aggregationtype;
	}

	public double getWeight() {
		return weight;
	}

	public void setWeight(double weight) {
		this.weight = weight;
	}

	public ArrayList<Constraint> getConstraints() {
		return constraints;
	}

	public void setConstraints(ArrayList<Constraint> constraints) {
		this.constraints = constraints;
	}

	public void addConstraint(Constraint newConstraint) {
		if (!this.constraints.contains(newConstraint))
			this.constraints.add(newConstraint);
	}

	public void addConstraint(ArrayList<Constraint> newContraints) {

		for (Constraint newConstraint : newContraints) {
			if (!this.constraints.contains(newConstraint))
				this.constraints.add(newConstraint);
		}
	}

	public ArrayList<UtilityFunction> getUtilityFunctions() {
		return utilityFunctions;
	}

	public void setUtilityFunctions(ArrayList<UtilityFunction> utilityFunctions) {
		this.utilityFunctions = utilityFunctions;
	}

	public void addUtilityFunction(UtilityFunction newUtilityFunction) {
		if (!utilityFunctions.contains(newUtilityFunction))
			this.utilityFunctions.add(newUtilityFunction);
	}

	public AGGREGATIONTYPE getAggregationType() {
		return aggregationType;
	}

	public void setAggregationType(AGGREGATIONTYPE aggreationType) {
		this.aggregationType = aggreationType;
	}

	public double getUtility(Bid bid) {

		double finalUtility = 0.0;
		double currentUtility = 0.0;

		if (aggregationType == AGGREGATIONTYPE.MIN)
			finalUtility = Integer.MAX_VALUE;
		else if (aggregationType == AGGREGATIONTYPE.MAX)
			finalUtility = -Integer.MAX_VALUE;

		// Utility from Constraints
		for (Constraint constraint : constraints) {

			currentUtility = constraint.getUtility(bid);

			switch (aggregationType) {

			case SUM:
				finalUtility += currentUtility;
				break;
			case MAX:
				if (finalUtility < currentUtility)
					finalUtility = currentUtility;
				break;
			case MIN:
				if (finalUtility > currentUtility)
					finalUtility = currentUtility;
				break;
			}
		}

		// Utility from Utility Functions
		for (UtilityFunction function : utilityFunctions) {

			currentUtility = function.getUtility(bid) * function.getWeight();

			switch (aggregationType) {

			case SUM:
				finalUtility += currentUtility;
				break;
			case MAX:
				if (finalUtility < currentUtility)
					finalUtility = currentUtility;
				break;
			case MIN:
				if (finalUtility > currentUtility)
					finalUtility = currentUtility;
				break;

			}
		} // for

		return finalUtility;
	}

}
