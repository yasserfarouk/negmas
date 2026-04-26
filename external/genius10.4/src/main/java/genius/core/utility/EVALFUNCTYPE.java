package genius.core.utility;

/**
 * This class specifies the possible evaluation functions.
 * In addition methods are included to calculate the utility given a value
 * and the parameters of a particular function.
 */
public enum EVALFUNCTYPE { 
	/** A constant evaluation function: all values have the same utility. */
	CONSTANT, 
	/** A linear utility function. */
	LINEAR, 
	/** A Faratin utilty function. */
	FARATIN,
	/** A triangular utility function. The top is assumed to have utility 1.0. */
	TRIANGULAR,
	/** A triangular utility function of which the top has a given utility. */
	TRIANGULAR_VARIABLE_TOP;

	/**
	 * Method which convert a string type to an object.
	 * For example the string "linear" bcomes EVALFUNCTYPE.LINEAR.
	 * 
	 * @param type of the evaluation function.
	 * @return EVALFUNCTYPE object corresponding to the given type.
	 */
	public static EVALFUNCTYPE convertToType(String type) {
		if (type.equalsIgnoreCase("linear"))
			return EVALFUNCTYPE.LINEAR;
		else if (type.equalsIgnoreCase("constant"))
			return EVALFUNCTYPE.CONSTANT;
		if (type.equalsIgnoreCase("faratin"))
			return EVALFUNCTYPE.FARATIN;
		if (type.equalsIgnoreCase("triangular"))
			return EVALFUNCTYPE.TRIANGULAR;
		else return null;
	}

	
	/**
	 * Method which given a value of a linear issue, returns the utility of the value.
	 * 
	 * @param value of the issue.
	 * @param slope of the linear evaluation function.
	 * @param offset of the linear evaluation function.
	 * @return utility of the given value.
	 */
	public static double evalLinear(double value, double slope, double offset) {
		return slope * value + offset;
	}
	
	/**
	 * Method which given the utility of an issue, converts it back to the value.
	 * @param utility of a value for the issue.
	 * @param offset of the linear evaluation function of the issue.
	 * @param slope of the linear evaluation function of the issue.
	 * @return value with the given utility.
	 */
	public static double evalLinearRev(double utility, double offset, double slope) {
		return (utility - slope) / offset;
	}

	/**
	 * Method which given the value of an issue, returns the utility of the value.
	 * @param x value of the issue.
	 * @param max
	 * @param min
	 * @param alpha
	 * @param epsilon
	 * @return utility of the value.
	 */
	public static double evalFaratin(double x, double max, double min, double alpha, double epsilon) {
		return 1 / Math.PI*Math.atan(((2*Math.abs(x-min)/(max-min)*Math.pow((x-min)/(max-min), alpha)-1)*Math.tan(Math.PI*(1/2-epsilon))))+Math.PI/2;
		
	}
	
	/**
	 * Method which given the value x, returns the utility of the value.
	 * @param x
	 * @param lowerBound
	 * @param upperBound
	 * @param top
	 * @return utiliy of the value.
	 */
	public static double evalTriangular(double x, double lowerBound, double upperBound, double top) {
		if (x<lowerBound) return 0;
		else
			if(x<top) return (x-lowerBound)/(top-lowerBound);
			else 
				if(x<upperBound) return (1-(x-top)/(upperBound -  top));
				else return 0;
	}
	
	/**
	 * Method which given the value x, returns the utility of the value.
	 * The different with the evalTriangular method, is that the top utility
	 * of the triangle is not necessary 1.0.
	 * 
	 * @param x
	 * @param lowerBound
	 * @param upperBound
	 * @param top
	 * @param topValue utility of the value with the highest utility.
	 * @return utility of the value.
	 */
	public static double evalTriangularVariableTop(double x, double lowerBound, double upperBound, double top, double topValue) {
		if(x<lowerBound) return 0;
		else
			if(x<top) return topValue*(x-lowerBound)/(top-lowerBound);
			else 
				if(x<upperBound) return topValue*(1-(x-top)/(upperBound -  top));
				else return 0;
	}
}