package genius.core.utility;

/**
 * Enum specifying the possible evaluation types.
 * This class can be used to convert the name of an evaluation type
 * to an enum, for example "discrete" to EVALUATORTYPE.DISCRETE. 
 */
public enum EVALUATORTYPE {	DISCRETE, INTEGER, REAL, OBJECTIVE;

	public static EVALUATORTYPE convertToType(String typeString) {
        if (typeString.equalsIgnoreCase("integer"))
        	return EVALUATORTYPE.INTEGER;
        else if (typeString.equalsIgnoreCase("real"))
        	return EVALUATORTYPE.REAL;
        else if (typeString.equalsIgnoreCase("discrete"))
        	return EVALUATORTYPE.DISCRETE;
        else if (typeString.equalsIgnoreCase("objective"))
        	return EVALUATORTYPE.OBJECTIVE;
        else {
        	// Type specified incorrectly!
        	System.out.println("Evaluator type specified incorrectly.");
        	// For now return DISCRETE type.
        	return EVALUATORTYPE.DISCRETE;
        	// TODO: Define corresponding exception.
        }
	}

}