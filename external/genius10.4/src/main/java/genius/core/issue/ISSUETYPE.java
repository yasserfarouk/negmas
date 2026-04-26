package genius.core.issue;

public enum ISSUETYPE {
	/** If the type of the issue is unknown. */
	UNKNOWN,
	/** If the issue is discrete. */
	DISCRETE,
	/** If the issue is represented by a linear function. */
	INTEGER, REAL, OBJECTIVE;

	public static ISSUETYPE convertToType(String typeString) {

		// If typeString is null for some reason (i.e. not spceified in the XML template
		// then we assume that we have DISCRETE type
		if(typeString==null) return ISSUETYPE.DISCRETE;
		else if (typeString.equalsIgnoreCase("integer"))
			return ISSUETYPE.INTEGER;
		else if (typeString.equalsIgnoreCase("real"))
			return ISSUETYPE.REAL;
		else if (typeString.equalsIgnoreCase("discrete"))
			return ISSUETYPE.DISCRETE;
		else {
			// Type specified incorrectly!
			System.out.println("Type specified incorrectly.");
			// For now return DISCRETE type.
			return ISSUETYPE.DISCRETE;
			// TODO: Define corresponding exception.
		}
	}	
}