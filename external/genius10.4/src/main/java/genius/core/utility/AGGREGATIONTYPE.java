package genius.core.utility;

public enum AGGREGATIONTYPE 
{ 
	SUM, MIN, MAX;
	
	public static AGGREGATIONTYPE getAggregationType(String type)
	{
		for (AGGREGATIONTYPE a : AGGREGATIONTYPE.values())
			if (a.toString().toLowerCase().equals(type.toLowerCase()))
				return a;
		// default
		return SUM; 		
	}
}
