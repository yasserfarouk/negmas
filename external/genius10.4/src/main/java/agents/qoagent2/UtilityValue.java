package agents.qoagent2;

/*****************************************************************
 * Class name: UtilityValue
 * Goal: This class holds utility values: the value itself (saved in
 * a string), the utility and the effect of time (saved in a double).
 ****************************************************************/
public class UtilityValue
{
	public String sValue;
	public double dUtility;
	public double dTimeEffect;
	
	//the constructor inits the value to an empty string, and the
	//utility and effect of time to 0.
	public UtilityValue()
	{
		sValue = "";
		dUtility = 0;
		dTimeEffect = 0;
	}
}