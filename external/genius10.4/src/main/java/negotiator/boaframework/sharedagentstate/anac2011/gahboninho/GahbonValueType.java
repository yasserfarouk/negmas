package negotiator.boaframework.sharedagentstate.anac2011.gahboninho;

import genius.core.issue.Issue;
import genius.core.issue.Value;

public interface GahbonValueType
{
	void INIT (Issue I);
	void UpdateImportance(Value OpponentBid /* value of this Issue as received from opponent*/ );	
	double GetNormalizedVariance ();
	
	int    GetUtilitiesCount (); // 
	
	double GetExpectedUtilityByValue (Value V);
}