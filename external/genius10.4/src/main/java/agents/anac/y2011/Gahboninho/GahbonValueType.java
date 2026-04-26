package agents.anac.y2011.Gahboninho;

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